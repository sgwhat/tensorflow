/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"

#include <cstring>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_common/device_id.h"
#include "tensorflow/core/common_runtime/device_common/device_id_manager.h"
#include "tensorflow/core/common_runtime/device_common/device_id_utils.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

PluggableDeviceProcessState* PluggableDeviceProcessState::singleton(
    const string& platform_name) {
  static std::unordered_map<string, PluggableDeviceProcessState*>
      process_state_map;
  auto iter = process_state_map.find(platform_name);
  if (iter != process_state_map.end()) return iter->second;
  process_state_map[platform_name] =
      new PluggableDeviceProcessState(platform_name);
  return process_state_map[platform_name];
}

PluggableDeviceProcessState::PluggableDeviceProcessState(
    const string& platform_name)
    : pluggable_device_enabled_(false), platform_name_(platform_name) {
  process_state_ = ProcessState::singleton();
}

int PluggableDeviceProcessState::BusIdForPluggableDevice(
    TfDeviceId tf_device_id) {
  // Return the NUMA node accociated with the PluggableDevice's StreamExecutor.
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  se::StreamExecutor* se =
      DeviceIdUtil::ExecutorForTfDeviceId(platform, tf_device_id).ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // bus_id must be non-negative. If the numa_node is unknown, use 0
  return numa_node >= 0 ? numa_node : 0;
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes) {
  CHECK(process_state_);
  const string& allocator_type = options.allocator_type();
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(
      PluggableDeviceMachineManager(platform_name_), tf_device_id);

  if (tf_device_id.value() >=
      static_cast<int64>(pluggable_device_allocators_.size())) {
    pluggable_device_allocators_.resize(tf_device_id.value() + 1);
  }

  AllocatorParts& allocator_parts =
      pluggable_device_allocators_[tf_device_id.value()];
  if (allocator_parts.allocator == nullptr) {
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(tf_device_id,
                                                      &platform_device_id));

    int bus_id = BusIdForPluggableDevice(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= pluggable_device_visitors_.size()) {
      pluggable_device_visitors_.push_back({});
    }

    DeviceMemAllocator* sub_allocator = new DeviceMemAllocator(
        DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_device_id)
            .ValueOrDie(),
        platform_device_id,
        (options.per_process_gpu_memory_fraction() > 1.0 ||
         options.experimental().use_unified_memory()),
        pluggable_device_visitors_[bus_id], {});

    PluggableDeviceBFCAllocator* device_bfc_allocator =
        new PluggableDeviceBFCAllocator(
            sub_allocator, total_bytes, options,
            strings::StrCat("PluggableDevice_", tf_device_id.value(), "_bfc"));
    Allocator* device_allocator = device_bfc_allocator;

    SharedCounter* timing_counter = nullptr;
    if (options.experimental().timestamped_allocator()) {
      timing_counter = new SharedCounter;
      device_bfc_allocator->SetTimingCounter(timing_counter);
    }

    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_device_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, device_allocator, md, &mu_);
    }

    allocator_parts = {std::unique_ptr<Allocator>(device_allocator),
                       std::unique_ptr<SharedCounter>(timing_counter),
                       device_bfc_allocator, sub_allocator,
                       std::unique_ptr<Allocator>(recording_allocator)};
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return allocator_parts.recording_allocator.get();
  } else {
    return allocator_parts.allocator.get();
  }
}

SharedCounter* PluggableDeviceProcessState::PluggableDeviceAllocatorCounter(
    TfDeviceId tf_device_id) {
  DCHECK(process_state_);
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  DeviceIdUtil::CheckValidTfDeviceId(platform, tf_device_id);
  mutex_lock l(mu_);
  if (tf_device_id.value() >=
      static_cast<int64>(pluggable_device_allocators_.size())) {
    LOG(ERROR) << "Asked for counter for PluggableDevice allocator "
               << tf_device_id.value() << " but only have "
               << pluggable_device_allocators_.size();
    return nullptr;
  }
  AllocatorParts& allocator_parts =
      pluggable_device_allocators_[tf_device_id.value()];
  if (allocator_parts.counter.get() == nullptr) {
    SharedCounter* timing_counter = new SharedCounter;
    allocator_parts.bfc_allocator->SetTimingCounter(timing_counter);
    allocator_parts.counter.reset(timing_counter);
  }
  return allocator_parts.counter.get();
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceHostAllocator(
    int numa_node) {
  CHECK(process_state_);
  return process_state_->GetCPUAllocator(
      numa_node);  // TODO: switch to dma copy
}

}  // namespace tensorflow
