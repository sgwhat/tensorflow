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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLEDEVICE_PLUGGABLEDEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLEDEVICE_PLUGGABLEDEVICE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id_manager.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id_utils.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class PluggableDevice : public LocalDevice {
 public:
  PluggableDevice(const SessionOptions& options, const std::string& name,
                  const std::string& device_type,
                  const std::string& subdevice_type, Bytes memory_limit,
                  const DeviceLocality& locality,
                  TfPluggableDeviceId tf_device_id,
                  const std::string& physical_device_desc,
                  Allocator* device_allocator, Allocator* cpu_allocator,
                  bool sync_every_op);
  ~PluggableDevice() override;
  // Initialize the device and return the status of initialization.
  Status Init(const SessionOptions& options);

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  Status Sync() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override;

  // The executor that provides control for the device; e.g., for CUDA this
  // corresponds to the cuda context.
  se::StreamExecutor* executor() const { return executor_; }

 private:
  Allocator* device_allocator_;
  Allocator* cpu_allocator_;

  se::StreamExecutor* executor_ = nullptr;
  struct StreamGroup {
    se::Stream* compute = nullptr;
    se::Stream* host_to_device = nullptr;
    se::Stream* device_to_host = nullptr;
    gtl::InlinedVector<se::Stream*, 4> device_to_device;
  };

  class StreamGroupFactory;

  StreamGroup* stream_;
  PluggableDeviceContext* device_context_;
  GpuDeviceInfo* pluggable_device_info_ = nullptr;
  TfPluggableDeviceId tf_device_id_;
  const bool sync_every_op_ = false;
  EventMgr* em_ = nullptr;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  bool timestamped_allocator_ = false;
  bool force_gpu_compatible_ = false;
  std::string ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                         const int& stream_id);

  // This method returns an initialization status, in addition to
  // calling the "done" Statuscallback, if there is a failure to
  // allocate memory or if the tensor "from" is not DMA-copyable.
  // If there is no error prior to enqueueing the copy, an OK status
  // is returned.
  Status MaybeCopyTensorToPluggableDevice(const AllocatorAttributes& allc_attrs,
                                          const Tensor& from, Tensor* to,
                                          StatusCallback done);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLEDEVICE_PLUGGABLEDEVICE_H_
