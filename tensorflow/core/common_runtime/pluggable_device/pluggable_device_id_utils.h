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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLEDEVICE_PLUGGABLEDEVICE_ID_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLEDEVICE_PLUGGABLEDEVICE_ID_UTILS_H_

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id_manager.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// Utility methods for translation between Tensorflow PluggableDevice ids
// and platform PluggableDevice ids

class PluggableDeviceIdUtil {
 public:
  // Convenient methods for getting the associated executor given a
  // TfPluggableDeviceId or PlatformPluggableDeviceId
  static se::port::StatusOr<se::StreamExecutor*>
  ExecutorForPlatformPluggableDeviceId(
      se::Platform* platform,
      PlatformPluggableDeviceId platform_pluggabledevice_id) {
    return platform->ExecutorForDevice(platform_pluggabledevice_id.value());
  }

  static se::port::StatusOr<se::StreamExecutor*> ExecutorForTfPluggableDeviceId(
      se::Platform* platform, TfPluggableDeviceId tf_pluggabledevice_id) {
    PlatformPluggableDeviceId platform_pluggabledevice_id;
    TF_RETURN_IF_ERROR(PluggableDeviceIdManager::TfToPlatformPluggableDeviceId(
        tf_pluggabledevice_id, &platform_pluggabledevice_id));
    return platform->ExecutorForDevice(platform_pluggabledevice_id.value());
  }

  // Verity that the platform_pluggabledevice_id is associated with a
  // TfPluggableDeviceId is legitimate.
  static void CheckValidTfPluggableDeviceId(
      se::Platform* platform, TfPluggableDeviceId tf_pluggabledevice_id) {
    PlatformPluggableDeviceId platform_pluggabledevice_id;
    TF_CHECK_OK(PluggableDeviceIdManager::TfToPlatformPluggableDeviceId(
        tf_pluggabledevice_id, &platform_pluggabledevice_id));
    const int visible_device_count = platform->VisibleDeviceCount();
    CHECK_LT(platform_pluggabledevice_id.value(), visible_device_count)
        << "platform_pluggabledevice_id is outside discovered device range."
        << " TF PluggableDevice id: " << tf_pluggabledevice_id
        << " platform PluggableDevice id" << platform_pluggabledevice_id
        << " visible device count: " << visible_device_count;
  }
};

}  // namespace tensorflow

#endif
