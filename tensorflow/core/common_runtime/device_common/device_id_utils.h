/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_COMMON_DEVICE_ID_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_COMMON_DEVICE_ID_UTILS_H_

#include "tensorflow/core/common_runtime/device_common/device_id.h"
#include "tensorflow/core/common_runtime/device_common/device_id_manager.h"
#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// Utility methods for translation between Tensorflow Device ids and platform
// Device ids.
class DeviceIdUtil {
 public:
  // Convenient methods for getting the associated executor given a TfDeviceId
  // or PlatformDeviceId.
  static se::port::StatusOr<se::StreamExecutor*> ExecutorForPlatformDeviceId(
      se::Platform* device_manager, PlatformDeviceId platform_device_id) {
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }

  static se::port::StatusOr<se::StreamExecutor*> ExecutorForTfDeviceId(
      se::Platform* device_manager, TfDeviceId tf_device_id) {
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
        tf_device_id, &platform_device_id));
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }

  // Verify that the platform_device_id associated with a TfDeviceId is
  // legitimate.
  static void CheckValidTfDeviceId(se::Platform* device_manager,
                                   TfDeviceId tf_device_id) {
    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(tf_device_id,
                                                      &platform_device_id));
    const int visible_device_count = device_manager->VisibleDeviceCount();
    CHECK_LT(platform_device_id.value(), visible_device_count)
        << "platform_device_id is outside discovered device range."
        << " TF Device id: " << tf_device_id
        << " platform Device id: " << platform_device_id
        << " visible device count: " << visible_device_count;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_COMMON_DEVICE_ID_UTILS_H_
