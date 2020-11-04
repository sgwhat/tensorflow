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

#include "tensorflow/core/common_runtime/device_common/device_id_manager.h"

#include "tensorflow/core/common_runtime/device_common/device_id.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PlatformDeviceId TfToPlatformDeviceId(TfDeviceId tf) {
  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(tf, &platform_device_id));
  return platform_device_id;
}

TEST(DeviceIdManagerTest, Basics) {
  TfDeviceId key_0(0);
  PlatformDeviceId value_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(key_0, value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(key_0));

  // Multiple calls to map the same value is ok.
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(key_0, value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(key_0));

  // Map a different TfDeviceId to a different value.
  TfDeviceId key_1(3);
  PlatformDeviceId value_1(2);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(key_1, value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(key_1));

  // Mapping a different TfDeviceId to the same value is ok.
  TfDeviceId key_2(10);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(key_2, value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(key_2));

  // Mapping the same TfDeviceId to a different value.
  ASSERT_FALSE(
      DeviceIdManager::InsertTfPlatformDeviceIdPair(key_2, value_0).ok());

  // Getting a nonexistent mapping.
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId(TfDeviceId(100), &value_0).ok());
}

}  // namespace
}  // namespace tensorflow
