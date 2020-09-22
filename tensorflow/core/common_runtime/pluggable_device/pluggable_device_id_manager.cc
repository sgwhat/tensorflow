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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id_manager.h"
#include <unordered_map>
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {
// Manages the map between TfPluggableDeviceId to platform PluggableDevice id.
class TfToPlatformPluggableDeviceIdMap {
 public:
  static TfToPlatformPluggableDeviceIdMap* singleton() {
    static auto* id_map = new TfToPlatformPluggableDeviceIdMap;
    return id_map;
  }

  Status Insert(TfPluggableDeviceId tf_device_id,
                PlatformPluggableDeviceId platform_device_id)
      TF_LOCKS_EXCLUDED(mu_) {
    std::pair<IdMapType::iterator, bool> result;
    {
      mutex_lock lock(mu_);
      result =
          id_map_.insert({tf_device_id.value(), platform_device_id.value()});
    }
    if (!result.second && platform_device_id.value() != result.first->second) {
      return errors::AlreadyExists(
          "TensorFlow device (PluggableDevice:", tf_device_id.value(),
          ") is being mapped to "
          "multiple Plugged devices (",
          platform_device_id.value(), " now, and ", result.first->second,
          " previously), which is not supported. "
          "This may be the result of providing different PluggableDevice "
          "configurations "
          "(ConfigProto.gpu_options, for example different visible_device_list)"
          " when creating multiple Sessions in the same process. This is not "
          " currently supported, see "
          "https://github.com/tensorflow/tensorflow/issues/19083");
    }
  }

  bool Find(TfPluggableDeviceId tf_device_id,
            PlatformPluggableDeviceId* platform_device_id) const
      TF_LOCKS_EXCLUDED(mu_) {
    // TODO(mrry): Consider replacing this with an atomic `is_initialized` bit,
    // to avoid writing to a shared cache line in the tf_shared_lock.
    tf_shared_lock lock(mu_);
    auto result = id_map_.find(tf_device_id.value());
    if (result == id_map_.end()) return false;
    *platform_device_id = result->second;
    return true;
  }

 private:
  TfToPlatformPluggableDeviceIdMap() = default;

  void TestOnlyReset() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    id_map_.clear();
  }

  using IdMapType = std::unordered_map<int32, int32>;
  mutable mutex mu_;
  IdMapType id_map_ TF_GUARDED_BY(mu_);

  friend class ::tensorflow::PluggableDeviceIdManager;
  TF_DISALLOW_COPY_AND_ASSIGN(TfToPlatformPluggableDeviceIdMap);
};
}  // namespace

Status PluggableDeviceIdManager::InsertTfPlatformPluggableDeviceIdPair(
    TfPluggableDeviceId tf_device_id,
    PlatformPluggableDeviceId platform_device_id) {
  return TfToPlatformPluggableDeviceIdMap::singleton()->Insert(
      tf_device_id, platform_device_id);
}

Status PluggableDeviceIdManager::TfToPlatformPluggableDeviceId(
    TfPluggableDeviceId tf_device_id,
    PlatformPluggableDeviceId* platform_device_id) {
  if (TfToPlatformPluggableDeviceIdMap::singleton()->Find(tf_device_id,
                                                          platform_device_id)) {
    return Status::OK();
  }
  return errors::NotFound("TensorFlow device PluggableDevice:",
                          tf_device_id.value(), " was not registered");
}

void PluggableDeviceIdManager::TestOnlyReset() {
  TfToPlatformPluggableDeviceIdMap::singleton()->TestOnlyReset();
}

}  // namespace tensorflow
