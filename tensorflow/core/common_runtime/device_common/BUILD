load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "device_id",
    hdrs = [
        "device_id.h",
        "device_id_manager.h",
        "device_id_utils.h",
    ],
    deps = [
        ":device_id_impl",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "device_id_impl",
    srcs = ["device_id_manager.cc"],
    hdrs = [
        "device_id.h",
        "device_id_manager.h",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cuda_library(
    name = "device_mem_allocator",
    srcs = [
        "device_id.h",
    ],
    hdrs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/framework:allocator",
    ],
)

# -----------------------------------------------------------------------------
# Libraries with device facilities that are useful for writing kernels.
cc_library(
    name = "device_lib",
    srcs = [
        "device_event_mgr.cc",
    ],
    hdrs = [
        "device_event_mgr.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
    ],
)

cc_library(
    name = "device_headers_lib",
    hdrs = [
        "device_event_mgr.h",
    ],
)
