load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
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
    name = "pluggable_device_id",
    hdrs = [
        "pluggable_device_id.h",
        "pluggable_device_id_manager.h",
    ],
    deps = [
        ":pluggable_device_id_impl",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "pluggable_device_id_impl",
    srcs = ["pluggable_device_id_manager.cc"],
    hdrs = [
        "pluggable_device_id.h",
        "pluggable_device_id_manager.h",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
)

filegroup(
    name = "pluggable_device_runtime_headers",
    srcs = [
        "pluggable_device.h",
        "pluggable_device_bfc_allocator.h",
        "pluggable_device_context.h",
        "pluggable_device_factory.h",
        "pluggable_device_host_allocator.h",
        "pluggable_device_id.h",
        "pluggable_device_id_manager.h",
        "pluggable_device_id_utils.h",
        "pluggable_device_init.h",
        "pluggable_device_process_state.h",
        "pluggable_device_util.h",
        "//tensorflow/core/common_runtime/gpu:gpu_event_mgr.h",
    ],
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "pluggable_device_runtime_impl",
    srcs = [
        "pluggable_device.cc",
        "pluggable_device_context.cc",
        "pluggable_device_factory.cc",
        "pluggable_device_process_state.cc",
        "pluggable_device_util.cc",
    ],
    hdrs = [":pluggable_device_runtime_headers"],
    copts = tf_copts(),
    deps = [
        ":pluggable_device_bfc_allocator",
        ":pluggable_device_id_impl",
        ":pluggable_device_init_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/common_runtime/gpu:gpu_lib",
        "//tensorflow/core/platform:tensor_float_32_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pluggable_device_plugin_init",
    srcs = [
        "pluggable_device_plugin_init.cc",
    ],
    hdrs = [
        "pluggable_device_plugin_init.h",
        ":pluggable_device_runtime_headers",
        "//tensorflow/c:headers",
        "//tensorflow/core/common_runtime:core_cpu_base_headers",
        "//tensorflow/core/public:session_options.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
    ],
)

exports_files(
    srcs = [
        "pluggable_device_plugin_init.h",
    ],
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "pluggable_device_runtime",
    hdrs = [":pluggable_device_runtime_headers"],
    linkstatic = 1,
    deps = [
        ":pluggable_device_runtime_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
    ],
)

# This is redundant with the "gpu_runtime_*" targets above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).

tf_cuda_library(
    name = "pluggable_device_bfc_allocator",
    srcs = [
        "pluggable_device_bfc_allocator.cc",
    ],
    hdrs = ["pluggable_device_bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":pluggable_device_mem_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:bfc_allocator",
    ],
)

tf_cuda_library(
    name = "pluggable_device_mem_allocator",
    srcs = [
        "pluggable_device_id.h",
    ],
    hdrs = [
        "pluggable_device_host_allocator.h",
        "pluggable_device_mem_allocator.h",
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

tf_cuda_library(
    name = "pluggable_device_init",
    hdrs = [
        "pluggable_device_init.h",
    ],
    deps = [
        "pluggable_device_init_impl",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
    ],
)

tf_cuda_library(
    name = "pluggable_device_init_impl",
    srcs = [
        "pluggable_device_init.cc",
    ],
    hdrs = [
        "pluggable_device_init.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor",
    ],
    alwayslink = 1,
)
