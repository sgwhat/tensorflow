# Description:
# StreamExecutor C API.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_protos_grappler",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "grappler_hdrs",
    hdrs = ["grappler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "grappler",
    srcs = ["grappler.cc"],
    hdrs = [
        "grappler.h",
        "grappler_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/platform:logging",
    ],
)

exports_files(
    srcs = [
        "grappler.h",
        "grappler_internal.h",
    ],
    visibility = ["//visibility:public"],
)
