# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import ctypes
from tensorflow.python.framework import load_library
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import errors_impl
from tensorflow.python import pywrap_tensorflow as py_tf
import importlib
import os
import sys
import time
import getpass
from platform import system

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


__all__ = [
    'enable', 'disable', 'is_enabled', 'backends_len', 'list_backends',
    'set_backend', 'get_currently_set_backend_name',
    'start_logging_placement', 'stop_logging_placement',
    'is_logging_placement', '__version__', 'cxx11_abi_flag'
    'is_grappler_enabled', 'update_config', 'are_variables_enabled',
    'set_disabled_ops', 'get_disabled_ops',
]

TF_VERSION = tf.version.VERSION
TF_GIT_VERSION = tf.version.GIT_VERSION
TF_VERSION_NEEDED = "${TensorFlow_VERSION}"
TF_GIT_VERSION_BUILT_WITH = "${TensorFlow_GIT_VERSION}"

# converting version representations to strings if not already
try:
    TF_VERSION = str(TF_VERSION, 'ascii')
except TypeError:  # will happen for python 2 or if already string
    pass

try:
    TF_VERSION_NEEDED = str(TF_VERSION_NEEDED, 'ascii')
except TypeError:
    pass

try:
    if TF_GIT_VERSION.startswith("b'"):  # TF version can be a bytes __repr__()
        TF_GIT_VERSION = eval(TF_GIT_VERSION)
    TF_GIT_VERSION = str(TF_GIT_VERSION, 'ascii')
except TypeError:
    pass

try:
    if TF_GIT_VERSION_BUILT_WITH.startswith("b'"):
        TF_GIT_VERSION_BUILT_WITH = eval(TF_GIT_VERSION_BUILT_WITH)
    TF_GIT_VERSION_BUILT_WITH = str(TF_GIT_VERSION_BUILT_WITH, 'ascii')
except TypeError:
    pass

TF_INSTALLED_VER = TF_VERSION.split('.')
TF_NEEDED_VER = TF_VERSION_NEEDED.split('.')

libpath = os.path.dirname(__file__)

full_lib_path = os.path.join(libpath, 'libdnnl_graph.so.0.1')
print(full_lib_path)
_ = load_library.load_op_library(full_lib_path)

full_lib_path = os.path.join(libpath, 'libtf_dnnl_graph_bridge.so')
print(full_lib_path)
_ = load_library.load_op_library(full_lib_path)
tf_dnnl_graph_bridge_lib = ctypes.cdll.LoadLibrary(full_lib_path)
tf_dnnl_graph_bridge_lib.TfLlgaIsGrapplerEnabled.restype = ctypes.c_bool

# raise exception check if tf ABI and tf_dnnl_graph ABI version are not same
if (tf_dnnl_graph_bridge_lib.TfLlgaCxx11AbiFlag() != tf.__cxx11_abi_flag__):
    raise Error("TensorFlow Cxx11 ABI is not same as TFLlgaCxx11 ABI")


def is_grappler_enabled():
    return tf_dnnl_graph_bridge_lib.TfLlgaIsGrapplerEnabled()


def update_config(config, backend_name="CPU", device_id=""):
    # updating session config if grappler is enabled
    if(tf_dnnl_graph_bridge_lib.TfLlgaIsGrapplerEnabled()):
        opt_name = 'dnnl_graph-partitioner'
        # If the config already has dnnl_graph-partitioner, then do not update it
        if config.HasField('graph_options'):
            if config.graph_options.HasField('rewrite_options'):
                custom_opts = config.graph_options.rewrite_options.custom_optimizers
                for i in range(len(custom_opts)):
                    if custom_opts[i].name == opt_name:
                        return config
        rewriter_options = rewriter_config_pb2.RewriterConfig()
        rewriter_options.meta_optimizer_iterations = (
            rewriter_config_pb2.RewriterConfig.ONE)
        rewriter_options.min_graph_nodes = -1
        dnnl_graph_optimizer = rewriter_options.custom_optimizers.add()
        dnnl_graph_optimizer.name = opt_name
        dnnl_graph_optimizer.parameter_map["dnnl_graph_backend"].s = backend_name.encode(
        )
        dnnl_graph_optimizer.parameter_map["device_id"].s = device_id.encode()
        config.MergeFrom(tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(rewrite_options=rewriter_options)))
    return config


__version__ = \
    "TensorFlow version used for this build: " + TF_GIT_VERSION_BUILT_WITH + "\n" \
    "CXX11_ABI flag used for this build: " + str(tf_dnnl_graph_bridge_lib.TfLlgaCxx11AbiFlag()) + "\n" \
    "dnnl_graph bridge built with Grappler: " + str(tf_dnnl_graph_bridge_lib.TfLlgaIsGrapplerEnabled()) + "\n" \
    "dnnl_graph bridge built with Variables and Optimizers Enablement: " + \
    str(tf_dnnl_graph_bridge_lib.TfLlgaAreVariablesEnabled())
