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
"""DNNL_GRAPH TensorFlow axpy

"""

from tensorflow.core.protobuf import rewriter_config_pb2
import tf_dnnl_graph_bridge
import json
from tensorflow.python.client import timeline
import getpass
import ctypes

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Define the data
W = tf.compat.v1.placeholder(tf.float32, shape=(3, 2), name='W')
x = tf.compat.v1.placeholder(tf.float32, shape=(2, 3), name='x')

bias = tf.compat.v1.constant(1.0, shape=[3], name='bias')

matmul = tf.matmul(W, x)
matmul_bias = tf.nn.bias_add(matmul, bias)

# Configure the sessios, we will turn off the remapper graph optimizer
graph_options = tf.compat.v1.GraphOptions(
    rewrite_options=rewriter_config_pb2.RewriterConfig(
        remapping=rewriter_config_pb2.RewriterConfig.OFF))

# Configure the session
config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  graph_options=graph_options)
config_dnnl_graph_enabled = tf_dnnl_graph_bridge.update_config(config)


# Create session and run
with tf.compat.v1.Session(config=config_dnnl_graph_enabled) as sess:
    matmul_dnnl_graph = sess.run((matmul_bias),
                                 feed_dict={
        W: np.ones((3, 2)),
        x: np.ones((2, 3)),
    })

    print(matmul_dnnl_graph)
