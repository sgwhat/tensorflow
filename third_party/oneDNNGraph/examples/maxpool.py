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
"""
    DNNL_GRAPH TensorFlow Conv + BatchNorm + Relu 
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


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


input = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 4, 1), name='input')
max_pool = max_pool_2x2(input)

# Configure the sessios, we will turn off the remapper graph optimizer
graph_options = tf.compat.v1.GraphOptions(
    rewrite_options=rewriter_config_pb2.RewriterConfig(
        remapping=rewriter_config_pb2.RewriterConfig.OFF))

config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  graph_options=graph_options)

config_dnnl_graph_enabled = tf_dnnl_graph_bridge.update_config(config)

# Create session and run
with tf.compat.v1.Session(config=config_dnnl_graph_enabled) as sess:
    print("Python: Running with Session")
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    max_pool_output = sess.run(max_pool,
                               feed_dict={
                                   input: np.ones(((1, 3, 4, 1)))
                               },
                               options=options,
                               run_metadata=run_metadata)
    print(max_pool_output)
