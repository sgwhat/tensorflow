
/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef TF_DNNL_GRAPH_BRIDGE_UTILS_H_
#define TF_DNNL_GRAPH_BRIDGE_UTILS_H_

#include <tensorflow/core/common_runtime/process_state.h>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "graph_visualize.h"

// Define all typedefs here
namespace tensorflow {

namespace tf_dnnl_graph_bridge {
using dnnl_graph_data_type_t = dnnl::graph::logical_tensor::data_type;
using dnnl_graph_shape_t = std::vector<int64_t>;
using dnnl_graph_shape_t = std::vector<int64_t>;
}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

Status GetPartitionId(const Node* node, int* partition_id);

tensorflow::Allocator* GetTfAllocator();

void DumpGraphs(Graph& graph, int idx, std::string filename_prefix,
                std::string title);

// TODO(nbpatel): Check this
static int graph_file_idx = 0;

dnnl_graph_data_type_t GetLlgaDataType(tensorflow::DataType dt);

// DNNL_GRAPH uses uint64_t for node ids, TF uses int
uint64_t LlgaNodeId(int tf_node_id);
int TfNodeId(uint64_t dnnl_graph_node_id);

// DNNL_GRAPH uses uint64_t for tensor ids that encode a node_id and an index
// TensorFlow attributes must use int64 (no unsigned)
uint64_t LlgaTensorId(int tf_node_id, int tf_index);
uint64_t LlgaTensorId(int64 tf_tensor_id);
int64 TfTensorId(uint64_t dnnl_graph_tensor_id);

int GetTfNodeId(uint64_t dnnl_graph_tensor_id);
int GetTfIndex(uint64_t dnnl_graph_tensor_id);

std::vector<int64_t> get_tensor_shape(TensorShape& tf_shape);

void ExtractSpatialDims(bool is_channel_last, const std::vector<int32_t>& src,
                        std::vector<int64_t>* dst);

template <typename T>
bool Contains(std::vector<T> vec, T value) {
  return (std::find(vec.begin(), vec.end(), value) != vec.end());
}

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
#endif  // TF_DNNL_GRAPH_BRIDGE_UTILS_H_