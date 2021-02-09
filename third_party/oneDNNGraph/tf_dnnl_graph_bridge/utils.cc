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

#include "utils.h"

using namespace std;

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

Status GetPartitionId(const Node* node, int* partition_id) {
  Status s = GetNodeAttr(node->attrs(), "_partition_id", partition_id);
  if (s != Status::OK()) {
    *partition_id = -1;
  }
  return s;
}

tensorflow::Allocator* GetTfAllocator() {
  static tensorflow::Allocator* allocator =
      tensorflow::ProcessState::singleton()->GetCPUAllocator(0);
  return allocator;
}

void DumpGraphs(Graph& graph, int idx, std::string filename_prefix,
                std::string title) {
  // If we have a "main" graph, dump that.
  auto dot_filename = DotFilename(filename_prefix, idx);
  auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);

  GraphToDotFile(&graph, dot_filename, title);
  GraphToPbTextFile(&graph, pbtxt_filename);
}

// Utility function which maps TF Dtype to DNNL_GRAPH Dtype
dnnl_graph_data_type_t GetLlgaDataType(tensorflow::DataType dt) {
  // TODO (pruthvi): Populate the list once more data types are
  // supported by the dnnl_graph
  switch (dt) {
    case DT_FLOAT:
      return dnnl_graph_data_type_t::f32;
    case DT_HALF:
      return dnnl_graph_data_type_t::f16;
    case DT_BFLOAT16:
      return dnnl_graph_data_type_t::bf16;
    case DT_INT32:
      return dnnl_graph_data_type_t::s32;
    case DT_INT8:
      return dnnl_graph_data_type_t::s8;
    case DT_UINT8:
      return dnnl_graph_data_type_t::u8;
    default:
      return dnnl_graph_data_type_t::undef;
  }
}

uint64_t LlgaNodeId(int tf_node_id) {
  int64_t node_id = tf_node_id;
  return *reinterpret_cast<uint64_t*>(&node_id);
}
int TfNodeId(uint64_t dnnl_graph_node_id) {
  return static_cast<int>(*reinterpret_cast<int64_t*>(&dnnl_graph_node_id));
}

uint64_t LlgaTensorId(int tf_node_id, int tf_index) {
  return static_cast<int64_t>(tf_index) << 32 | (tf_node_id & 0xFFFFFFFF);
}

uint64_t LlgaTensorId(int64 tf_tensor_id) {
  return *reinterpret_cast<uint64_t*>(&tf_tensor_id);
}
int64 TfTensorId(uint64_t dnnl_graph_tensor_id) {
  return *reinterpret_cast<int64*>(&dnnl_graph_tensor_id);
}

int GetTfNodeId(uint64_t dnnl_graph_tensor_id) {
  unsigned int unode_id = dnnl_graph_tensor_id & 0xFFFFFFFF;
  return *reinterpret_cast<int*>(&unode_id);
}

int GetTfIndex(uint64_t dnnl_graph_tensor_id) {
  unsigned int uindex = dnnl_graph_tensor_id >> 32;
  return *reinterpret_cast<int*>(&uindex);
}

std::vector<int64_t> get_tensor_shape(TensorShape& tf_shape) {
  std::vector<int64_t> shape;
  int num_dimensions = tf_shape.dims();
  for (int i = 0; i < num_dimensions; i++) {
    shape.push_back(tf_shape.dim_size(i));
  }
  return shape;
}

void ExtractSpatialDims(bool is_channel_last, const std::vector<int32_t>& src,
                        std::vector<int64_t>* dst) {
  if (is_channel_last) {
    dst->at(0) = src[1];
    dst->at(1) = src[2];
  } else {
    dst->at(0) = src[2];
    dst->at(1) = src[3];
  }
}

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
