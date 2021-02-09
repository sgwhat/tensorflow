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

#ifndef TF_DNNL_GRAPH_BRIDGE_REPLACE_FUSIONS_H_
#define TF_DNNL_GRAPH_BRIDGE_REPLACE_FUSIONS_H_

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "log.h"
#include "partitioner.h"

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

class TfLlgaPartitioner;

class LlgaFuser {
 public:
  LlgaFuser(Graph* g);
  // Perform the actual graph surgery
  Status RewritePass(TfLlgaPartitioner& tf_dnnl_graph_partitioner);

 private:
  Graph* graph;

  // boolean to indicate that rewrite is done;
  bool rewrite_done_;

  struct fused_op_input_info {
    Node* dst_node;
    int src_node_id;
    int output_slot_number;
    DataType dt;
  };

  std::map<std::tuple<int, int>, int> output_remap_map_;

  // A map from partition indices to a vector of input data types.
  std::map<int, std::vector<fused_op_input_info>> partition_input_map_;
  // A map from partition indices to a vector of output data types.
  std::map<int, std::vector<DataType>> partition_output_dt_map_;
  std::map<int, std::vector<int64>> output_node_ids_;

  // A map from partition indices to the expected device name for nodes
  // in that partition.
  std::map<int, std::string> device_name_map_;
  // A map from partition indices to corresponding LlgaFused nodes.
  std::map<int, Node*> partition_node_map_;

  std::unordered_set<int64> unique_output_node_ids_;

  std::vector<int64> unique_output_node_ids_layout_any;
};

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow

#endif  // TF_DNNL_GRAPH_BRIDGE_REPLACE_FUSIONS_H_
