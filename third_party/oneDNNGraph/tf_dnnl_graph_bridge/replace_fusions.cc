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

#include "replace_fusions.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "partitioner.h"
#include "utils.h"

using namespace std;

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

//
// For each Partition K in the input graph, the LlgaFuser takes the set
// of all nodes in K and replaces them with a single LlgaFused op that
// stands in for the internal subgraph represented by the partition K.
//
LlgaFuser::LlgaFuser(Graph* g) : graph(g), rewrite_done_(false) {}

Status LlgaFuser::RewritePass(TfLlgaPartitioner& tf_dnnl_graph_partitioner) {
  DNNL_GRAPH_VLOG(0) << " IN RewritePass ";

  if (rewrite_done_) {
    return errors::Internal("In LlgaFuser, called RewritePass more than once");
  }

  auto partitioned_nodes = tf_dnnl_graph_partitioner.GetPartitionedNodes();

  // Step 1: Find all nodes that are feeding into/out of each partition, and
  // add inputs for them to the corresponding FunctionDef(s).
  for (auto node : partitioned_nodes) {
    // If the destination node lies within a partition, we must create an input
    // for the source node to the destination partition. For the moment we will
    // just store this fact in the partition_input_map_.
    for (auto edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }

      Node* src = edge->src();
      Node* dst = node;

      int src_partition_id;
      int dst_partition_id;
      GetPartitionId(dst, &dst_partition_id);
      GetPartitionId(src, &src_partition_id);

      // Check if its not a sink or source node also, Ignore edges within a
      // partition.
      if (!src->IsOp() || (dst_partition_id == src_partition_id)) {
        continue;
      }

      partition_input_map_[dst_partition_id].push_back(
          {dst, src->id(), edge->src_output(),
           dst->input_type(edge->dst_input())});

      auto it = device_name_map_.find(dst_partition_id);
      if (it == device_name_map_.end()) {
        device_name_map_[dst_partition_id] = dst->assigned_device_name();
      }
    }

    // If the source node lies within a partition, we must create an output for
    // it from the source partition. For the moment we will just store this
    // fact in the output_remap_map_.
    for (auto edge : node->out_edges()) {
      Node* src = node;
      Node* dst = edge->dst();

      DataType dt = dst->input_type(edge->dst_input());
      int src_partition_id;
      int dst_partition_id;
      GetPartitionId(dst, &dst_partition_id);
      GetPartitionId(src, &src_partition_id);

      // Check if its not a sink or source node also, Ignore edges within a
      // partition.
      if (!dst->IsOp() || dst_partition_id == src_partition_id) {
        continue;
      }

      uint64_t unique_logical_tensor_id =
          LlgaTensorId(src->id(), edge->src_output());
      if (unique_output_node_ids_.find(unique_logical_tensor_id) !=
          unique_output_node_ids_.end()) {
        // This if statement is for layout propagation. If it is true and if it
        // has one edge going to dnnl_graph partition and another to a non
        // dnnl_graph node and if its already added to the set of nodes with
        // "layout_type::any" then remove it.

        if (dst_partition_id == -1 &&
            Contains<int64>(unique_output_node_ids_layout_any,
                            unique_logical_tensor_id)) {
          unique_output_node_ids_layout_any.erase(
              std::remove(unique_output_node_ids_layout_any.begin(),
                          unique_output_node_ids_layout_any.end(),
                          unique_logical_tensor_id),
              unique_output_node_ids_layout_any.end());
        }

        continue;
      }
      unique_output_node_ids_.insert(unique_logical_tensor_id);
      output_node_ids_[src_partition_id].push_back(
          TfTensorId(unique_logical_tensor_id));

      output_remap_map_[std::make_tuple(src->id(), edge->src_output())] =
          static_cast<int>(partition_output_dt_map_[src_partition_id].size());

      partition_output_dt_map_[src_partition_id].push_back(dt);

      // For Layout Propagation
      if (dst_partition_id != -1) {
        DNNL_GRAPH_VLOG(0) << "OUTPUT FEEDS TO ANOTHER PARTITION  -- OUTPUT ID "
                           << unique_logical_tensor_id;
        unique_output_node_ids_layout_any.push_back(unique_logical_tensor_id);
      }
    }
  }

  DNNL_GRAPH_VLOG(0) << " Step 1 successful ";

  // Step 2: Create fused ops for all partitions.
  for (auto node : partitioned_nodes) {
    int partition_id;
    GetPartitionId(node, &partition_id);

    if (partition_node_map_.find(partition_id) != partition_node_map_.end()) {
      continue;
    }

    std::stringstream ss;
    ss << "dnnl_graph_fused_op_" << partition_id;

    string fused_op_name = ss.str();
    std::vector<DataType> input_types;
    std::vector<NodeBuilder::NodeOut> inputs;
    std::vector<int64> input_node_ids;
    std::map<int64, bool> is_node_visited;
    for (auto& fused_op_inputs : partition_input_map_[partition_id]) {
      input_types.push_back(fused_op_inputs.dt);
      inputs.push_back(
          NodeBuilder::NodeOut(graph->FindNodeId(fused_op_inputs.src_node_id),
                               fused_op_inputs.output_slot_number));
      int64_t unique_logical_tensor_id = LlgaTensorId(
          fused_op_inputs.src_node_id, fused_op_inputs.output_slot_number);
      input_node_ids.push_back(TfTensorId(unique_logical_tensor_id));
    }

    gtl::ArraySlice<int64> input_slice(input_node_ids);
    gtl::ArraySlice<int64> output_slice(output_node_ids_[partition_id]);
    gtl::ArraySlice<int64> output_ids_with_layout_any(
        unique_output_node_ids_layout_any);

    Node* n;
    NodeBuilder nb =
        NodeBuilder(fused_op_name, "LlgaFused")
            .Attr("dnnl_graph_partition", partition_id)
            .Attr("dnnl_graph_backend", "CPU")
            .Attr("input_node_ids", input_slice)
            .Attr("output_node_ids", output_slice)
            .Attr("output_ids_with_layout_any", output_ids_with_layout_any)
            .Attr("Targuments", input_types)
            .Attr("Tresults", partition_output_dt_map_[partition_id])
            .Device(device_name_map_[partition_id])
            .Input(inputs);

    Status status = nb.Finalize(graph, &n);
    TF_RETURN_IF_ERROR(status);

    partition_node_map_[partition_id] = n;
  }
  DNNL_GRAPH_VLOG(0) << " Step 2 successful ";

  // Step 3: Remap all non-partitioned inputs that are reading from
  // dnnl_graph_fused_op edges, and all control edges that cross partition
  // boundaries.

  for (auto node : partitioned_nodes) {
    // Copy the edge pointers, so as not to invalidate the iterator.
    std::vector<const Edge*> edges;
    for (auto edge : node->out_edges()) {
      edges.push_back(edge);
    }
    for (auto edge : edges) {
      Node* src = node;
      Node* dst = edge->dst();

      int src_partition_id;
      int dst_partition_id;
      GetPartitionId(dst, &dst_partition_id);
      GetPartitionId(src, &src_partition_id);

      // Ignore edges within a partition.
      if (dst_partition_id == src_partition_id) {
        continue;
      }
      bool dst_is_partitioned = (dst_partition_id != -1);

      if (edge->IsControlEdge()) {
        graph->RemoveControlEdge(edge);
        graph->AddControlEdge(
            partition_node_map_[src_partition_id],
            (dst_is_partitioned ? partition_node_map_[dst_partition_id]
                                : edge->dst()));
      } else {
        if (dst_is_partitioned) {
          continue;
        }
        auto it = output_remap_map_.find(
            std::make_tuple(edge->src()->id(), edge->src_output()));

        if (it == output_remap_map_.end()) {
          continue;
        }

        int partition_output = it->second;

        Status status =
            graph->UpdateEdge(partition_node_map_[src_partition_id],
                              partition_output, edge->dst(), edge->dst_input());

        TF_RETURN_IF_ERROR(status);
      }
    }
  }
  DNNL_GRAPH_VLOG(0) << " Step 3 successful ";

  // Setp 4: Remove clustered nodes from the graph.
  for (auto node : partitioned_nodes) {
    graph->RemoveNode(node);
  }

  DNNL_GRAPH_VLOG(0) << " Step 4 successful ";

  rewrite_done_ = true;
  return Status::OK();
}
}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow