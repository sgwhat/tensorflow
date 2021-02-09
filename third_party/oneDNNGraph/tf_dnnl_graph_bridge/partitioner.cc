/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "partitioner.h"

#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

#include "add_identityn.h"
#include "allocator.h"
#include "graph_visualize.h"
#include "log.h"
#include "partitioner.h"
#include "replace_fusions.h"
#include "translate_op.h"
#include "utils.h"

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

std::map<int, dnnl::graph::partition>
    TfLlgaPartitioner::partition_id_to_dnnl_graph_partition_;

Status TfLlgaPartitioner::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

Node* TfLlgaPartitioner::GetTFNodeFromLlgaId(uint64_t id) {
  auto search = dnnl_graph_id_selected_node_map_.find(id);
  if (search == dnnl_graph_id_selected_node_map_.end()) return nullptr;
  return search->second;
}

Status TfLlgaPartitioner::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item, GraphDef* output) {
  DNNL_GRAPH_VLOG(0) << " IN GRAPPLER OPTIMIZER ";
  // Convert the GraphDef to Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph));

  // Fetch Nodes
  std::set<string> fetch_nodes;
  for (const string& f : item.fetch) {
    int pos = f.find(":");
    fetch_nodes.insert(f.substr(0, pos));
  }

  // Rewrite graph to add IdentityN node so the fetch node can be encapsulated
  // as well
  // If the fetch node in question has 0 outputs or any of the outputs
  // has ref type as a data type then don't add IdentityN node
  TF_RETURN_IF_ERROR(AddIdentityN(&graph, fetch_nodes));

  if (tf_dnnl_graph_bridge::DumpAllGraphs()) {
    DumpGraphs(graph, tf_dnnl_graph_bridge::graph_file_idx++,
               "tf_unmodified_graph", "UnModified");
    DNNL_GRAPH_VLOG(0) << " Dumping unmodified TF graph";
  }
  // We will visit ops in topological order.
  std::vector<Node*> ordered;
  GetReversePostOrder(graph, &ordered, NodeComparatorName());

  //
  // Now create the dnnl_graph ops from TensorFlow ops.
  //
  // TODO(nbpatel): Check this. TF would not want us to manage resources.
  // engine.set_allocator(allocator.GetLlgaAllocator());
  dnnl::graph::graph graph_ctx(dnnl::graph::engine::kind::cpu);

  for (auto f_node : ordered) {
    // Check if we can pass this as wildcard ops
    if (f_node->IsSink() || f_node->IsSource() ||
        f_node->type_string() == "IdentityN" ||
        f_node->type_string() == "Const") {
      continue;
    }

    DNNL_GRAPH_VLOG(0) << " FW Node " << f_node->type_string();
    dnnl::graph::op* dnnl_graph_node = nullptr;
    if (op_translation_map_.find(f_node->type_string()) !=
        op_translation_map_.end()) {
      op_translation_map_[f_node->type_string()](f_node, &dnnl_graph_node);
    } else {
      (op_translation_map_["Unhandled"])(f_node, &dnnl_graph_node);
    }

    if (dnnl_graph_node == nullptr) {
      return errors::Internal("DNNL_GRAPH node conversion failed");
    }

    dnnl_graph_id_selected_node_map_.emplace(f_node->id(), f_node);
    for (auto in_edge : f_node->in_edges()) {
      if (in_edge->IsControlEdge()) {
        continue;
      }
      Node* src = in_edge->src();
      Node* dst = f_node;
      DataType dt = dst->input_type(in_edge->dst_input());

      dnnl_graph_shape_t dnnl_graph_input_shape;
      int64_t unique_logical_tensor_id =
          LlgaTensorId(src->id(), in_edge->src_output());
      auto input_logical_tensor = dnnl::graph::logical_tensor(
          unique_logical_tensor_id, GetLlgaDataType(dt), dnnl_graph_input_shape,
          dnnl::graph::logical_tensor::layout_type::undef);
      DNNL_GRAPH_VLOG(0) << "INPUT ID " << unique_logical_tensor_id;
      dnnl_graph_node->add_input(input_logical_tensor);
    }

    // This is to ensure we add only one unique output
    // based on the dst->src_output() of the out_edge
    std::unordered_set<int64_t> uniq_edges;

    for (auto out_edge : f_node->out_edges()) {
      if (out_edge->IsControlEdge()) {
        continue;
      }
      Node* src = f_node;
      Node* dst = out_edge->dst();
      DataType dt = src->output_type(out_edge->src_output());
      dnnl_graph_shape_t dnnl_graph_output_shape;

      int64_t unique_logical_tensor_id =
          LlgaTensorId(src->id(), out_edge->src_output());

      if (uniq_edges.find(unique_logical_tensor_id) != uniq_edges.end()) {
        continue;
      }
      uniq_edges.insert(unique_logical_tensor_id);
      auto output_logical_tensor = dnnl::graph::logical_tensor(
          unique_logical_tensor_id, GetLlgaDataType(dt),
          dnnl_graph_output_shape,
          dnnl::graph::logical_tensor::layout_type::undef);
      DNNL_GRAPH_VLOG(0) << "OUTPUT ID " << unique_logical_tensor_id;
      dnnl_graph_node->add_output(output_logical_tensor);
    }
    graph_ctx.add_op(*dnnl_graph_node);
  }
  // Get Partitions from DNNL_GRAPH
  auto l_partition_list =
      graph_ctx.get_partitions(dnnl::graph::partition::policy::fusion);

  // Mark nodes in a single partition with a unique partition ID in TF graph
  int partition_idx = 0;
  for (auto it : l_partition_list) {
    if (it.get_ops_num() == 0 || it.get_ops_num() == 1) continue;
    for (auto partitioned_node : it.get_ops()) {
      auto tf_node = GetTFNodeFromLlgaId(partitioned_node);
      if (tf_node == nullptr) {
        return errors::Internal("No TF node corresponding to llga node");
      }
      tf_node->AddAttr("_partition_id", partition_idx);
      partitioned_fw_node_list_.emplace_back(tf_node);
    }

    partition_id_to_dnnl_graph_partition_[partition_idx] = std::move(it);
    partition_idx++;
  }

  DNNL_GRAPH_VLOG(0) << "Number of Non Single Op Partitions " << partition_idx;
  LlgaFuser fuser(&graph);
  auto status = fuser.RewritePass(*this);
  if (status != Status::OK()) {
    return status;
  }

  if (tf_dnnl_graph_bridge::DumpAllGraphs()) {
    DumpGraphs(graph, tf_dnnl_graph_bridge::graph_file_idx++,
               "tf_modified_graph", "Modified");
    DNNL_GRAPH_VLOG(0) << " Dumping modified TF graph tf_modified_graph_"
                       << graph_file_idx;
  }

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  return Status::OK();
}  // namespace tf_dnnl_graph_bridge

void TfLlgaPartitioner::Feedback(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 const GraphDef& optimize_output,
                                 double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(TfLlgaPartitioner, "dnnl_graph-partitioner");

}  // namespace tf_dnnl_graph_bridge

}  // namespace tensorflow
