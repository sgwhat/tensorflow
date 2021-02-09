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

#ifndef TF_DNNL_GRAPH_BRIDGE_PARTITIONER_H_
#define TF_DNNL_GRAPH_BRIDGE_PARTITIONER_H_

#include <iomanip>
#include <iostream>
#include <list>
#include <string>

#include "common.h"
#include "replace_fusions.h"

namespace tensorflow {
namespace tf_dnnl_graph_bridge {
// Custom Grappler Optimizer for TF-DNNL_GRAPH
class TfLlgaPartitioner : public tensorflow::grappler::CustomGraphOptimizer {
 public:
  TfLlgaPartitioner() = default;
  ~TfLlgaPartitioner() override = default;

  string name() const override { return "TfLlgaPartitioner"; };

#if TF_MAJOR_VERSION >= 2
  bool UsesFunctionLibrary() const override { return false; }
#endif

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  // This is a grappler pass to change a TF graph to DNNL_GRAPH enabled TF
  // graph. It accepts TF nodes that can be processed by DNNL_GRAPH and
  // encapsulates them into LlgaFusedOp

  Status Optimize(tensorflow::grappler::Cluster*,
                  const tensorflow::grappler::GrapplerItem&,
                  GraphDef*) override;

  void Feedback(tensorflow::grappler::Cluster*,
                const tensorflow::grappler::GrapplerItem&, const GraphDef&,
                double) override;

  Node* GetTFNodeFromLlgaId(uint64_t id);

  std::vector<tensorflow::Node*> GetPartitionedNodes() {
    return partitioned_fw_node_list_;
  }

  // TODO(nbpatel): Fix this. No globals
  static std::map<int, dnnl::graph::partition>
      partition_id_to_dnnl_graph_partition_;

 private:
  // dnnl_graph node id -> fw node that is selected by backend
  std::map<size_t, tensorflow::Node*> dnnl_graph_id_selected_node_map_;

  // list of partitioned framework nodes
  std::vector<tensorflow::Node*> partitioned_fw_node_list_;
};

}  // namespace tf_dnnl_graph_bridge

}  // namespace tensorflow
#endif  // TF_DNNL_GRAPH_BRIDGE_INTEGRATION_H_
