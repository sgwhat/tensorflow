
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
#include "add_identityn.h"

using namespace std;

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

Status AddIdentityN(Graph* input_graph, std::set<string> skip_these_nodes) {
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    bool ref_type = false;
    fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
    if (fetch_node) {
      DNNL_GRAPH_VLOG(5) << "DNNL_GRAPH_OPTIMIZER: Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs()) {
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (int i = 0; i < node->num_outputs(); i++) {
          if (IsRefType(node->output_type(i))) {
            DNNL_GRAPH_VLOG(5) << "DNNL_GRAPH_OPTIMIZER: "
                               << "Datatype for the node output"
                               << " at index " << i << " is ref type";
            ref_type = true;
            break;
          }
          input_types.push_back(node->output_type(i));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
          // if the fetch node is FusedBatchNorm && its an inference Op, we will
          // just add the first output of Bn to IdentityN, since other Outputs
          // are relevant only in training;
          if (node->type_string() == "FusedBatchNormV3") {
            bool is_training;
            GetNodeAttr(node->attrs(), "is_training", &is_training);
            if (!is_training) {
              break;
            }
          }
        }

        if (ref_type) {
          DNNL_GRAPH_VLOG(5)
              << "DNNL_GRAPH_OPTIMIZER: Cannot construct an IdentityN node";
          continue;
        }

        DNNL_GRAPH_VLOG(5)
            << "DNNL_GRAPH_OPTIMIZER: Creating an IdentityN node";
        Node* identityN_node;
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                               .Attr("T", input_types)
                               .Input(inputs)
                               .Device(node->assigned_device_name())
                               .Finalize(input_graph, &identityN_node));

        identityN_node->set_assigned_device_name(node->assigned_device_name());
        DNNL_GRAPH_VLOG(0) << "IDENTITYN DEVICE NAME "
                           << node->assigned_device_name();

        // Rename the skip node
        // Get a new name for the node with the given prefix
        // We will use the 'original-node-name_dnnl_graph as the prefix
        string new_name = input_graph->NewName(node->name() + "_dnnl_graph");
        // TODO: Use (guaranteed) unique name here
        node->set_name(new_name);
      } else {
        DNNL_GRAPH_VLOG(0) << "DNNL_GRAPH_OPTIMIZER: num outputs "
                           << node->num_outputs();
        DNNL_GRAPH_VLOG(0)
            << "DNNL_GRAPH_OPTIMIZER: Cannot construct an IdentityN node";
      }
    }
  }
  return Status::OK();
}

}  // namespace tf_dnnl_graph_bridge

}  // namespace tensorflow
