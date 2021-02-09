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

#ifndef TF_DNNL_GRAPH_BRIDGE_DNNL_GRAPH_FUSED_OP_H_
#define TF_DNNL_GRAPH_BRIDGE_DNNL_GRAPH_FUSED_OP_H_

#include <ostream>
#include <vector>

#include "common.h"
#include "partitioner.h"

namespace tensorflow {
namespace tf_dnnl_graph_bridge {

class LlgaFusedOp : public OpKernel {
 public:
  explicit LlgaFusedOp(OpKernelConstruction* ctx);
  ~LlgaFusedOp() override;
  void Compute(OpKernelContext* ctx) override;

  struct opaque_desc {
    size_t dnnl_graph_layout_id_;
    std::vector<int64_t> dnnl_graph_input_shape_;
  };

  int partition_id_{-1};
  std::vector<int64> input_node_ids_;
  std::vector<int64> output_node_ids_;
  std::vector<DataType> input_dt_types_;
  std::vector<DataType> output_dt_types_;
  std::vector<int64> output_ids_with_layout_any_;

  static std::map<int64, opaque_desc> input_desc_map_;
};

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
#endif  // TF_DNNL_GRAPH_BRIDGE_DNNL_GRAPH_FUSED_OP_H_
