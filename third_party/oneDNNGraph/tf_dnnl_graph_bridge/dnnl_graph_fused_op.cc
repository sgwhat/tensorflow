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

#include "dnnl_graph_fused_op.h"

#include <cstdlib>
#include <mutex>
#include <numeric>
#include <utility>

#include "allocator.h"
#include "replace_fusions.h"
#include "runtime.h"
#include "utils.h"

using namespace std;

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

std::map<int64, LlgaFusedOp::opaque_desc> LlgaFusedOp::input_desc_map_;

LlgaFusedOp::LlgaFusedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr<int>("dnnl_graph_partition", &partition_id_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_node_ids", &input_node_ids_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_node_ids", &output_node_ids_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Targuments", &input_dt_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tresults", &output_dt_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_ids_with_layout_any",
                                   &output_ids_with_layout_any_));
}

void LlgaFusedOp::Compute(OpKernelContext* ctx) {
  dnnl::graph::compiled_partition compiled_partition;
  vector<dnnl::graph::logical_tensor> l_input_list;
  vector<dnnl::graph::logical_tensor> l_output_list;
  vector<dnnl::graph::tensor> l_input_tensor_list;
  vector<dnnl::graph::tensor> l_output_tensor_list;

  // TF input tensor
  for (int i = 0; i < ctx->num_inputs(); i++) {
    void* current_src_ptr = const_cast<void*>(DMAHelper::base(&ctx->input(i)));
    auto tf_input_shape = ctx->input(i).shape();
    auto input_data_type = GetLlgaDataType(ctx->input(i).dtype());

    bool is_opaque =
        Contains<int64>(output_ids_with_layout_any_, input_node_ids_[i]);

    if (is_opaque) {
      auto opaque_mem_desc = input_desc_map_[input_node_ids_[i]];
      size_t opaque_layout_id = opaque_mem_desc.dnnl_graph_layout_id_;
      std::vector<int64_t> dnnl_graph_input_shape =
          opaque_mem_desc.dnnl_graph_input_shape_;
      auto input_logical_tensor =
          dnnl::graph::logical_tensor(input_node_ids_[i], input_data_type,
                                      dnnl_graph_input_shape, opaque_layout_id);
      l_input_list.emplace_back(input_logical_tensor);
      l_input_tensor_list.emplace_back(input_logical_tensor, current_src_ptr);
    } else {
      std::vector<int64_t> dnnl_graph_input_shape =
          get_tensor_shape(tf_input_shape);
      auto input_logical_tensor = dnnl::graph::logical_tensor(
          input_node_ids_[i], input_data_type, dnnl_graph_input_shape,
          dnnl::graph::logical_tensor::layout_type::strided);
      l_input_list.emplace_back(input_logical_tensor);
      l_input_tensor_list.emplace_back(input_logical_tensor, current_src_ptr);
    }
  }

  // TF output tensor
  for (int i = 0; i < output_node_ids_.size(); i++) {
    auto output_data_type = GetLlgaDataType(output_dt_types_[i]);

    dnnl::graph::logical_tensor::layout_type dnnl_graph_layout_type;
    bool is_opaque =
        Contains<int64>(output_ids_with_layout_any_, output_node_ids_[i]);

    if (is_opaque) {
      dnnl_graph_layout_type = dnnl::graph::logical_tensor::layout_type::any;
    } else {
      dnnl_graph_layout_type =
          dnnl::graph::logical_tensor::layout_type::strided;
    }
    auto output_logical_tensor = dnnl::graph::logical_tensor(
        output_node_ids_[i], output_data_type, dnnl_graph_layout_type);
    l_output_list.emplace_back(output_logical_tensor);

    auto partition =
        TfLlgaPartitioner::partition_id_to_dnnl_graph_partition_[partition_id_];
    partition.infer_shape(l_input_list, l_output_list);
    vector<int64_t> dnnl_graph_output_shape = l_output_list[i].get_dims();

    if (!is_opaque) {
      // Set the strides for output
      std::vector<int64_t> dst_strides(dnnl_graph_output_shape.size(), 1);
      for (size_t i = 0; i < dst_strides.size() - 1; ++i) {
        dst_strides[i] = std::accumulate(
            dnnl_graph_output_shape.begin() + i + 1,
            dnnl_graph_output_shape.end(), 1, std::multiplies<int64_t>());
      }
      // Rewrite the output logical tensor with new strides
      l_output_list[i] =
          dnnl::graph::logical_tensor(output_node_ids_[i], output_data_type,
                                      dnnl_graph_output_shape, dst_strides);
    }

    // Compile the partition
    compiled_partition =
        partition.compile(l_input_list, l_output_list, Engine::getEngine());

    // Convert dnnl_graph shape to tf shape
    std::vector<int64> tf_shape(dnnl_graph_output_shape.size());
    std::copy(begin(dnnl_graph_output_shape), end(dnnl_graph_output_shape),
              begin(tf_shape));

    l_output_list[i] =
        compiled_partition.query_logical_tensor(output_node_ids_[i]);

    // Trick the TF allocate_output with 1D shapes, since the mem_size required
    // by opaque layout is more than the dims multiplied by each other *
    // sizeof(DT)
    if (is_opaque) {
      vector<int64> one_dim_shape(1);
      one_dim_shape[0] = l_output_list[i].get_mem_size() / sizeof(_Float32);
      tf_shape.resize(1);
      std::copy(begin(one_dim_shape), end(one_dim_shape), begin(tf_shape));
    }

    // Allocate memory for output tensor
    TensorShape tf_output_shape(tf_shape);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(i, tf_output_shape, &output_tensor));
    void* current_dst_ptr = DMAHelper::base(output_tensor);
    l_output_tensor_list.emplace_back(l_output_list[i], current_dst_ptr);

    // Here we store the layout_id & original shape (4D) for the next input
    // since we dont have access to this partition's output logical tensor in
    // the next partition.
    if (is_opaque) {
      input_desc_map_[output_node_ids_[i]] = {l_output_list[i].get_layout_id(),
                                              dnnl_graph_output_shape};
    }

    // DNNL_GRAPH submit stream here
    compiled_partition.execute(Stream::getStream(), l_input_tensor_list,
                               l_output_tensor_list);
    DNNL_GRAPH_VLOG(0) << "SUCESSFULLY RAN THROUGH DNNL_GRAPH ";
  }

}  // end compute

LlgaFusedOp::~LlgaFusedOp() {}

}  // namespace tf_dnnl_graph_bridge

REGISTER_KERNEL_BUILDER(Name("LlgaFused").Device(DEVICE_CPU),
                        tf_dnnl_graph_bridge::LlgaFusedOp);

}  // namespace tensorflow
