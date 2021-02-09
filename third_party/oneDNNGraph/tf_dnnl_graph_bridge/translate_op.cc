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

#include "translate_op.h"

#include "utils.h"

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

Status TranslateAddOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node =
      new dnnl::graph::op(node->id(), dnnl::graph::op::kind::Add, node->name());
  return Status::OK();
}

Status TranslateBiasAdd(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::BiasAdd, node->name());
  return Status::OK();
}

Status TranslateConv2DOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Convolution, node->name());
  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;

  GetNodeAttr(node->attrs(), "strides", &tf_strides);
  GetNodeAttr(node->attrs(), "dilations", &tf_dilations);
  GetNodeAttr(node->attrs(), "padding", &tf_padding_type);
  GetNodeAttr(node->attrs(), "data_format", &tf_data_format);

  if (tf_padding_type == "SAME") {
    // TODO(nbpatel): Check what TF sets it
    tf_padding_type = "SAME_UPPER";
  }

  bool is_channel_last =
      (tf_data_format.size() == 4 ? tf_data_format[3] == 'C' ? true : false
                                  : false);

  // Strides in the batch and depth dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_channel_last ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Strides in batch and depth dimensions is not supported: ",
        node->type_string());
  }

  std::vector<int64_t> strides(2);
  std::vector<int64_t> dilations(2);

  ExtractSpatialDims(is_channel_last, tf_strides, &strides);
  ExtractSpatialDims(is_channel_last, tf_dilations, &dilations);

  (*dnnl_graph_node)->set_attr("strides", strides);
  (*dnnl_graph_node)->set_attr("dilations", dilations);
  (*dnnl_graph_node)->set_attr("groups", int64_t{1});
  (*dnnl_graph_node)->set_attr("pads_begin", std::vector<int64_t>{1, 1});
  (*dnnl_graph_node)->set_attr("pads_end", std::vector<int64_t>{1, 1});
  (*dnnl_graph_node)->set_attr("auto_pad", tf_padding_type);
  (*dnnl_graph_node)->set_attr("data_format", std::string("NXC"));
  (*dnnl_graph_node)->set_attr("filter_format", std::string("XIO"));

  return Status::OK();
}

Status TranslateDivideOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Divide, node->name());
  return Status::OK();
}

Status TranslateFusedBatchNormV3Op(const Node* node,
                                   dnnl::graph::op** dnnl_graph_node) {
  float tf_epsilon;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "epsilon", &tf_epsilon));
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::BatchNormInference, node->name());
  (*dnnl_graph_node)->set_attr("epsilon", tf_epsilon);
  return Status::OK();
}

Status TranslateMatMulOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::MatMul, node->name());
  (*dnnl_graph_node)->set_attr("transpose_a", false);
  (*dnnl_graph_node)->set_attr("transpose_b", false);
  return Status::OK();
}

Status TranslateMaxPoolOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::MaxPool, node->name());
  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;

  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "data_format", &tf_data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "ksize", &tf_ksize));

  if (tf_padding_type == "SAME") {
    tf_padding_type = "SAME_UPPER";
  }

  bool is_channel_last =
      (tf_data_format.size() == 4 ? tf_data_format[3] == 'C' ? true : false
                                  : false);

  // Strides in the batch and depth dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_channel_last ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Strides in batch and depth dimensions is not supported: ",
        node->type_string());
  }

  std::vector<int64_t> strides(2);
  std::vector<int64_t> kernel(2);

  ExtractSpatialDims(is_channel_last, tf_strides, &strides);
  ExtractSpatialDims(is_channel_last, tf_ksize, &kernel);

  (*dnnl_graph_node)->set_attr("strides", strides);
  // TODO (pruthvi), generalize dialation computation
  (*dnnl_graph_node)->set_attr("dilations", std::vector<int64_t>{1, 1});
  (*dnnl_graph_node)->set_attr("pads_begin", std::vector<int64_t>{1, 1});
  (*dnnl_graph_node)->set_attr("pads_end", std::vector<int64_t>{1, 1});
  (*dnnl_graph_node)->set_attr("auto_pad", tf_padding_type);
  (*dnnl_graph_node)->set_attr("data_format", std::string("NXC"));
  (*dnnl_graph_node)->set_attr("kernel", kernel);

  return Status::OK();
}

Status TranslateMulOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Multiply, node->name());
  return Status::OK();
}

Status TranslateReluOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::ReLU, node->name());
  return Status::OK();
}

Status TranslateSoftmaxOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::SoftMax, node->name());
  return Status::OK();
}

Status TranslateSquareOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Square, node->name());
  return Status::OK();
}

Status TranslateTanhOp(const Node* node, dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Tanh, node->name());
  return Status::OK();
}

Status TranslateTransposeOp(const Node* node,
                            dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Transpose, node->name());
  return Status::OK();
}

Status TranslateUnhandledOp(const Node* node,
                            dnnl::graph::op** dnnl_graph_node) {
  *dnnl_graph_node = new dnnl::graph::op(
      node->id(), dnnl::graph::op::kind::Wildcard, node->name());
  return Status::OK();
}

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
