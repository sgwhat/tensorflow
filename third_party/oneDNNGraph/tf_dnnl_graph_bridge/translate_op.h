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

#ifndef TF_DNNL_GRAPH_BRIDGE_TRANSLATE_OP_H_
#define TF_DNNL_GRAPH_BRIDGE_TRANSLATE_OP_H_

#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <string>

#include "common.h"

namespace tensorflow {
namespace tf_dnnl_graph_bridge {

typedef Status (*FnPtr)(const Node* node, dnnl::graph::op** dnnl_graph_node);

Status TranslateAddOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateBiasAdd(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateConv2DOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateDivideOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateFusedBatchNormV3Op(const Node* node,
                                   dnnl::graph::op** dnnl_graph_node);
Status TranslateMatMulOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateMaxPoolOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateMulOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateReluOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateSoftmaxOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateSquareOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateTanhOp(const Node* node, dnnl::graph::op** dnnl_graph_node);
Status TranslateTransposeOp(const Node* node,
                            dnnl::graph::op** dnnl_graph_node);
Status TranslateUnhandledOp(const Node* node,
                            dnnl::graph::op** dnnl_graph_node);

static std::map<std::string, FnPtr> op_translation_map_ = {
    {"Add", TranslateAddOp},
    {"AddV2", TranslateAddOp},
    {"BiasAdd", TranslateBiasAdd},
    {"Conv2D", TranslateConv2DOp},
    {"FusedBatchNormV3", TranslateFusedBatchNormV3Op},
    {"MatMul", TranslateMatMulOp},
    {"MaxPool", TranslateMaxPoolOp},
    {"Mul", TranslateMulOp},
    {"RealDiv", TranslateDivideOp},
    {"Relu", TranslateReluOp},
    {"Softmax", TranslateSoftmaxOp},
    {"Square", TranslateSquareOp},
    {"Tanh", TranslateTanhOp},
    {"Transpose", TranslateTransposeOp},
    {"Unhandled", TranslateUnhandledOp}};

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
#endif