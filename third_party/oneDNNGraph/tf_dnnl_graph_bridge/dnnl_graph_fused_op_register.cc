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

#include "common.h"

namespace tensorflow {

namespace tf_dnnl_graph_bridge {

// ------------------------------------------------------------------
REGISTER_OP("LlgaFused")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("dnnl_graph_partition: int")
    .Attr("dnnl_graph_backend: string")
    .Attr("input_node_ids: list(int) >= 0")
    .Attr("output_node_ids: list(int) >= 0")
    .Attr("output_ids_with_layout_any: list(int) >= 0")
    .SetIsStateful()
    .Doc("DNNL_GRAPH Fused Op.");
}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow