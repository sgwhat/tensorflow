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

#ifndef TF_DNNL_GRAPH_BRIDGE_VERSION_H_
#define TF_DNNL_GRAPH_BRIDGE_VERSION_H_

#include "common.h"

namespace tensorflow {
namespace tf_dnnl_graph_bridge {
extern "C" {

// Returns the 0 if _GLIBCXX_USE_CXX11_ABI wasn't set by the
// compiler (e.g., clang or gcc pre 4.8) or the value of the
// _GLIBCXX_USE_CXX11_ABI set during the compilation time
int TfLlgaCxx11AbiFlag();

// Returns true when dnnl_graph is using Grappler optimizer APIs for
// graph rewriting
bool TfLlgaIsGrapplerEnabled();

// Returns true when tf-dnnl_graph bridge is built with
// --enable_variables_and_optimizers flag
bool TfLlgaAreVariablesEnabled();

// Returns the tensorflow version
const char* TfVersion();
}
}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow

#endif  // TF_DNNL_GRAPH_BRIDGE_VERSION_H_
