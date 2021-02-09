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

#include "version.h"

#include <iostream>
#include <string>

// Tf-dnnl_graph bridge uses semantic versioning: see http://semver.org/

#define VERSION_STR_HELPER(x) #x
#define VERSION_STR(x) VERSION_STR_HELPER(x)

namespace tensorflow {
namespace tf_dnnl_graph_bridge {

int TfLlgaCxx11AbiFlag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}

bool TfLlgaIsGrapplerEnabled() { return true; }

bool TfLlgaAreVariablesEnabled() {
#if defined(TF_DNNL_GRAPH_ENABLE_VARIABLES_AND_OPTIMIZERS)
  return true;
#else
  return false;
#endif
}

const char* TfVersion() { return (TF_VERSION_STRING); }

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
