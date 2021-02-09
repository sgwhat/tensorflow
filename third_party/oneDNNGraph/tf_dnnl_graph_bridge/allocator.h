/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef TF_DNNL_GRAPH_BRIDGE_ALLOCATOR_H
#define TF_DNNL_GRAPH_BRIDGE_ALLOCATOR_H

#include <vector>

#include "common.h"

namespace tensorflow {

class Allocator;

namespace tf_dnnl_graph_bridge {
class TfLlgaAllocator {
 public:
  TfLlgaAllocator();
  ~TfLlgaAllocator();

  void* allocate_persistent(size_t n);
  void deallocate_persistent(void* buffer);
  void* allocate_output(size_t n);
  void* allocate_temp(size_t n);

  dnnl::graph::allocator& GetLlgaAllocator();

  static void* allocate_persistent(void* allocator, size_t mem_size);
  static void deallocate_persistent(void* allocator, void* mem);
  static void* allocate_output(void* allocator, size_t mem_size);
  static void* allocate_temp(void* allocator, size_t mem_size);

 private:
  tensorflow::Allocator* allocator_;
  dnnl::graph::allocator dnnl_graph_allocator_;
  std::vector<void*> temp_allocations_;
};

}  // namespace tf_dnnl_graph_bridge
}  // namespace tensorflow
#endif
