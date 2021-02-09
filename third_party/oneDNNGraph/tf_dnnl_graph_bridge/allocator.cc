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

#include "allocator.h"

#include <vector>

#include "utils.h"

tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::TfLlgaAllocator()
    : allocator_(GetTfAllocator()),
      dnnl_graph_allocator_(this, allocate_persistent, deallocate_persistent,
                            allocate_output, allocate_temp) {}

tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::~TfLlgaAllocator() {
  for (auto alloc : temp_allocations_) {
    allocator_->DeallocateRaw(alloc);
  }
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_persistent(
    size_t n) {
  return allocator_->AllocateRaw(32, n);
}

void tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::deallocate_persistent(
    void* buffer) {
  allocator_->DeallocateRaw(buffer);
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_output(
    size_t n) {
  return allocator_->AllocateRaw(32, n);
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_temp(
    size_t n) {
  void* result = allocator_->AllocateRaw(32, n);
  temp_allocations_.push_back(result);
  return result;
}

dnnl::graph::allocator&
tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::GetLlgaAllocator() {
  return dnnl_graph_allocator_;
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_persistent(
    void* allocator, size_t mem_size) {
  return static_cast<TfLlgaAllocator*>(allocator)->allocate_persistent(
      mem_size);
}

void tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::deallocate_persistent(
    void* allocator, void* mem) {
  return static_cast<TfLlgaAllocator*>(allocator)->deallocate_persistent(mem);
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_output(
    void* allocator, size_t mem_size) {
  return static_cast<TfLlgaAllocator*>(allocator)->allocate_output(mem_size);
}

void* tensorflow::tf_dnnl_graph_bridge::TfLlgaAllocator::allocate_temp(
    void* allocator, size_t mem_size) {
  return static_cast<TfLlgaAllocator*>(allocator)->allocate_temp(mem_size);
}
