/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

const int kRows = 100000;

int RowsAndColsArg(int r, int c) { return r * kRows + c; }
int RowsFromArg(int arg) { return (arg / kRows); }
int ColsFromArg(int arg) { return (arg % kRows); }

template <class T>
Graph* BiasAdd(int rows, int cols, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(type, TensorShape({rows, cols}));
  lhs.template flat<T>().setRandom();
  TensorShape rhs_shape;
  rhs_shape = TensorShape({cols});
  Tensor rhs(type, rhs_shape);
  rhs.template flat<T>().setRandom();
  test::graph::Binary(g, "BiasAdd", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, R, C)                             \
  void BM_##DEVICE##_##C_TYPE##_BiasAdd_R##R##_C##C(int iters, int arg) {      \
    const int rows = RowsFromArg(arg);                                         \
    const int cols = ColsFromArg(arg);                                         \
    const int64 tot = static_cast<int64>(iters) * rows * cols;                 \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(tot);                                              \
    test::Benchmark(#DEVICE, BiasAdd<C_TYPE>(rows, cols, TF_TYPE)).Run(iters); \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_##C_TYPE##_BiasAdd_R##R##_C##C)                      \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BIAS_ADD_ALL(DEVICE, C_TYPE, TF_TYPE)   \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 5120, 2048); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 5120, 4096); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 2048, 5120); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 4096, 5120);

using Eigen::half;
BM_BIAS_ADD_ALL(cpu, float, DT_FLOAT);
BM_BIAS_ADD_ALL(cpu, bfloat16, DT_BFLOAT16);
#undef BM_BIAS_ADD_ALL
#undef BM_BIAS_ADD

template <class T>
Graph* BcastAdd(int rows, int cols, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  TensorShape lhs_shape, rhs_shape;

  // row
  lhs_shape = TensorShape({rows, cols});
  rhs_shape = TensorShape({rows, 1});

  Tensor lhs(type, lhs_shape);
  lhs.template flat<T>().setRandom();
  Tensor rhs(type, rhs_shape);
  rhs.template flat<T>().setRandom();
  test::graph::Binary(g, "Add", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BCAST_ADD_ROW(DEVICE, C_TYPE, TF_TYPE, R, C)                         \
  void BM_##DEVICE##_##C_TYPE##_BcastAddRow_R##R##_C##C(int iters, int arg) {   \
    const int rows = RowsFromArg(arg);                                          \
    const int cols = ColsFromArg(arg);                                          \
    const int64 tot = static_cast<int64>(iters) * rows * cols;                  \
    testing::UseRealTime();                                                     \
    testing::ItemsProcessed(tot);                                               \
    test::Benchmark(#DEVICE, BcastAdd<C_TYPE>(rows, cols, TF_TYPE)).Run(iters); \
  }                                                                             \
  BENCHMARK(BM_##DEVICE##_##C_TYPE##_BcastAddRow_R##R##_C##C)                   \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_ROW_ALL(DEVICE, C_TYPE, TF_TYPE)    \
  BM_BCAST_ADD_ROW(DEVICE, C_TYPE, TF_TYPE, 5120, 2048); \
  BM_BCAST_ADD_ROW(DEVICE, C_TYPE, TF_TYPE, 5120, 4096); \
  BM_BCAST_ADD_ROW(DEVICE, C_TYPE, TF_TYPE, 2048, 5120); \
  BM_BCAST_ADD_ROW(DEVICE, C_TYPE, TF_TYPE, 4096, 5120);

BM_BCAST_ADD_ROW_ALL(cpu, float, DT_FLOAT);
BM_BCAST_ADD_ROW_ALL(cpu, bfloat16, DT_BFLOAT16);
#undef BM_BCAST_ADD_ROW_ALL
#undef BM_BCAST_ADD_ROW

}  // namespace
}  // namespace tensorflow
