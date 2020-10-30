/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.
#ifdef INTEL_MKL

#include <unordered_map>

#include "mkldnn.hpp"
#include "tensorflow/core/kernels/mkl/mkl_eltwise_ops.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklReluOp : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                             0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    (static_cast<T*>(out_o))[0] =
        std::max((static_cast<T*>(user_i))[0], static_cast<T>(0));
    return;
  }
};

template <typename Device, typename T>
class MklReluGradOp
    : public MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                                 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] *
        (static_cast<T>((static_cast<T*>(user_i))[0] > static_cast<T>(0)));
    return;
  }
};

template <typename Device, typename T>
class MklEluOp : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_elu> {
 public:
  ~MklEluOp() {}

  explicit MklEluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_elu>(context, 1.0f,
                                                            0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // return exp(feature) - 1 if feature > 0; feature otherwise
    T feature = (static_cast<T*>(user_i))[0];
    if (feature < static_cast<T>(0))
      (static_cast<T*>(out_o))[0] =
          Eigen::numext::exp(feature) - static_cast<T>(1);
    else
      (static_cast<T*>(out_o))[0] = feature;
    return;
  }
};

template <typename Device, typename T>
class MklEluGradOp
    : public MklEltwiseGradOpBase<Device, T,
                                  ALGORITHM::eltwise_elu_use_dst_for_bwd> {
 public:
  ~MklEluGradOp() {}

  explicit MklEluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_elu_use_dst_for_bwd>(
            context, 1.0f, 0.0f) {}

  // EluGrad gets 'y' from Elu, where 'y' is output of Elu(x).
  virtual int GetTypeOfInputTensorFromFwdOp() const { return MKLDNN_ARG_DST; }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);
    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    // The gradient of elu(x) = 1 if elu(x) > 0; elu(x) + 1 otherwise.
    T elu = (static_cast<T*>(user_i))[0];
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    if (elu > static_cast<T>(0)) {
      (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0];
    } else {
      (static_cast<T*>(out_o))[0] =
          (static_cast<T*>(user_g))[0] * (elu + static_cast<T>(1));
    }
    return;
  }
};

#define RELU6_UPPER_BOUND 6.0f
template <typename Device, typename T>
class MklRelu6Op
    : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_bounded_relu> {
 public:
  ~MklRelu6Op() {}

  explicit MklRelu6Op(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = std::min(std::max(user_i[0], static_cast<T>(0)),
                        static_cast<T>(RELU6_UPPER_BOUND));
    return;
  }
};

template <typename Device, typename T>
class MklRelu6GradOp
    : public MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_bounded_relu> {
 public:
  ~MklRelu6GradOp() {}

  explicit MklRelu6GradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_g[0] *
               static_cast<T>(user_i[0] > static_cast<T>(0) &&
                              (user_i[0] < static_cast<T>(RELU6_UPPER_BOUND)));
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluOp
    : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklLeakyReluOp() {}

  explicit MklLeakyReluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                             0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = user_i[0] > T(0) ? user_i[0] : user_i[0] * T(this->alpha_);
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluGradOp
    : public MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklLeakyReluGradOp() {}

  explicit MklLeakyReluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                                 0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_i[0] > static_cast<T>(0)
                   ? user_g[0]
                   : user_g[0] * static_cast<T>(this->alpha_);
    return;
  }
};

template <typename Device, typename T>
class MklTanhOp : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_tanh> {
 public:
  ~MklTanhOp() {}

  explicit MklTanhOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_tanh>(context, 0.0f,
                                                             0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // tanh(x) = (e^x - e^(-x))/ (e^x + e^(-x))
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = Eigen::numext::exp(feature);
    T e2 = Eigen::numext::exp(-feature);
    (static_cast<T*>(out_o))[0] = (e1 - e2) / (e1 + e2);
    return;
  }
};

template <typename Device, typename T>
class MklTanhGradOp
    : public MklEltwiseGradOpBase<Device, T,
                                  ALGORITHM::eltwise_tanh_use_dst_for_bwd> {
 public:
  ~MklTanhGradOp() {}

  explicit MklTanhGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T,
                             ALGORITHM::eltwise_tanh_use_dst_for_bwd>(
            context, 0.0f, 0.0f) {}

  virtual int GetDiffDstIndex() const { return 1; }
  virtual int GetSrcIndex() const { return 0; }
  virtual int GetDiffSrcIndex() const { return 0; }

  // TanhGrad gets 'y' from Tanh, where 'y' is output of Tanh(x).
  virtual int GetTypeOfInputTensorFromFwdOp() const { return MKLDNN_ARG_DST; }

  virtual void Compute_Scalar(OpKernelContext* context) {
    // NOTE: Order of y and dy for Tanh is reverse of that for Relu/Elu/other
    // element-wise ops. Tanh is math op in Tensorflow; others are NN ops.
    const size_t diff_dst_index = GetDiffDstIndex();
    const size_t src_index = GetSrcIndex();
    const size_t diff_src_index = GetDiffSrcIndex();
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    // gradient of tanh(x) = 1 - tanh(x)^2
    // Input to TanhGrad is output of Tanh. So we do not need to compute
    // Tanh again.
    T tanh = (static_cast<T*>(user_i))[0];
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * (static_cast<T>(1) - tanh * tanh);
  }
};

template <typename Device, typename T>
class MklSwishOp
    : public MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_swish> {
 public:
  ~MklSwishOp() {}

  explicit MklSwishOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, ALGORITHM::eltwise_swish>(context, 1.0f,
                                                              0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    // swish(x) =  x * sigmoid(x).
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = Eigen::numext::exp(-feature);
    (static_cast<T*>(out_o))[0] = feature / (static_cast<T>(1) + e1);
    return;
  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReluGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES(type)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklElu")                                          \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluOp<CPUDevice, type>);                              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklEluGrad")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES(type)       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6Op<CPUDevice, type>);                            \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6Grad")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6GradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_LEAKYRELU_MKL_SUPPORTED_KERNELS_TYPES(type)   \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyRelu")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluOp<CPUDevice, type>);                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyReluGrad")                                \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_LEAKYRELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_LEAKYRELU_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanh")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanhGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES(type)       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklSwish")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklSwishOp<CPUDevice, type>);
TF_CALL_float(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);

// Regist Swish Kernel for Eigen CPU. Because TF registers it in Python API.
#define REGISTER_CPU(T)    \
  REGISTER_KERNEL_BUILDER( \
      Name("_FusedSwish").Device(DEVICE_CPU).TypeConstraint<T>("T"), NoOp);

TF_CALL_FLOAT_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

}  // namespace tensorflow

#endif  // INTEL_MKL
