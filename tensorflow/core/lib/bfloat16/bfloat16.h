/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
#define TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_

#include <cmath>
#include <complex>

#include "tensorflow/core/platform/byte_order.h"

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define B16_DEVICE_FUNC __host__ __device__

#else
#define B16_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/bfloat16.h for description.
struct bfloat16 {
  // The default constructor must yield a zero value, not an uninitialized
  // value; some TF kernels use T() as a zero value.
  B16_DEVICE_FUNC bfloat16() : value(ZERO_VALUE) {}

  B16_DEVICE_FUNC static bfloat16 truncate_to_bfloat16(const float v) {
    bfloat16 output;
    if (float_isnan(v)) {
      output.value = NAN_VALUE;
      return output;
    }
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    output.value = p[0];
#else
    output.value = p[1];
#endif
    return output;
  }

  B16_DEVICE_FUNC explicit bfloat16(const float v) {
    value = round_to_bfloat16(v).value;
  }

  B16_DEVICE_FUNC explicit bfloat16(const double val)
      : bfloat16(static_cast<float>(val)) {}
  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  B16_DEVICE_FUNC explicit bfloat16(const complex64& val)
      : bfloat16(val.real()) {}

  B16_DEVICE_FUNC explicit bfloat16(const complex128& val)
      : bfloat16(static_cast<float>(val.real())) {}

  B16_DEVICE_FUNC explicit bfloat16(const unsigned short val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const unsigned int val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const int val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const long val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit bfloat16(const long long val)
      : bfloat16(static_cast<float>(val)) {}

  template <class T>
  B16_DEVICE_FUNC explicit bfloat16(const T& val)
      : bfloat16(static_cast<float>(val)) {}

  B16_DEVICE_FUNC explicit operator float() const {
    float result = 0;

    uint16_t* q = reinterpret_cast<uint16_t*>(&result);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    q[0] = value;
#else
    q[1] = value;
#endif
    return result;
  }

  B16_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator Eigen::half() const;

  B16_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator double() const {
    return static_cast<double>(float(*this));
  }

  B16_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  B16_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  union FP32 {
    unsigned int u;
    float f;
  };

  // Converts a float point to bfloat16, with round-nearest-to-even as rounding
  // method.
  // TODO: There is a slightly faster implementation (8% faster on CPU)
  // than this (documented in cl/175987786), that is exponentially harder to
  // understand and document. Switch to the faster version when converting to
  // BF16 becomes compute-bound.
  B16_DEVICE_FUNC static bfloat16 round_to_bfloat16(float v) {
    uint32_t input;
    FP32 f;
    f.f = v;
    input = f.u;
    bfloat16 output;

    if (float_isnan(v)) {
      // If the value is a NaN, squash it to a qNaN with msb of fraction set,
      // this makes sure after truncation we don't end up with an inf.
      //
      // qNaN magic: All exponent bits set + most significant bit of fraction
      // set.
      output.value = 0x7fc0;
    } else {
      // Fast rounding algorithm that rounds a half value to nearest even. This
      // reduces expected error when we convert a large number of floats. Here
      // is how it works:
      //
      // Definitions:
      // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
      // with the following tags:
      //
      // Sign |  Exp (8 bits) | Frac (23 bits)
      //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
      //
      //  S: Sign bit.
      //  E: Exponent bits.
      //  F: First 6 bits of fraction.
      //  L: Least significant bit of resulting bfloat16 if we truncate away the
      //  rest of the float32. This is also the 7th bit of fraction
      //  R: Rounding bit, 8th bit of fraction.
      //  T: Sticky bits, rest of fraction, 15 bits.
      //
      // To round half to nearest even, there are 3 cases where we want to round
      // down (simply truncate the result of the bits away, which consists of
      // rounding bit and sticky bits) and two cases where we want to round up
      // (truncate then add one to the result).
      //
      // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
      // 1s) as the rounding bias, adds the rounding bias to the input, then
      // truncates the last 16 bits away.
      //
      // To understand how it works, we can analyze this algorithm case by case:
      //
      // 1. L = 0, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input may create any carry, depending on
      //   whether there is any value set to 1 in T bits.
      //   - R may be set to 1 if there is a carry.
      //   - L remains 0.
      //   - Note that this case also handles Inf and -Inf, where all fraction
      //   bits, including L, R and Ts are all 0. The output remains Inf after
      //   this algorithm.
      //
      // 2. L = 1, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits but
      //   adds 1 to rounding bit.
      //   - L remains 1.
      //
      // 3. L = 0, R = 1, all of T are 0:
      //   Expect: round down, this is exactly at half, the result is already
      //   even (L=0).
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input sets all sticky bits to 1, but
      //   doesn't create a carry.
      //   - R remains 1.
      //   - L remains 0.
      //
      // 4. L = 1, R = 1:
      //   Expect: round up, this is exactly at half, the result needs to be
      //   round to the next even number.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits, but
      //   creates a carry from rounding bit.
      //   - The carry sets L to 0, creates another carry bit and propagate
      //   forward to F bits.
      //   - If all the F bits are 1, a carry then propagates to the exponent
      //   bits, which then creates the minimum value with the next exponent
      //   value. Note that we won't have the case where exponents are all 1,
      //   since that's either a NaN (handled in the other if condition) or inf
      //   (handled in case 1).
      //
      // 5. L = 0, R = 1, any of T is 1:
      //   Expect: round up, this is greater than half.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input creates a carry from sticky bits,
      //   sets rounding bit to 0, then create another carry.
      //   - The second carry sets L to 1.
      //
      // Examples:
      //
      //  Exact half value that is already even:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0     1000000000000000
      //
      //     This falls into case 3. We truncate the rest of 16 bits and no
      //     carry is created into F and L:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //  Exact half value, round to next even number:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 0 1     1000000000000000
      //
      //     This falls into case 4. We create a carry from R and T,
      //     which then propagates into L and F:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //
      //  Max denormal value round to min normal value:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 1      0 0 0 0 0 0 0
      //
      //  Max normal value round to Inf:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     1 1 1 1 1 1 1 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     1 1 1 1 1 1 1 1      0 0 0 0 0 0 0
      //
      //
      // Least significant bit of resulting bfloat.
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;
      output.value = static_cast<uint16_t>(input >> 16);
    }
    return output;
  }

  static bfloat16 epsilon() {
    bfloat16 x;
    x.value = 0x3c00;  // 0x1.0p-7
    return x;
  }

  static bfloat16 highest() {
    bfloat16 x;
    x.value = 0x7F7F;  // 0x1.FEp127
    return x;
  }

  static bfloat16 lowest() {
    bfloat16 x;
    x.value = 0xFF7F;  // -0x1.FEp127
    return x;
  }

  static bfloat16 min_positive_normal() {
    bfloat16 x;
    x.value = 0x0080;  // 0x1p-126
    return x;
  }

  bool IsZero() const { return (value & 0x7FFF) == ZERO_VALUE; }

  uint16_t value;

  // A value that represents "not a number".
  static const uint16_t NAN_VALUE = 0x7FC0;

 private:
  // A value that represents "zero".
  static const uint16_t ZERO_VALUE = 0;

  B16_DEVICE_FUNC static bool float_isnan(const float& x) {
#ifdef __CUDA_ARCH__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
};

B16_DEVICE_FUNC inline std::ostream& operator<<(std::ostream& os,
                                                const bfloat16& dt) {
  os << static_cast<float>(dt);
  return os;
}

B16_DEVICE_FUNC inline bfloat16 operator+(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator+(bfloat16 a, int b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator+(int a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator-(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator*(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator/(bfloat16 a, bfloat16 b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}
B16_DEVICE_FUNC inline bfloat16 operator-(bfloat16 a) {
  a.value ^= 0x8000;
  return a;
}
B16_DEVICE_FUNC inline bool operator<(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator<=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator==(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator!=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator>(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
B16_DEVICE_FUNC inline bool operator>=(bfloat16 a, bfloat16 b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
B16_DEVICE_FUNC inline bfloat16& operator+=(bfloat16& a, bfloat16 b) {
  a = a + b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16& operator-=(bfloat16& a, bfloat16 b) {
  a = a - b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator++(bfloat16& a) {
  a += bfloat16(1);
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator--(bfloat16& a) {
  a -= bfloat16(1);
  return a;
}
B16_DEVICE_FUNC inline bfloat16 operator++(bfloat16& a, int) {
  bfloat16 original_value = a;
  ++a;
  return original_value;
}
B16_DEVICE_FUNC inline bfloat16 operator--(bfloat16& a, int) {
  bfloat16 original_value = a;
  --a;
  return original_value;
}
B16_DEVICE_FUNC inline bfloat16& operator*=(bfloat16& a, bfloat16 b) {
  a = a * b;
  return a;
}
B16_DEVICE_FUNC inline bfloat16& operator/=(bfloat16& a, bfloat16 b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::bfloat16> {
  size_t operator()(const tensorflow::bfloat16& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};

using tensorflow::bfloat16;
inline bool isinf(const bfloat16& a) { return std::isinf(float(a)); }
inline bool isnan(const bfloat16& a) { return std::isnan(float(a)); }
inline bool isfinite(const bfloat16& a) { return std::isfinite(float(a)); }
inline bfloat16 abs(const bfloat16& a) { return bfloat16(std::abs(float(a))); }
inline bfloat16 exp(const bfloat16& a) { return bfloat16(std::exp(float(a))); }
inline bfloat16 expm1(const bfloat16& a) {
  return bfloat16(std::expm1(float(a)));
}
inline bfloat16 log(const bfloat16& a) { return bfloat16(std::log(float(a))); }
inline bfloat16 log1p(const bfloat16& a) {
  return bfloat16(std::log1p(float(a)));
}
inline bfloat16 log10(const bfloat16& a) {
  return bfloat16(std::log10(float(a)));
}
inline bfloat16 sqrt(const bfloat16& a) {
  return bfloat16(std::sqrt(float(a)));
}
inline bfloat16 pow(const bfloat16& a, const bfloat16& b) {
  return bfloat16(std::pow(float(a), float(b)));
}
inline bfloat16 sin(const bfloat16& a) { return bfloat16(std::sin(float(a))); }
inline bfloat16 cos(const bfloat16& a) { return bfloat16(std::cos(float(a))); }
inline bfloat16 tan(const bfloat16& a) { return bfloat16(std::tan(float(a))); }
inline bfloat16 tanh(const bfloat16& a) {
  return bfloat16(std::tanh(float(a)));
}
inline bfloat16 floor(const bfloat16& a) {
  return bfloat16(std::floor(float(a)));
}
inline bfloat16 ceil(const bfloat16& a) {
  return bfloat16(std::ceil(float(a)));
}
}  // namespace std

#ifdef EIGEN_VECTORIZE_AVX512

#include "./Eigen/src/Core/arch/AVX512/PacketMath.h"
namespace Eigen {

namespace internal {

using tensorflow::bfloat16;

typedef union {
  __m256i v;
  unsigned short arr[16];
} Packet16b;

template <>
struct packet_traits<bfloat16> : default_packet_traits {
  typedef Packet16b type;
  // There is no half-size packet for Packet8h.
  typedef Packet16b half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
    HasHalfPacket = 0,
    HasAdd = 1,
    HasSub = 0,
    HasMul = 0,
    HasDiv = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasConj = 0,
    HasSetLinear = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};

template <>
struct unpacket_traits<Packet16b> {
  typedef bfloat16 type;
  enum {
    size = 16,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
  typedef Packet16b half;
};

template <>
EIGEN_STRONG_INLINE Packet16b pset1<Packet16b>(const bfloat16& from) {
  Packet16b r;
  r.v = _mm256_set1_epi16(from.value);
  return r;
}

template <>
EIGEN_STRONG_INLINE bfloat16 pfirst<Packet16b>(const Packet16b& from) {
  bfloat16 t;
  t.value = static_cast<unsigned short>(_mm256_extract_epi16(from.v, 0));
  return t;
}

template <>
EIGEN_STRONG_INLINE Packet16b pload<Packet16b>(const bfloat16* from) {
  Packet16b r;
  r.v = _mm256_load_si256(reinterpret_cast<const __m256i*>(from));
  return r;
}

template <>
EIGEN_STRONG_INLINE Packet16b ploadu<Packet16b>(const bfloat16* from) {
  Packet16b r;
  r.v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from));
  return r;
}

template <>
EIGEN_STRONG_INLINE void pstore<bfloat16>(bfloat16* to, const Packet16b& from) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(to), from.v);
}

template <>
EIGEN_STRONG_INLINE void pstoreu<bfloat16>(bfloat16* to,
                                           const Packet16b& from) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(to), from.v);
}

EIGEN_STRONG_INLINE Packet16f Bf16ToF32(const Packet16b& a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a.v), 16));
}

EIGEN_STRONG_INLINE Packet16b F32ToBf16(const Packet16f& a) {
  Packet16b r;
  __m512i t;
  __m512i input = _mm512_castps_si512(a);
  __m512i nan = _mm512_set1_epi32(0x7fc0);

  // uint32_t lsb = (input >> 16) & 1;
  t = _mm512_and_si512(_mm512_srli_epi32(input, 16), _mm512_set1_epi32(1));
  // uint32_t rounding_bias = 0x7fff + lsb;
  t = _mm512_add_epi32(t, _mm512_set1_epi32(0x7fff));
  // input += rounding_bias;
  t = _mm512_add_epi32(t, input);
  // input = input >> 16;
  t = _mm512_srli_epi32(t, 16);

  // Check NaN before converting back to bf16
  __mmask16 mask = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  t = _mm512_mask_blend_epi32(mask, nan, t);

  // output.value = static_cast<uint16_t>(input);
  // t[12-15] t[12-15] t[8-11] t[8-11] t[4-7] t[4-7] t[0-4] t[0-4]
  // 7[0]     6[0]     5[0]    4[0]    3[6]   2[4]   1[2]   0[0]
  __m512i idx = _mm512_set_epi64(0, 0, 0, 0, 7, 5, 3, 0);
  t = _mm512_packus_epi32(t, t);
  t = _mm512_permutexvar_epi64(idx, t);
  r.v = _mm512_castsi512_si256(t);
  return r;
}

template <>
EIGEN_STRONG_INLINE Packet16b padd<Packet16b>(const Packet16b& a,
                                              const Packet16b& b) {
  return F32ToBf16(_mm512_add_ps(Bf16ToF32(a), Bf16ToF32(b)));
}

template <>
EIGEN_STRONG_INLINE Packet16b preduxp<Packet16b>(const Packet16b* p) {
  Packet16f t = Bf16ToF32(*p);
  return F32ToBf16(preduxp<Packet16f>(&t));
}

template <>
EIGEN_STRONG_INLINE bfloat16 predux<Packet16b>(const Packet16b& p) {
  return (bfloat16)predux<Packet16f>(Bf16ToF32(p));
}

template <>
EIGEN_STRONG_INLINE Packet16b pmul<Packet16b>(const Packet16b& a,
                                              const Packet16b& b) {
  return F32ToBf16(_mm512_mul_ps(Bf16ToF32(a), Bf16ToF32(b)));
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16b, 16>& kernel) {
  __m256i a = kernel.packet[0].v;
  __m256i b = kernel.packet[1].v;
  __m256i c = kernel.packet[2].v;
  __m256i d = kernel.packet[3].v;
  __m256i e = kernel.packet[4].v;
  __m256i f = kernel.packet[5].v;
  __m256i g = kernel.packet[6].v;
  __m256i h = kernel.packet[7].v;
  __m256i i = kernel.packet[8].v;
  __m256i j = kernel.packet[9].v;
  __m256i k = kernel.packet[10].v;
  __m256i l = kernel.packet[11].v;
  __m256i m = kernel.packet[12].v;
  __m256i n = kernel.packet[13].v;
  __m256i o = kernel.packet[14].v;
  __m256i p = kernel.packet[15].v;

  __m256i ab_07 = _mm256_unpacklo_epi16(a, b);
  __m256i cd_07 = _mm256_unpacklo_epi16(c, d);
  __m256i ef_07 = _mm256_unpacklo_epi16(e, f);
  __m256i gh_07 = _mm256_unpacklo_epi16(g, h);
  __m256i ij_07 = _mm256_unpacklo_epi16(i, j);
  __m256i kl_07 = _mm256_unpacklo_epi16(k, l);
  __m256i mn_07 = _mm256_unpacklo_epi16(m, n);
  __m256i op_07 = _mm256_unpacklo_epi16(o, p);

  __m256i ab_8f = _mm256_unpackhi_epi16(a, b);
  __m256i cd_8f = _mm256_unpackhi_epi16(c, d);
  __m256i ef_8f = _mm256_unpackhi_epi16(e, f);
  __m256i gh_8f = _mm256_unpackhi_epi16(g, h);
  __m256i ij_8f = _mm256_unpackhi_epi16(i, j);
  __m256i kl_8f = _mm256_unpackhi_epi16(k, l);
  __m256i mn_8f = _mm256_unpackhi_epi16(m, n);
  __m256i op_8f = _mm256_unpackhi_epi16(o, p);

  __m256i abcd_03 = _mm256_unpacklo_epi32(ab_07, cd_07);
  __m256i abcd_47 = _mm256_unpackhi_epi32(ab_07, cd_07);
  __m256i efgh_03 = _mm256_unpacklo_epi32(ef_07, gh_07);
  __m256i efgh_47 = _mm256_unpackhi_epi32(ef_07, gh_07);
  __m256i ijkl_03 = _mm256_unpacklo_epi32(ij_07, kl_07);
  __m256i ijkl_47 = _mm256_unpackhi_epi32(ij_07, kl_07);
  __m256i mnop_03 = _mm256_unpacklo_epi32(mn_07, op_07);
  __m256i mnop_47 = _mm256_unpackhi_epi32(mn_07, op_07);

  __m256i abcd_8b = _mm256_unpacklo_epi32(ab_8f, cd_8f);
  __m256i abcd_cf = _mm256_unpackhi_epi32(ab_8f, cd_8f);
  __m256i efgh_8b = _mm256_unpacklo_epi32(ef_8f, gh_8f);
  __m256i efgh_cf = _mm256_unpackhi_epi32(ef_8f, gh_8f);
  __m256i ijkl_8b = _mm256_unpacklo_epi32(ij_8f, kl_8f);
  __m256i ijkl_cf = _mm256_unpackhi_epi32(ij_8f, kl_8f);
  __m256i mnop_8b = _mm256_unpacklo_epi32(mn_8f, op_8f);
  __m256i mnop_cf = _mm256_unpackhi_epi32(mn_8f, op_8f);

  __m256i abcdefgh_01 = _mm256_unpacklo_epi64(abcd_03, efgh_03);
  __m256i abcdefgh_23 = _mm256_unpackhi_epi64(abcd_03, efgh_03);
  __m256i ijklmnop_01 = _mm256_unpacklo_epi64(ijkl_03, mnop_03);
  __m256i ijklmnop_23 = _mm256_unpackhi_epi64(ijkl_03, mnop_03);
  __m256i abcdefgh_45 = _mm256_unpacklo_epi64(abcd_47, efgh_47);
  __m256i abcdefgh_67 = _mm256_unpackhi_epi64(abcd_47, efgh_47);
  __m256i ijklmnop_45 = _mm256_unpacklo_epi64(ijkl_47, mnop_47);
  __m256i ijklmnop_67 = _mm256_unpackhi_epi64(ijkl_47, mnop_47);
  __m256i abcdefgh_89 = _mm256_unpacklo_epi64(abcd_8b, efgh_8b);
  __m256i abcdefgh_ab = _mm256_unpackhi_epi64(abcd_8b, efgh_8b);
  __m256i ijklmnop_89 = _mm256_unpacklo_epi64(ijkl_8b, mnop_8b);
  __m256i ijklmnop_ab = _mm256_unpackhi_epi64(ijkl_8b, mnop_8b);
  __m256i abcdefgh_cd = _mm256_unpacklo_epi64(abcd_cf, efgh_cf);
  __m256i abcdefgh_ef = _mm256_unpackhi_epi64(abcd_cf, efgh_cf);
  __m256i ijklmnop_cd = _mm256_unpacklo_epi64(ijkl_cf, mnop_cf);
  __m256i ijklmnop_ef = _mm256_unpackhi_epi64(ijkl_cf, mnop_cf);

  // NOTE: no unpacklo/hi instr in this case, so using permute instr.
  __m256i a_p_0 = _mm256_permute2x128_si256(abcdefgh_01, ijklmnop_01, 0x20);
  __m256i a_p_1 = _mm256_permute2x128_si256(abcdefgh_23, ijklmnop_23, 0x20);
  __m256i a_p_2 = _mm256_permute2x128_si256(abcdefgh_45, ijklmnop_45, 0x20);
  __m256i a_p_3 = _mm256_permute2x128_si256(abcdefgh_67, ijklmnop_67, 0x20);
  __m256i a_p_4 = _mm256_permute2x128_si256(abcdefgh_89, ijklmnop_89, 0x20);
  __m256i a_p_5 = _mm256_permute2x128_si256(abcdefgh_ab, ijklmnop_ab, 0x20);
  __m256i a_p_6 = _mm256_permute2x128_si256(abcdefgh_cd, ijklmnop_cd, 0x20);
  __m256i a_p_7 = _mm256_permute2x128_si256(abcdefgh_ef, ijklmnop_ef, 0x20);
  __m256i a_p_8 = _mm256_permute2x128_si256(abcdefgh_01, ijklmnop_01, 0x31);
  __m256i a_p_9 = _mm256_permute2x128_si256(abcdefgh_23, ijklmnop_23, 0x31);
  __m256i a_p_a = _mm256_permute2x128_si256(abcdefgh_45, ijklmnop_45, 0x31);
  __m256i a_p_b = _mm256_permute2x128_si256(abcdefgh_67, ijklmnop_67, 0x31);
  __m256i a_p_c = _mm256_permute2x128_si256(abcdefgh_89, ijklmnop_89, 0x31);
  __m256i a_p_d = _mm256_permute2x128_si256(abcdefgh_ab, ijklmnop_ab, 0x31);
  __m256i a_p_e = _mm256_permute2x128_si256(abcdefgh_cd, ijklmnop_cd, 0x31);
  __m256i a_p_f = _mm256_permute2x128_si256(abcdefgh_ef, ijklmnop_ef, 0x31);

  kernel.packet[0].v = a_p_0;
  kernel.packet[1].v = a_p_1;
  kernel.packet[2].v = a_p_2;
  kernel.packet[3].v = a_p_3;
  kernel.packet[4].v = a_p_4;
  kernel.packet[5].v = a_p_5;
  kernel.packet[6].v = a_p_6;
  kernel.packet[7].v = a_p_7;
  kernel.packet[8].v = a_p_8;
  kernel.packet[9].v = a_p_9;
  kernel.packet[10].v = a_p_a;
  kernel.packet[11].v = a_p_b;
  kernel.packet[12].v = a_p_c;
  kernel.packet[13].v = a_p_d;
  kernel.packet[14].v = a_p_e;
  kernel.packet[15].v = a_p_f;
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16b, 8>& kernel) {
  EIGEN_ALIGN64 bfloat16 in[8][16];
  pstore<bfloat16>(in[0], kernel.packet[0]);
  pstore<bfloat16>(in[1], kernel.packet[1]);
  pstore<bfloat16>(in[2], kernel.packet[2]);
  pstore<bfloat16>(in[3], kernel.packet[3]);
  pstore<bfloat16>(in[4], kernel.packet[4]);
  pstore<bfloat16>(in[5], kernel.packet[5]);
  pstore<bfloat16>(in[6], kernel.packet[6]);
  pstore<bfloat16>(in[7], kernel.packet[7]);

  EIGEN_ALIGN64 bfloat16 out[8][16];

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      out[i][j] = in[j][2 * i];
    }
    for (int j = 0; j < 8; ++j) {
      out[i][j + 8] = in[j][2 * i + 1];
    }
  }

  kernel.packet[0] = pload<Packet16b>(out[0]);
  kernel.packet[1] = pload<Packet16b>(out[1]);
  kernel.packet[2] = pload<Packet16b>(out[2]);
  kernel.packet[3] = pload<Packet16b>(out[3]);
  kernel.packet[4] = pload<Packet16b>(out[4]);
  kernel.packet[5] = pload<Packet16b>(out[5]);
  kernel.packet[6] = pload<Packet16b>(out[6]);
  kernel.packet[7] = pload<Packet16b>(out[7]);
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16b, 4>& kernel) {
  EIGEN_ALIGN64 bfloat16 in[4][16];
  pstore<bfloat16>(in[0], kernel.packet[0]);
  pstore<bfloat16>(in[1], kernel.packet[1]);
  pstore<bfloat16>(in[2], kernel.packet[2]);
  pstore<bfloat16>(in[3], kernel.packet[3]);

  EIGEN_ALIGN64 bfloat16 out[4][16];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i][j] = in[j][4 * i];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j + 4] = in[j][4 * i + 1];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j + 8] = in[j][4 * i + 2];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j + 12] = in[j][4 * i + 3];
    }
  }

  kernel.packet[0] = pload<Packet16b>(out[0]);
  kernel.packet[1] = pload<Packet16b>(out[1]);
  kernel.packet[2] = pload<Packet16b>(out[2]);
  kernel.packet[3] = pload<Packet16b>(out[3]);
}
}  // namesapce internal
}  // namespaec Eigen
#endif  // EIGEN_VECTORIZE_AVX512

#endif  // TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
