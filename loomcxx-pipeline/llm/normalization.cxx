// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Numerical definitions use ordinary C++ pointers. The bounded native bridge
// publishes the original raw-buffer families, supplying zero pointer origins.
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/scalar.h>
#include <loomcxx/target/amdgpu.h>

using Index = loom::type::index;
using Offset = loom::type::offset;
using Input = loom::type::buffer<const float>;
using Output = loom::type::buffer<unsigned char>;

[[loom::config("qwen.model.hidden_size")]] extern const Index hidden_size;
[[loom::config("qwen.model.hidden_divisor")]] extern const float hidden_divisor;

// These are the existing raw-buffer families. All reusable callers dispatch
// through them so independently selected providers affect every call site.
LOOM_TEMPLATE_DECL("llm.rmsnorm.row")
void public_rmsnorm_row(Index width, float divisor, Index rows, Index row,
                        Offset output_origin, bool half_output, float epsilon,
                        Input input, Input weight, Output output);
LOOM_TEMPLATE_DECL("qwen38.rmsnorm_f32.row")
void public_qwen_row(Index rows, Index row, Offset output_origin, float epsilon,
                     Input input, Input weight, Output output);

#ifndef LOOMCXX_CALLERS_ONLY
LOOM_DEVICE LOOM_TEMPLATE_DECL("cxx.llm.rmsnorm.row") void rmsnorm_row(
    Index width, float divisor, Index rows, Index row, Offset output_origin,
    bool half_output, float epsilon, const float *input, const float *weight,
    unsigned char *output);

LOOM_TEMPLATE_DEF(rmsnorm_row)
[[loom::symbol("cxx.llm_rmsnorm_row")]]
LOOM_FORCE_INLINE void
rmsnorm_row_provider(Index width, float divisor, Index rows, Index row,
                     Offset output_origin, bool half_output, float epsilon,
                     const float *input, const float *weight,
                     unsigned char *output) {
  loom::assume(loom::predicate::range(width, 1, 32768));
  Index lane = static_cast<Index>(loom::kernel::workitem::id.x);
  loom::assume(loom::predicate::range(lane, 0, 255));
  auto *input_root = loom::buffer::assume::noalias(input);
  auto *weight_root = loom::buffer::assume::noalias(weight);
  auto *output_root = loom::buffer::assume::noalias(output);
  auto input_view =
      loom::buffer::view<loom::type::dynamic, loom::type::dynamic>(
          input_root, loom::type::indices{rows, width},
          loom::encoding::layout::dense<2>());
  auto weight_view = loom::buffer::view<loom::type::dynamic>(
      weight_root, loom::type::indices{width},
      loom::encoding::layout::dense<1>());
  float partial = 0.0f;
  for (Index element = lane; element < width; element = element + Index{256}) {
    float value = loom::view::load(input_view, row, element);
    float square;
    [[loom::fastmath("reassoc", "nnan", "ninf", "nsz", "contract")]] square =
        value * value;
    [[loom::fastmath("reassoc", "nnan", "ninf", "nsz")]] partial =
        partial + square;
  }
  float sum = loom::kernel::workgroup::reduce::addf(partial);
  float mean;
  [[loom::fastmath("nnan", "ninf", "nsz", "arcp")]] mean = sum / divisor;
  float biased;
  [[loom::fastmath("nnan", "ninf", "nsz")]] biased = mean + epsilon;
  float inverse;
  [[loom::fastmath("nnan", "ninf", "nsz", "afn")]] inverse =
      loom::scalar::rsqrtf(biased);
  auto *output_start =
      output_root + loom::offset::cast<unsigned long long>(output_origin);
  for (Index element = lane; element < width; element = element + Index{256}) {
    float value = loom::view::load(input_view, row, element);
    float weight_value = loom::view::load(weight_view, element);
    float normalized;
    float result;
    [[loom::fastmath("nnan", "ninf", "nsz")]] {
      normalized = value * inverse;
      result = normalized * weight_value;
    }
    if (half_output) {
      auto view = loom::buffer::view<loom::type::dynamic>(
          reinterpret_cast<_Float16 *>(output_start),
          loom::type::indices{width}, loom::encoding::layout::dense<1>());
      loom::view::store(static_cast<_Float16>(result), view, element);
    } else {
      auto view = loom::buffer::view<loom::type::dynamic>(
          reinterpret_cast<float *>(output_start), loom::type::indices{width},
          loom::encoding::layout::dense<1>());
      loom::view::store(result, view, element);
    }
  }
}

LOOM_DEVICE LOOM_TEMPLATE_DECL("cxx.qwen38.rmsnorm_f32.row") void qwen_row(
    Index rows, Index row, Offset output_origin, float epsilon, Input input,
    Input weight, Output output);

LOOM_TEMPLATE_DEF(qwen_row)
[[loom::symbol("cxx.qwen38_rmsnorm_f32_row")]]
void qwen_row_provider(Index rows, Index row, Offset output_origin,
                       float epsilon, Input input, Input weight,
                       Output output) {
  public_rmsnorm_row(hidden_size, hidden_divisor, rows, row, output_origin,
                     false, epsilon, input, weight, output);
}
#endif

#ifndef LOOMCXX_PROVIDERS_ONLY
constexpr loom::amdgpu::target decode_target{.kind = "gfx11-generic",
                                             .subgroup_size = 32};

[[loom::kernel, loom::symbol("qwen38_rmsnorm_f32_decode"),
  loom::target(decode_target), loom::workgroup_count(1, 1, 1),
  loom::workgroup_size(256, 1, 1)]]
void decode(float epsilon, Input input, Input weight, Output output) {
  public_qwen_row(Index{1}, Index{0}, Offset{0}, epsilon, input, weight,
                  output);
}

static loom::kernel::configuration prefill_configuration(Index tokens) {
  loom::assume(loom::predicate::range(tokens, 1, 512));
  return {{1, loom::index::cast<unsigned>(tokens), 1}, {256, 1, 1}};
}

template <bool Half>
LOOM_FORCE_INLINE void prefill_body(Index tokens, float epsilon, Input input,
                                    Input weight, Output output) {
  loom::assume(loom::predicate::range(tokens, 1, 512));
  Index row = static_cast<Index>(loom::kernel::workgroup::id.y);
  loom::assume(loom::predicate::lt(row, tokens));
  Offset origin = loom::index::scale(
      row, loom::index::scale(hidden_size, Offset{Half ? 2 : 4}));
  public_rmsnorm_row(hidden_size, hidden_divisor, tokens, row, origin, Half,
                     epsilon, input, weight, output);
}

[[loom::kernel(prefill_configuration),
  loom::symbol("qwen38_rmsnorm_f32_prefill")]]
void prefill_f32(Index tokens, float epsilon, Input input, Input weight,
                 Output output) {
  prefill_body<false>(tokens, epsilon, input, weight, output);
}

[[loom::kernel(prefill_configuration),
  loom::symbol("qwen38_rmsnorm_f16_prefill")]]
void prefill_f16(Index tokens, float epsilon, Input input, Input weight,
                 Output output) {
  prefill_body<true>(tokens, epsilon, input, weight, output);
}
#endif
