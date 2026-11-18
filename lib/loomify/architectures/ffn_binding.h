// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_FFN_BINDING_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_FFN_BINDING_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace hrx::model {
struct LayerDescriptor;
}

namespace hrx::loomify_gguf::internal {
struct Producer;

// Bind the three standard FFN projections and append their existing lookup row.
// The selected caller retains its down-format admission; only gate and up use
// the canonical projection geometry checks.
bool BindFfnProjections(Producer &producer, const model::LayerDescriptor &layer,
                        uint64_t hidden,
                        std::initializer_list<uint32_t> down_formats,
                        std::vector<std::vector<uint64_t>> &rows);
} // namespace hrx::loomify_gguf::internal

#endif
