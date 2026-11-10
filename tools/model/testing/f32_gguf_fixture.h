// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_MODEL_TESTING_F32_GGUF_FIXTURE_H_
#define HRX_SOLUTIONS_MODEL_TESTING_F32_GGUF_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdint>
#include <string>
#include <vector>
namespace hrx::model::testing {
// Complete synthetic one-layer Llama descriptor in a real external GGUF. The
// table is behind a distractor tensor and contains distinct known token rows.
std::vector<uint8_t> EmbeddingGguf(uint32_t width = 4, uint32_t vocab = 3);
bool ReadFixture(const std::vector<uint8_t> &bytes, model::GgufCatalog *catalog,
                 model::ModelDescriptor *model, std::string *error);
// Same pinned IREE GGUF index used by Loom's named parameter file loader.
bool ReadNamedF32(const std::vector<uint8_t> &bytes, const std::string &name,
                  std::vector<float> *values, std::string *error);
} // namespace hrx::model::testing
#endif
