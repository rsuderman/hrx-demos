// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_MODEL_TENSOR_NAMES_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_TENSOR_NAMES_H_
#include "tools/model/model_descriptor.h"
#include <string_view>
namespace hrx::model {
// Name spelling only; required roles, dimensions and format admission belong
// to the callers. Failure leaves *name unchanged, including wrong role scope.
bool ExpectedTensorName(const ModelDescriptor &, std::string_view role,
                        std::optional<uint32_t> layer, std::string *name);
} // namespace hrx::model
#endif
