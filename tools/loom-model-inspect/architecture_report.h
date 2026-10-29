// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_LOOM_MODEL_INSPECT_ARCHITECTURE_REPORT_H_
#define HRX_SOLUTIONS_TOOLS_LOOM_MODEL_INSPECT_ARCHITECTURE_REPORT_H_

#include <optional>
#include <string>

#include "tools/model/model_descriptor.h"

namespace hrx::model_inspect {

struct ReportOptions {
  std::string source;
  std::optional<uint32_t> layer_index;
  std::optional<std::string> tensor_name;
};

// Reports validated model/catalog facts, including unbound tensors. Source is
// an optional caller-provided display label. On failure leaves report
// unchanged. Selectors are mutually exclusive; tensor names match raw bytes.
bool FormatArchitecture(const model::GgufCatalog &catalog,
                        const model::ModelDescriptor &model,
                        std::string *report, std::string *error,
                        const ReportOptions &options = {});

} // namespace hrx::model_inspect

#endif // HRX_SOLUTIONS_TOOLS_LOOM_MODEL_INSPECT_ARCHITECTURE_REPORT_H_
