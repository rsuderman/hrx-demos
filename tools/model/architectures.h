// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_MODEL_ARCHITECTURES_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_ARCHITECTURES_H_
#include "tools/model/model_descriptor.h"
#include <string_view>
namespace hrx::model::internal {
struct Loader;
struct ModelArchitecture {
  const char *family;
  void (*load)(const Loader &, ModelDescriptor &);
  const char *(*extra_tensor_name)(std::string_view role, bool is_layer);
};
const ModelArchitecture *SelectModelArchitecture(std::string_view family);
} // namespace hrx::model::internal
#endif
