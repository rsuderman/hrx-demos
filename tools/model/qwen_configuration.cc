// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/qwen_configuration.h"
namespace hrx::model {
bool ExtractQwenConfiguration(FILE *file, ModelDescriptor *out,
                              std::string *error) {
  GgufCatalog catalog;
  if (!out) {
    if (error)
      *error = "Qwen configuration requires output";
    return false;
  }
  if (!ReadGgufCatalog(file, &catalog, error))
    return false;
  if (catalog.architecture != "qwen35") {
    if (error)
      *error = "expected qwen35 GGUF architecture";
    return false;
  }
  return LoadModelDescriptor(catalog, out, error);
}
} // namespace hrx::model
