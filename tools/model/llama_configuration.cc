// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/llama_configuration.h"
namespace hrx::model {
bool ExtractLlamaConfiguration(FILE *file, ModelDescriptor *out,
                               std::string *error) {
  GgufCatalog catalog;
  if (!out) {
    if (error)
      *error = "Llama configuration requires output";
    return false;
  }
  if (!ReadGgufCatalog(file, &catalog, error))
    return false;
  if (catalog.architecture != "llama") {
    if (error)
      *error = "expected llama GGUF architecture";
    return false;
  }
  return LoadModelDescriptor(catalog, out, error);
}
} // namespace hrx::model
