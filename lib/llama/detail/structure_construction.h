// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LLAMA_DETAIL_STRUCTURE_CONSTRUCTION_H_
#define HRX_SOLUTIONS_LIB_LLAMA_DETAIL_STRUCTURE_CONSTRUCTION_H_
#include "lib/llama/structure.h"
namespace hrx::llama::detail {
class StructureTestPeer;
// Internal recipe state, never a successful partial model factory. Discard the
// construction after any append failure. Publication belongs to complete-model
// composition; tests may explicitly retain an incomplete fixture via their
// peer.
class StructureConstruction {
public:
  static std::unique_ptr<StructureConstruction>
  Create(model::ModelDescriptor facts, LlamaBuildOptions options,
         const operations::OperationRegistry &registry, std::string *error);
  operations::OperationGraph &graph() { return owner_->graph_; }
  const LlamaStructure &structure() const { return *owner_; }
  bool createInputs(std::string *error);
  bool resolve(ParameterCoordinate coordinate, operations::ValueId *out,
               std::string *error);
  bool record(LlamaRegion region, size_t begin, size_t end, std::string *error);
  bool exportValue(operations::ValueId value, std::string *error);

private:
  friend class StructureTestPeer;
  friend bool ::hrx::llama::BuildLlamaStructure(
      model::ModelDescriptor, const LlamaBuildOptions &,
      std::shared_ptr<const LlamaStructure> *, std::string *);
  explicit StructureConstruction(std::shared_ptr<LlamaStructure> owner)
      : owner_(std::move(owner)) {}
  std::shared_ptr<LlamaStructure> owner_;
};
// Shared explicit numeric admission. diagnostic storage_name is never
// consulted.
bool MatrixEncoding(uint32_t storage, operations::Encoding *out);
} // namespace hrx::llama::detail
#endif
