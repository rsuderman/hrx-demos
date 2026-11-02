// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LLAMA_STRUCTURE_H_
#define HRX_SOLUTIONS_LIB_LLAMA_STRUCTURE_H_

#include "lib/operations/operation_graph.h"
#include "tools/model/model_descriptor.h"
#include <memory>
#include <string>
#include <vector>

namespace hrx::llama {
namespace detail {
class StructureConstruction;
}
struct LlamaBuildOptions {
  operations::Phase phase = operations::Phase::Prefill;
  uint64_t row_capacity = 0;
  uint64_t cache_capacity = 0;
  uint64_t history_capacity = 0;
  std::string source_asset;
};
enum class ParameterScope { Global, Layer };
struct ParameterCoordinate {
  ParameterScope scope = ParameterScope::Global;
  size_t layer_slot = 0; // Zero for globals; vector slot, not descriptor index.
  size_t tensor_slot = 0;
};
struct ParameterAssociation {
  operations::ValueId value;
  ParameterCoordinate coordinate;
};
struct LlamaInputs {
  operations::ValueId tokens;    // I32 [row_capacity].
  operations::ValueId positions; // I32 absolute positions, not buffer offsets.
  operations::ValueId row_begin; // I64 offset within row_capacity.
  operations::ValueId
      row_count; // I64 active count; runtime bounds are external.
  operations::ValueId residual; // Mutable F32 [row_capacity, hidden].
  struct LayerState {
    operations::ValueId key_cache, value_cache;
  };
  std::vector<LayerState> layers; // Explicit distinct logical F16 cache planes.
};
enum class LlamaRegionKind { Embedding, Attention, Ffn, Layer, Layers, Output };
struct LlamaRegion {
  LlamaRegionKind kind = LlamaRegionKind::Embedding;
  size_t layer_begin = 0;
  size_t layer_count = 0; // One for Attention/Ffn/Layer; positive for Layers.
};
struct RegionPort {
  operations::ValueId value;
  operations::Access access = operations::Access::Read;
};
struct LlamaRegionView {
  size_t operation_begin = 0;
  size_t operation_end = 0;
  std::vector<RegionPort> outside;
};

// Owns immutable facts and the sole nonmovable logical graph. ValueIds are
// meaningful only while this owner (or another retaining shared_ptr) lives.
class LlamaStructure {
public:
  const model::ModelDescriptor &facts() const { return facts_; }
  const LlamaBuildOptions &options() const { return options_; }
  const operations::OperationGraph &graph() const { return graph_; }
  const LlamaInputs &inputs() const { return inputs_; }
  const std::vector<ParameterAssociation> &parameters() const {
    return parameters_;
  }
  const model::TensorBinding *parameter(ParameterCoordinate coordinate) const;
  bool select(LlamaRegion selection, LlamaRegionView *out,
              std::string *error) const;

private:
  friend class detail::StructureConstruction;
  LlamaStructure(model::ModelDescriptor facts, LlamaBuildOptions options,
                 const operations::OperationRegistry &registry);
  struct RecordedRegion {
    LlamaRegion selection;
    size_t begin, end;
  };
  model::ModelDescriptor facts_;
  LlamaBuildOptions options_;
  operations::OperationGraph graph_;
  LlamaInputs inputs_;
  std::vector<ParameterAssociation> parameters_;
  std::vector<RecordedRegion> regions_;
  std::vector<operations::ValueId> exports_;
};

// Reserved complete-model API. Implementation arrives with all component
// helpers; this checkpoint deliberately provides no successful partial factory.
bool BuildLlamaStructure(model::ModelDescriptor facts,
                         const LlamaBuildOptions &options,
                         std::shared_ptr<const LlamaStructure> *out,
                         std::string *error);
} // namespace hrx::llama
#endif
