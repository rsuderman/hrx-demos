// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_QWEN_STRUCTURE_H_
#define HRX_SOLUTIONS_LIB_QWEN_STRUCTURE_H_
#include "lib/operations/operation_graph.h"
#include "tools/model/model_descriptor.h"
#include <memory>
namespace hrx::qwen {
struct QwenBuildOptions {
  uint64_t token_capacity = 0, span_capacity = 1, selected_output_capacity = 1;
  uint64_t cache_capacity = 0;
  std::string source_asset;
};
struct ParameterCoordinate {
  bool global = true;
  size_t layer_slot = 0, tensor_slot = 0;
};
struct ParameterAssociation {
  operations::ValueId value;
  ParameterCoordinate coordinate;
};
struct QwenInputs {
  operations::ValueId tokens, residual, epoch, rows, output;
  struct LayerState {
    size_t layer_slot = 0;
    uint32_t original_layer_index = 0;
    uint64_t ordinal = 0;
    model::LayerKind kind = model::LayerKind::kAttention;
    operations::ValueId history, recurrent, key_cache, value_cache;
  };
  std::vector<LayerState> layers;
};
class QwenStructure {
public:
  const model::ModelDescriptor &facts() const { return facts_; }
  const QwenBuildOptions &options() const { return options_; }
  const operations::OperationGraph &graph() const { return graph_; }
  const QwenInputs &inputs() const { return inputs_; }
  const std::vector<ParameterAssociation> &parameters() const {
    return parameters_;
  }
  const model::TensorBinding *parameter(ParameterCoordinate) const;

private:
  friend bool BuildQwenStructure(model::ModelDescriptor,
                                 const QwenBuildOptions &,
                                 std::shared_ptr<const QwenStructure> *,
                                 std::string *);
  QwenStructure(model::ModelDescriptor, QwenBuildOptions,
                const operations::OperationRegistry &);
  model::ModelDescriptor facts_;
  QwenBuildOptions options_;
  operations::OperationGraph graph_;
  QwenInputs inputs_;
  std::vector<ParameterAssociation> parameters_;
};
bool BuildQwenStructure(model::ModelDescriptor, const QwenBuildOptions &,
                        std::shared_ptr<const QwenStructure> *,
                        std::string *error);
} // namespace hrx::qwen
#endif
