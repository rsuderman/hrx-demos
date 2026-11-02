// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LIB_LLAMA_DETAIL_FFN_STRUCTURE_H_
#define HRX_SOLUTIONS_LIB_LLAMA_DETAIL_FFN_STRUCTURE_H_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "lib/operations/operation_graph.h"
#include "tools/model/model_descriptor.h"

namespace hrx::llama::detail {

// Indices refer to the caller-owned descriptor, never a copied tensor schema or
// native parameter offset. The graph owner and descriptor must be retained.
struct FfnParameterAssociation {
  operations::ValueId value;
  size_t layer_slot = 0;
  size_t tensor_slot = 0;
};

struct NormalizedFfnRegion {
  size_t layer_slot = 0;
  size_t operation_begin = 0;
  size_t operation_end = 0; // Exclusive; exactly three logical operations.
  operations::ValueId
      residual; // One mutable input/output, not a new SSA value.
  operations::ValueId normalized;
  operations::ValueId activation;
  std::vector<FfnParameterAssociation> parameters;
};

// Internal production append helper for the eventual complete model builder.
// layer_slot selects facts.layers by vector index, not LayerDescriptor::index.
// Required definitions are RmsNorm, GatedMatmul and LinearResidual. The caller
// owns a private graph while building and publishes it only after all appends
// succeed. Failure leaves *region unchanged; discard a graph after an append
// failure because the graph does not support multi-operation rollback.
// Logical structure only: no kernel/provider admission, packing or byte sizing.
bool AppendNormalizedFfn(operations::OperationGraph *graph,
                         const model::ModelDescriptor &facts, size_t layer_slot,
                         operations::ValueId residual,
                         NormalizedFfnRegion *region, std::string *error);

// Owner-resolved weights in post_norm/gate/up/down order. Each value must match
// the descriptor binding; aliases intentionally retain the same ValueId. Paired
// row controls are propagated unchanged to all three logical operations.
bool AppendNormalizedFfnResolved(
    operations::OperationGraph *graph, const model::ModelDescriptor &facts,
    size_t layer_slot, operations::ValueId residual,
    const std::array<operations::ValueId, 4> &weights,
    operations::ValueId row_begin, operations::ValueId row_count,
    NormalizedFfnRegion *region, std::string *error);

} // namespace hrx::llama::detail

#endif // HRX_SOLUTIONS_LIB_LLAMA_DETAIL_FFN_STRUCTURE_H_
