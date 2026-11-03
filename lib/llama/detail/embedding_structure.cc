// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/embedding_structure.h"

#include <array>
#include <limits>

namespace hrx::llama::detail {
namespace {
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
} // namespace

bool AppendEmbedding(StructureConstruction *owner, std::string *error) {
  using namespace operations;
  if (!owner)
    return Fail(error, "Llama embedding requires a construction owner");
  const auto &structure = owner->structure();
  const auto &facts = structure.facts();
  const auto &inputs = structure.inputs();
  auto &graph = owner->graph();
  if (!inputs.residual.valid() || !graph.definition(OpCode::Embedding))
    return Fail(error,
                "Llama embedding requires inputs and Embedding definition");
  LlamaRegionView existing;
  std::string ignored;
  if (structure.select({LlamaRegionKind::Embedding, 0, 0}, &existing, &ignored))
    return Fail(error, "Llama embedding region already exists");
  const auto max = uint64_t(std::numeric_limits<int64_t>::max());
  if (!facts.embedding_width || facts.embedding_width > max ||
      !facts.vocabulary_size || facts.vocabulary_size > max)
    return Fail(error,
                "Llama embedding requires positive representable dimensions");
  size_t matches = 0, slot = 0;
  for (size_t i = 0; i < facts.globals.size(); ++i)
    if (facts.globals[i].role == "token_embedding") {
      ++matches;
      slot = i;
    }
  if (matches != 1)
    return Fail(error,
                "Llama embedding requires exactly one token_embedding role");
  const auto &tensor = facts.globals[slot].tensor;
  Encoding encoding;
  if (tensor.normalized_extents !=
      std::array<int64_t, 4>{static_cast<int64_t>(facts.embedding_width),
                             static_cast<int64_t>(facts.vocabulary_size), 1, 1})
    return Fail(error, "Llama embedding has incompatible GGML axes");
  if (!MatrixEncoding(tensor.storage_type, &encoding))
    return Fail(error, "Llama embedding requires Q4_K, Q6_K or Q8_0 storage");
  ValueId weight;
  if (!owner->resolve({ParameterScope::Global, 0, slot}, &weight, error))
    return false;
  const auto begin = graph.invocations().size();
  if (!graph.operation(OpCode::Embedding)
           .bind(Port::TokenIds, inputs.tokens)
           .bind(Port::Weight, weight)
           .bind(Port::Output, inputs.residual)
           .bind(Port::RowBegin, inputs.row_begin)
           .bind(Port::RowCount, inputs.row_count)
           .attributes(
               EmbeddingAttrs{facts.vocabulary_size, facts.embedding_width})
           .append(error))
    return false;
  return owner->record({LlamaRegionKind::Embedding, 0, 0}, begin, begin + 1,
                       error);
}
} // namespace hrx::llama::detail
