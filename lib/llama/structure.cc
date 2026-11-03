// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/structure_construction.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace hrx::llama {
namespace {
using namespace operations;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
bool Same(ParameterCoordinate a, ParameterCoordinate b) {
  return a.scope == b.scope && a.layer_slot == b.layer_slot &&
         a.tensor_slot == b.tensor_slot;
}
bool Contains(const std::vector<ValueId> &values, ValueId value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}
} // namespace
LlamaStructure::LlamaStructure(model::ModelDescriptor facts,
                               LlamaBuildOptions options,
                               const OperationRegistry &registry)
    : facts_(std::move(facts)), options_(std::move(options)), graph_(registry) {
}
const model::TensorBinding *
LlamaStructure::parameter(ParameterCoordinate c) const {
  if (c.scope == ParameterScope::Global)
    return c.layer_slot == 0 && c.tensor_slot < facts_.globals.size()
               ? &facts_.globals[c.tensor_slot]
               : nullptr;
  if (c.scope != ParameterScope::Layer || c.layer_slot >= facts_.layers.size())
    return nullptr;
  const auto &tensors = facts_.layers[c.layer_slot].tensors;
  return c.tensor_slot < tensors.size() ? &tensors[c.tensor_slot] : nullptr;
}
bool LlamaStructure::select(LlamaRegion s, LlamaRegionView *out,
                            std::string *error) const {
  if (!out)
    return Fail(error, "region requires output");
  const bool layered =
      s.kind == LlamaRegionKind::Attention || s.kind == LlamaRegionKind::Ffn ||
      s.kind == LlamaRegionKind::Layer || s.kind == LlamaRegionKind::Layers;
  if (layered ? (!s.layer_count || s.layer_begin >= facts_.layers.size() ||
                 s.layer_count > facts_.layers.size() - s.layer_begin ||
                 (s.kind != LlamaRegionKind::Layers && s.layer_count != 1))
              : (s.layer_begin != 0 || s.layer_count != 0))
    return Fail(error, "invalid region layer range");
  LlamaRegionView staged;
  bool found = false;
  for (size_t offset = 0;
       offset < (s.kind == LlamaRegionKind::Layers ? s.layer_count : 1);
       ++offset) {
    auto kind =
        s.kind == LlamaRegionKind::Layers ? LlamaRegionKind::Layer : s.kind;
    const RecordedRegion *match = nullptr;
    for (const auto &r : regions_)
      if (r.selection.kind == kind &&
          r.selection.layer_begin == s.layer_begin + offset)
        match = &r;
    if (!match || (found && staged.operation_end != match->begin))
      return Fail(error, "region is absent or layers are disconnected");
    if (!found)
      staged.operation_begin = match->begin;
    staged.operation_end = match->end;
    found = true;
  }
  // Reads of an invocation precede all its writes. No SSA versions: a write
  // defines this mutable identity for subsequent ordered invocations.
  // These are the existing operations' bounded row shapes, not symbolic
  // intervals. Runtime scalar equality/bounds are never inferred.
  enum class Rows {
    Whole,
    Active,
    LastActive,
    LeadingCount,
    RowZero,
    CachePositions,
    CachePrefix
  };
  struct Definition {
    ValueId value;
    Rows rows;
    ValueId begin, count, positions;
  };
  auto control = [](const Invocation &op, Port port) {
    auto *binding = op.binding(port);
    return binding ? binding->value() : ValueId{};
  };
  auto domain = [&](const Invocation &op, const PortBinding &binding) {
    Definition result{binding.value(), Rows::Whole, control(op, Port::RowBegin),
                      control(op, Port::RowCount),
                      control(op, Port::Positions)};
    switch (graph_.definition(op.opcode())->port(binding.port())->domain()) {
    case AccessDomain::Whole:
      break;
    case AccessDomain::ActiveRows:
      if (result.begin.valid())
        result.rows = Rows::Active;
      break;
    case AccessDomain::ActiveTokenPositions:
      result.rows = Rows::CachePositions;
      break;
    case AccessDomain::CausalPrefix:
      result.rows = Rows::CachePrefix;
      break;
    case AccessDomain::CopyRowsSource:
      result.rows = std::get<CopyRowsAttrs>(op.attributes()).selection ==
                            RowSelection::LastActive
                        ? Rows::LastActive
                        : Rows::Active;
      break;
    case AccessDomain::CopyRowsDestination:
      if (std::get<CopyRowsAttrs>(op.attributes()).selection ==
          RowSelection::LastActive) {
        // LastActive rebases to row zero. A one-row destination is fully
        // defined and can feed the legacy full-row one-row head operations
        // directly.
        auto *type = graph_.value(binding.value())->tensorType();
        result.rows = type->dimension(0) == 1 ? Rows::Whole : Rows::RowZero;
      } else
        result.rows = Rows::LeadingCount;
      break;
    }
    return result;
  };
  auto covers = [](const Definition &defined, const Definition &read) {
    if (defined.value != read.value)
      return false;
    if (defined.rows == Rows::Whole)
      return true;
    if (defined.rows == Rows::Active &&
        (read.rows == Rows::Active || read.rows == Rows::LastActive))
      return defined.begin == read.begin && defined.count == read.count;
    if (defined.rows != read.rows)
      return false;
    // Publication never proves a causal prefix initialized. Cache coverage
    // requires the identical coordinate source and query selection.
    if (defined.rows == Rows::CachePositions ||
        defined.rows == Rows::CachePrefix)
      return defined.positions == read.positions &&
             defined.begin == read.begin && defined.count == read.count;
    if (defined.rows == Rows::LastActive)
      return defined.begin == read.begin && defined.count == read.count;
    if (defined.rows == Rows::LeadingCount)
      return defined.count == read.count;
    return defined.rows == Rows::RowZero;
  };
  std::vector<Definition> definitions;
  std::vector<ValueId> written, required;
  for (size_t i = staged.operation_begin; i < staged.operation_end; ++i) {
    const auto &op = graph_.invocations()[i];
    for (const auto &b : op.bindings()) {
      const auto access =
          graph_.definition(op.opcode())->port(b.port())->access();
      if (access == Access::Write)
        continue;
      auto read = domain(op, b);
      bool covered = false;
      for (const auto &d : definitions)
        covered |= covers(d, read);
      if (!covered && !Contains(required, b.value()))
        required.push_back(b.value());
    }
    for (const auto &b : op.bindings()) {
      const auto access =
          graph_.definition(op.opcode())->port(b.port())->access();
      if (access == Access::Read)
        continue;
      // Write leaves inactive elements undefined, so prior coverage is stale.
      // Only ReadWrite preserves definitions outside the current domain.
      if (access == Access::Write)
        definitions.erase(std::remove_if(definitions.begin(), definitions.end(),
                                         [&](const Definition &d) {
                                           return d.value == b.value();
                                         }),
                          definitions.end());
      definitions.push_back(domain(op, b));
      if (!Contains(written, b.value()))
        written.push_back(b.value());
    }
  }
  std::vector<ValueId> live;
  for (auto value : written) {
    bool needed = false, killed = false;
    for (size_t i = staged.operation_end; i < graph_.invocations().size();
         ++i) {
      const auto &op = graph_.invocations()[i];
      bool read = false, write = false;
      for (const auto &b : op.bindings())
        if (b.value() == value) {
          auto access =
              graph_.definition(op.opcode())->port(b.port())->access();
          read |= access != Access::Write;
          write |= access == Access::Write;
        }
      // Reads precede writes within one operation. Unknown different runtime
      // spans may overlap; never infer disjointness from unequal ValueIds.
      if (read) {
        needed = true;
        break;
      }
      // A write-only operation defines its domain and undefines inactive rows;
      // this kills all prior contents, including matching active-span versions.
      if (write) {
        killed = true;
        break;
      }
    }
    if (needed || (!killed && Contains(exports_, value)))
      live.push_back(value);
  }
  for (auto value : required)
    staged.outside.push_back(
        {value, Contains(live, value) ? Access::ReadWrite : Access::Read});
  for (auto value : live)
    if (!Contains(required, value))
      staged.outside.push_back({value, Access::Write});
  *out = std::move(staged);
  return true;
}
namespace detail {
bool MatrixEncoding(uint32_t storage, Encoding *out) {
  if (!out)
    return false;
  switch (storage) {
  case 8:
    *out = Encoding::Q8_0;
    return true;
  case 12:
    *out = Encoding::Q4_K;
    return true;
  case 14:
    *out = Encoding::Q6_K;
    return true;
  default:
    return false;
  }
}
std::unique_ptr<StructureConstruction> StructureConstruction::Create(
    model::ModelDescriptor facts, LlamaBuildOptions options,
    const OperationRegistry &registry, std::string *error) {
  const auto max = uint64_t(std::numeric_limits<int64_t>::max());
  if (facts.family != "llama" || !facts.embedding_width ||
      facts.embedding_width > max || !facts.vocabulary_size ||
      facts.vocabulary_size > max || facts.layers.empty() ||
      !std::isfinite(facts.rms_epsilon) || facts.rms_epsilon <= 0 ||
      (options.phase != Phase::Prefill && options.phase != Phase::Decode) ||
      !options.row_capacity || options.row_capacity > max ||
      !options.cache_capacity || options.cache_capacity > max ||
      options.row_capacity > options.cache_capacity ||
      !options.history_capacity || options.history_capacity > max ||
      options.source_asset.empty()) {
    Fail(error, "invalid Llama facts, phase, capacities or source asset");
    return nullptr;
  }
  for (size_t i = 0; i < facts.layers.size(); ++i) {
    const auto &layer = facts.layers[i];
    if (layer.kind != model::LayerKind::kAttention ||
        !layer.feed_forward_width || layer.feed_forward_width > max ||
        !layer.query_heads || !layer.kv_heads ||
        layer.query_heads % layer.kv_heads || !facts.key_head_width ||
        facts.key_head_width != facts.value_head_width ||
        layer.query_heads > max / facts.key_head_width ||
        layer.kv_heads > max / facts.key_head_width) {
      Fail(error, "invalid dense Llama layer geometry");
      return nullptr;
    }
    for (size_t j = 0; j < i; ++j)
      if (facts.layers[j].index == layer.index) {
        Fail(error, "duplicate descriptor layer index");
        return nullptr;
      }
  }
  auto owner = std::shared_ptr<LlamaStructure>(
      new LlamaStructure(std::move(facts), std::move(options), registry));
  return std::unique_ptr<StructureConstruction>(
      new StructureConstruction(std::move(owner)));
}
bool StructureConstruction::createInputs(std::string *error) {
  if (owner_->inputs_.residual.valid())
    return Fail(error, "inputs already created");
  auto &g = owner_->graph_;
  const auto &o = owner_->options_;
  const auto &f = owner_->facts_;
  LlamaInputs inputs;
  inputs.tokens =
      g.tensor(TensorType::vector(o.row_capacity, ElementType::I32), error);
  inputs.positions =
      g.tensor(TensorType::vector(o.row_capacity, ElementType::I32), error);
  inputs.row_begin = g.scalar(ScalarType::I64, error);
  inputs.row_count = g.scalar(ScalarType::I64, error);
  inputs.residual = g.tensor(
      TensorType::matrix(o.row_capacity, f.embedding_width, ElementType::F32),
      error);
  if (!inputs.tokens.valid() || !inputs.positions.valid() ||
      !inputs.row_begin.valid() || !inputs.row_count.valid() ||
      !inputs.residual.valid())
    return false;
  for (const auto &layer : f.layers) {
    auto type =
        TensorType::matrix(o.cache_capacity, layer.kv_heads, ElementType::F16);
    type.addDimension(f.key_head_width);
    auto k = g.tensor(type, error), v = g.tensor(type, error);
    if (!k.valid() || !v.valid())
      return false;
    inputs.layers.push_back({k, v});
  }
  owner_->inputs_ = std::move(inputs);
  return true;
}
bool StructureConstruction::resolve(ParameterCoordinate c, ValueId *out,
                                    std::string *error) {
  const auto *binding = owner_->parameter(c);
  if (!binding || !out || binding->tensor.name.empty())
    return Fail(error, "invalid parameter coordinate or identity");
  const auto &t = binding->tensor;
  for (const auto &a : owner_->parameters_) {
    const auto &prior = owner_->parameter(a.coordinate)->tensor;
    if (prior.name != t.name)
      continue; // All records belong to options.source_asset.
    if (prior.normalized_extents != t.normalized_extents ||
        prior.storage_type != t.storage_type ||
        prior.absolute_offset != t.absolute_offset ||
        prior.byte_size != t.byte_size)
      return Fail(error, "conflicting repeated parameter identity");
    auto value = a.value;
    bool associated = false;
    for (const auto &existing : owner_->parameters_)
      associated |= Same(existing.coordinate, c);
    if (!associated)
      owner_->parameters_.push_back({value, c});
    *out = value;
    return true;
  }
  const auto &e = t.normalized_extents;
  if (e[0] <= 0 || e[1] <= 0 || e[2] != 1 || e[3] != 1)
    return Fail(error, "parameter requires vector or matrix GGML axes");
  TensorType type;
  if (t.storage_type == 0 && e[1] == 1)
    type = TensorType::vector(e[0], ElementType::F32);
  else {
    Encoding encoding;
    if (!MatrixEncoding(t.storage_type, &encoding))
      return Fail(error, "matrix requires Q4_K, Q6_K or Q8_0 storage");
    type = TensorType::matrix(e[0], e[1], ElementType::F32)
               .setEncoding(encoding)
               .setLayout(Layout::GgmlWeightRows);
  }
  auto value = graph().tensor(type, error);
  if (!value.valid())
    return false;
  owner_->parameters_.push_back({value, c});
  *out = value;
  return true;
}
bool StructureConstruction::record(LlamaRegion region, size_t begin, size_t end,
                                   std::string *error) {
  if (region.kind == LlamaRegionKind::Layers || begin >= end ||
      end > graph().invocations().size())
    return Fail(error, "invalid recorded region bounds");
  const bool layered = region.kind == LlamaRegionKind::Attention ||
                       region.kind == LlamaRegionKind::Ffn ||
                       region.kind == LlamaRegionKind::Layer;
  if (layered ? (region.layer_count != 1 ||
                 region.layer_begin >= owner_->facts_.layers.size())
              : ((region.kind != LlamaRegionKind::Embedding &&
                  region.kind != LlamaRegionKind::Output) ||
                 region.layer_begin || region.layer_count))
    return Fail(error, "invalid recorded region selector");
  for (const auto &r : owner_->regions_)
    if (r.selection.kind == region.kind &&
        r.selection.layer_begin == region.layer_begin)
      return Fail(error, "duplicate known region");
  owner_->regions_.push_back({region, begin, end});
  return true;
}
bool StructureConstruction::exportValue(ValueId value, std::string *error) {
  if (!graph().value(value))
    return Fail(error, "foreign declared export");
  if (!Contains(owner_->exports_, value))
    owner_->exports_.push_back(value);
  return true;
}
} // namespace detail
} // namespace hrx::llama
