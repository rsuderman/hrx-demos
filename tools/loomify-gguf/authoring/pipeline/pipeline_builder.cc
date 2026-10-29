// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/pipeline/pipeline_builder.h"

#include <limits>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"

namespace hrx::loomify_gguf {
namespace {
iree_status_t Invalid(const char *message) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message);
}

iree_status_t Find(loom_module_t *module, std::string_view name,
                   loom_symbol_ref_t *out) {
  for (size_t i = 0; i < module->symbols.count; ++i) {
    auto text = loom_string_table_get(&module->strings,
                                      module->symbols.entries[i].name_id);
    if (iree_string_view_equal(
            text, iree_make_string_view(name.data(), name.size()))) {
      *out = {0, static_cast<loom_symbol_id_t>(i)};
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "pipeline target not found");
}
} // namespace

PipelineBody::PipelineBody(loom_module_t *module, loom_builder_t builder,
                           const std::vector<loom_value_id_t> &args,
                           size_t specializations, bool strand)
    : module_(module), builder_(builder), args_(args),
      specializations_(specializations), strand_(strand) {}

loom_value_id_t PipelineBody::Specialization(size_t index) const {
  return index < specializations_ ? args_[index] : LOOM_VALUE_ID_INVALID;
}
loom_value_id_t PipelineBody::Binding(size_t index) const {
  return index < args_.size() - specializations_
             ? args_[specializations_ + index]
             : LOOM_VALUE_ID_INVALID;
}

iree_status_t PipelineBuilder::Define(loom_module_t *module,
                                      loom_builder_t *top,
                                      const PipelineInterface &interface,
                                      const PipelineBodyCallback &callback,
                                      PipelineRef *out) {
  if (out)
    *out = {};
  if (!module || !top || top->module != module || !out || !callback ||
      interface.name.empty() ||
      (interface.dependent_run && !interface.run.empty()) ||
      interface.name.find('\0') != std::string_view::npos) {
    return Invalid(
        "module, matching builder, name, callback and output required");
  }
  if (interface.specializations.size() + interface.run.size() >
      std::numeric_limits<uint16_t>::max()) {
    return Invalid("too many pipeline arguments");
  }
  loom_symbol_ref_t target = loom_symbol_ref_null();
  if (!interface.target.empty()) {
    IREE_RETURN_IF_ERROR(Find(module, interface.target, &target));
  }
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module,
      iree_make_string_view(interface.name.data(), interface.name.size()),
      &name));
  if (loom_module_find_symbol(module, name) != LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "pipeline symbol already exists");
  }
  loom_symbol_id_t id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &id));
  const loom_symbol_ref_t symbol = {0, id};
  uint32_t flags = 0;
  if (interface.kernel_scope)
    flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_SCOPE;
  if (interface.is_public)
    flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_VISIBILITY;
  if (!interface.target.empty())
    flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_TARGET;
  loom_op_t *definition = nullptr;
  IREE_RETURN_IF_ERROR(loom_pipeline_def_build(
      top, flags, LOOM_PIPELINE_SCOPE_KERNEL, LOOM_PIPELINE_VISIBILITY_PUBLIC,
      0, target, symbol, interface.specializations.data(),
      interface.specializations.size(), interface.run.data(),
      interface.run.size(), nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
  auto builder = *top;
  auto *region = loom_pipeline_def_body(definition);
  loom_builder_enter_region(&builder, definition, region);
  auto *block = loom_region_entry_block(region);
  std::vector<loom_value_id_t> args;
  if (block->arg_count)
    args.assign(block->arg_ids, block->arg_ids + block->arg_count);
  if (interface.dependent_run) {
    std::vector<loom_type_t> types;
    IREE_RETURN_IF_ERROR(interface.dependent_run(module, args, &types));
    if (types.size() > std::numeric_limits<uint16_t>::max() - args.size()) {
      return Invalid("too many dependent run arguments");
    }
    for (auto type : types) {
      loom_value_id_t value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_define_value(module, type, &value));
      IREE_RETURN_IF_ERROR(loom_block_add_arg(module, block, value));
      args.push_back(value);
    }
  }
  PipelineBody body(module, builder, args, interface.specializations.size(),
                    false);
  IREE_RETURN_IF_ERROR(callback(body));
  loom_op_t *finish = nullptr;
  IREE_RETURN_IF_ERROR(loom_pipeline_finish_build(
      body.Native(), LOOM_LOCATION_UNKNOWN, &finish));
  out->module_ = module;
  out->symbol_ = symbol;
  return iree_ok_status();
}

iree_status_t PipelineBody::Strand(const PipelineWorkers &workers,
                                   const PipelineBodyCallback &callback) {
  if (strand_ || !callback || workers.origins.empty() ||
      workers.origins.size() != workers.counts.size() ||
      workers.origins.size() != workers.strides.size()) {
    return Invalid(
        "strand requires equal nonempty worker dimensions outside strand");
  }
  for (size_t i = 0; i < workers.origins.size(); ++i) {
    if (workers.origins[i] < 0 || workers.counts[i] <= 0 ||
        workers.strides[i] <= 0)
      return Invalid("invalid worker domain");
  }
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      &builder_, 0, loom_symbol_ref_null(), nullptr, 0, workers.origins.data(),
      workers.origins.size(), nullptr, 0, workers.counts.data(),
      workers.counts.size(), nullptr, 0, workers.strides.data(),
      workers.strides.size(), LOOM_LOCATION_UNKNOWN, &op));
  auto nested = builder_;
  loom_builder_enter_region(&nested, op, loom_pipeline_strand_body(op));
  PipelineBody body(module_, nested, args_, specializations_, true);
  IREE_RETURN_IF_ERROR(callback(body));
  loom_op_t *end = nullptr;
  return loom_pipeline_end_build(body.Native(), LOOM_LOCATION_UNKNOWN, &end);
}

iree_status_t
PipelineBody::Compose(PipelineRef child,
                      const std::vector<loom_value_id_t> &specializations,
                      const std::vector<loom_value_id_t> &bindings) {
  if (strand_ || child.module_ != module_) {
    return Invalid("composition requires a same-module child outside strands");
  }
  loom_op_t *op = nullptr;
  return loom_pipeline_compose_build(
      &builder_, child.symbol_, specializations.data(), specializations.size(),
      bindings.data(), bindings.size(), LOOM_LOCATION_UNKNOWN, &op);
}

iree_status_t PipelineBody::IndexConstant(int64_t value, loom_value_id_t *out) {
  if (!out)
    return Invalid("missing output");
  *out = LOOM_VALUE_ID_INVALID;
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &builder_, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_index_constant_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::OffsetConstant(int64_t value,
                                           loom_value_id_t *out) {
  if (!out)
    return Invalid("missing output");
  *out = LOOM_VALUE_ID_INVALID;
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &builder_, loom_attr_i64(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_index_constant_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::View(loom_value_id_t buffer,
                                 loom_value_id_t byte_offset, loom_type_t type,
                                 loom_value_id_t *out) {
  if (!out)
    return Invalid("missing output");
  *out = LOOM_VALUE_ID_INVALID;
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      &builder_, buffer, byte_offset, type, LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_buffer_view_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::MemoryPool(loom_value_fact_memory_space_t space,
                                       const std::vector<int64_t> &coordinates,
                                       loom_value_id_t *out) {
  if (strand_ || !out)
    return Invalid("pool selection belongs outside strands");
  *out = LOOM_VALUE_ID_INVALID;
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_pipeline_memory_build(
      &builder_, space, nullptr, 0, coordinates.data(), coordinates.size(),
      loom_type_pool(), LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_pipeline_memory_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::Allocate(loom_value_id_t pool,
                                     loom_value_fact_memory_space_t space,
                                     int64_t alignment, loom_value_id_t bytes,
                                     loom_value_id_t *out) {
  if (strand_ || !out || alignment <= 0 || (alignment & (alignment - 1)) != 0)
    return Invalid("invalid storage allocation");
  *out = LOOM_VALUE_ID_INVALID;
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      &builder_, LOOM_BUFFER_ALLOCA_BUILD_FLAG_HAS_POOL, space, pool, alignment,
      bytes, loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_buffer_alloca_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::BindFifo(loom_value_id_t slots,
                                     loom_value_id_t capacity,
                                     loom_type_t payload,
                                     loom_value_id_t *out) {
  if (strand_ || !out)
    return Invalid("channel storage belongs outside strands");
  *out = LOOM_VALUE_ID_INVALID;
  loom_type_id_t payload_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(module_, payload, &payload_id));
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_channel_type_make(module_, payload_id, &type));
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_channel_bind_build(
      &builder_, LOOM_CHANNEL_BIND_DISCIPLINE_FIFO, slots, capacity, type,
      LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_channel_bind_result(op);
  return iree_ok_status();
}

iree_status_t PipelineBody::ChannelTypes(loom_value_id_t channel,
                                         loom_type_t *ownership,
                                         loom_type_t *view, bool write) {
  if (!strand_ || channel >= module_->values.count) {
    return Invalid("channel access requires a valid channel inside a strand");
  }
  const auto type = loom_module_value_type(module_, channel);
  if (!loom_channel_type_isa(type))
    return Invalid("expected a channel");
  const auto payload_id = loom_channel_type_payload(type);
  if (payload_id >= module_->types.count)
    return Invalid("invalid payload type");
  const auto payload = loom_type_table_get(&module_->types, payload_id);
  if (!loom_type_is_tile(payload))
    return Invalid("expected tile payload");
  *view = payload;
  view->header =
      loom_type_make_header(LOOM_TYPE_VIEW, loom_type_element_type(payload),
                            loom_type_rank(payload), loom_type_flags(payload));
  if (write)
    return loom_write_type_make(module_, payload_id, ownership);
  return loom_read_type_make(module_, 0, payload_id,
                             static_cast<loom_read_type_mode_t>(0), ownership);
}

iree_status_t PipelineBody::Reserve(loom_value_id_t channel,
                                    ChannelAccess *out) {
  if (!out)
    return Invalid("missing output");
  *out = {};
  loom_type_t write, view;
  IREE_RETURN_IF_ERROR(ChannelTypes(channel, &write, &view, true));
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
      &builder_, channel, write, view, LOOM_LOCATION_UNKNOWN, &op));
  *out = {loom_channel_reserve_write(op), loom_channel_reserve_view(op)};
  return iree_ok_status();
}

iree_status_t PipelineBody::Publish(loom_value_id_t write) {
  if (!strand_)
    return Invalid("publication requires a strand");
  loom_op_t *op = nullptr;
  return loom_channel_publish_build(&builder_, write, LOOM_LOCATION_UNKNOWN,
                                    &op);
}

iree_status_t PipelineBody::Acquire(loom_value_id_t channel,
                                    ChannelAccess *out) {
  if (!out)
    return Invalid("missing output");
  *out = {};
  loom_type_t read, view;
  IREE_RETURN_IF_ERROR(ChannelTypes(channel, &read, &view, false));
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
      &builder_, 0, 0, channel, read, view, LOOM_LOCATION_UNKNOWN, &op));
  *out = {loom_channel_acquire_read(op), loom_channel_acquire_view(op)};
  return iree_ok_status();
}

iree_status_t PipelineBody::Release(loom_value_id_t read) {
  if (!strand_)
    return Invalid("release requires a strand");
  loom_op_t *op = nullptr;
  return loom_channel_release_build(&builder_, read, LOOM_LOCATION_UNKNOWN,
                                    &op);
}

} // namespace hrx::loomify_gguf
