// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/model-full/llama_step.h"

#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/template/ops.h"

namespace hrx::loomify_gguf {
namespace {
iree_status_t Index(loom_builder_t *builder, int64_t value,
                    loom_scalar_type_t type, loom_value_id_t *out) {
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(value),
                                                 loom_type_scalar(type),
                                                 LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_index_constant_result(op);
  return iree_ok_status();
}
iree_status_t Scalar(loom_builder_t *builder, loom_attribute_t value,
                     loom_scalar_type_t type, loom_value_id_t *out) {
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, value, loom_type_scalar(type), LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_scalar_constant_result(op);
  return iree_ok_status();
}
loom_predicate_t Range(loom_value_id_t value, int64_t min, int64_t max) {
  loom_predicate_t result = {};
  result.kind = LOOM_PREDICATE_RANGE;
  result.arg_count = 3;
  result.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  result.arg_tags[1] = result.arg_tags[2] = LOOM_PRED_ARG_CONST;
  result.args[0] = value;
  result.args[1] = min;
  result.args[2] = max;
  return result;
}
iree_status_t Count(loom_builder_t *builder, loom_value_id_t value,
                    loom_value_id_t *out) {
  const auto range = Range(value, 1, 2048);
  const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t *op = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(
      builder, &value, 1, &range, 1, &index, 1, LOOM_LOCATION_UNKNOWN, &op));
  *out = loom_op_results(op)[0];
  return iree_ok_status();
}
} // namespace

loomc_status_t DefineLlamaRmsNorm(ModuleBuilder &module, std::string_view name,
                                  KernelRef *out) {
  if (!out || name.empty() || name.find('\0') != std::string_view::npos)
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "RMSNorm name and output required");
  *out = {};
  auto status = module.BuildNative([&](loom_module_t *native,
                                       loom_builder_t *top) {
    // Select the existing numerical motif by its native template contract.
    loom_symbol_id_t motif = LOOM_SYMBOL_ID_INVALID;
    for (size_t i = 0; i < native->symbols.count; ++i) {
      auto actual = loom_string_table_get(&native->strings,
                                          native->symbols.entries[i].name_id);
      if (std::string_view(actual.data, actual.size) == "llm.rmsnorm.row") {
        motif = static_cast<loom_symbol_id_t>(i);
        break;
      }
    }
    if (motif == LOOM_SYMBOL_ID_INVALID)
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "missing llm.rmsnorm.row motif");
    loom_string_id_t text = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        native, iree_make_string_view(name.data(), name.size()), &text));
    if (loom_module_find_symbol(native, text) != LOOM_SYMBOL_ID_INVALID)
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "RMSNorm symbol already exists");
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(native, text, &symbol));
    const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    const auto offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    const loom_type_t arguments[] = {index, loom_type_buffer(),
                                     loom_type_buffer(), loom_type_buffer()};
    loom_op_t *definition = nullptr;
    IREE_RETURN_IF_ERROR(loom_kernel_def_build(
        top, 0, 0, 0, loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, 0,
        {0, symbol}, &index, 1, arguments, 4, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &definition));
    auto config = *top;
    auto *config_region = loom_kernel_def_config(definition);
    loom_builder_enter_region(&config, definition, config_region);
    loom_value_id_t count;
    IREE_RETURN_IF_ERROR(Count(
        &config, loom_region_entry_block(config_region)->arg_ids[0], &count));
    loom_value_id_t one, threads;
    IREE_RETURN_IF_ERROR(Index(&config, 1, LOOM_SCALAR_TYPE_INDEX, &one));
    IREE_RETURN_IF_ERROR(Index(&config, 256, LOOM_SCALAR_TYPE_INDEX, &threads));
    loom_op_t *op = nullptr;
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_build(
        &config, 0, one, count, one, threads, one, one, LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_LOCATION_UNKNOWN,
        &op));
    auto body = *top;
    auto *body_region = loom_kernel_def_body(definition);
    loom_builder_enter_region(&body, definition, body_region);
    const auto *args = loom_region_entry_block(body_region)->arg_ids;
    IREE_RETURN_IF_ERROR(Count(&body, args[0], &count));
    loom_value_id_t hidden, divisor, epsilon, half, row_bytes;
    IREE_RETURN_IF_ERROR(Index(&body, 3072, LOOM_SCALAR_TYPE_INDEX, &hidden));
    IREE_RETURN_IF_ERROR(
        Scalar(&body, loom_attr_f64(3072), LOOM_SCALAR_TYPE_F32, &divisor));
    IREE_RETURN_IF_ERROR(
        Scalar(&body, loom_attr_f64(1e-5), LOOM_SCALAR_TYPE_F32, &epsilon));
    IREE_RETURN_IF_ERROR(
        Scalar(&body, loom_attr_i64(0), LOOM_SCALAR_TYPE_I1, &half));
    IREE_RETURN_IF_ERROR(
        Index(&body, 3072 * 4, LOOM_SCALAR_TYPE_OFFSET, &row_bytes));
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_id_build(
        &body, LOOM_KERNEL_DIMENSION_Y, index, LOOM_LOCATION_UNKNOWN, &op));
    auto raw_row = loom_kernel_workgroup_id_result(op);
    loom_predicate_t row_predicates[] = {Range(raw_row, 0, 2047), {}};
    row_predicates[1].kind = LOOM_PREDICATE_LT;
    row_predicates[1].arg_count = 2;
    row_predicates[1].arg_tags[0] = row_predicates[1].arg_tags[1] =
        LOOM_PRED_ARG_VALUE;
    row_predicates[1].args[0] = raw_row;
    row_predicates[1].args[1] = count;
    IREE_RETURN_IF_ERROR(loom_index_assume_build(&body, &raw_row, 1,
                                                 row_predicates, 2, &index, 1,
                                                 LOOM_LOCATION_UNKNOWN, &op));
    auto row = loom_op_results(op)[0];
    IREE_RETURN_IF_ERROR(loom_index_scale_build(&body, row, row_bytes, offset,
                                                LOOM_LOCATION_UNKNOWN, &op));
    auto origin = loom_index_scale_result(op);
    const loom_value_id_t operands[] = {hidden,  divisor, count,   row,
                                        origin,  half,    epsilon, args[1],
                                        args[2], args[3]};
    IREE_RETURN_IF_ERROR(loom_template_apply_build(
        &body, 0, {0, motif}, operands, 10, 0, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_kernel_return_build(&body, LOOM_LOCATION_UNKNOWN, &op);
  });
  if (!loomc_status_is_ok(status))
    return status;
  return module.Kernel(name, out);
}
} // namespace hrx::loomify_gguf
