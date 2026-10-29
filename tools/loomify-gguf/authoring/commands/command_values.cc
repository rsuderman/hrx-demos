// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/commands/command_values.h"

#include <cstring>
#include <functional>
#include <string>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"

namespace hrx::loomify_gguf {
namespace {
loomc_status_t Invalid(const char *message) {
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT, message);
}
iree_status_t Require(bool condition, const char *message) {
  return condition
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message);
}
bool ScalarIs(loom_type_t type, loom_scalar_type_t scalar) {
  return loom_type_is_scalar(type) && loom_type_element_type(type) == scalar;
}
bool Address(loom_type_t type) {
  return ScalarIs(type, LOOM_SCALAR_TYPE_INDEX) ||
         ScalarIs(type, LOOM_SCALAR_TYPE_OFFSET);
}
bool IntegerOrAddress(loom_type_t type) {
  if (!loom_type_is_scalar(type))
    return false;
  switch (loom_type_element_type(type)) {
  case LOOM_SCALAR_TYPE_INDEX:
  case LOOM_SCALAR_TYPE_OFFSET:
  case LOOM_SCALAR_TYPE_I1:
  case LOOM_SCALAR_TYPE_I8:
  case LOOM_SCALAR_TYPE_I16:
  case LOOM_SCALAR_TYPE_I32:
  case LOOM_SCALAR_TYPE_I64:
    return true;
  default:
    return false;
  }
}
using Check = std::function<iree_status_t(
    const loom_module_t *, const std::vector<loom_value_id_t> &)>;
loomc_status_t Preflight(CommandProgramBuilder &program,
                         const std::vector<Value> &inputs, const Check &check,
                         Value *out) {
  if (!out)
    return Invalid("missing value output");
  *out = {};
  return program.InspectNative(inputs, check);
}
iree_status_t AddressPair(const loom_module_t *module,
                          const std::vector<loom_value_id_t> &ids) {
  auto type = loom_module_value_type(module, ids[0]);
  return Require(Address(type) && loom_type_equal(type, loom_module_value_type(
                                                            module, ids[1])),
                 "operation requires matching index or offset operands");
}
iree_status_t IndexPair(const loom_module_t *module,
                        const std::vector<loom_value_id_t> &ids) {
  return Require(ScalarIs(loom_module_value_type(module, ids[0]),
                          LOOM_SCALAR_TYPE_INDEX) &&
                     ScalarIs(loom_module_value_type(module, ids[1]),
                              LOOM_SCALAR_TYPE_INDEX),
                 "operation requires index operands");
}
iree_status_t WordPair(const loom_module_t *module,
                       const std::vector<loom_value_id_t> &ids) {
  auto type = loom_module_value_type(module, ids[0]);
  return Require(
      (ScalarIs(type, LOOM_SCALAR_TYPE_I32) ||
       ScalarIs(type, LOOM_SCALAR_TYPE_I64)) &&
          loom_type_equal(type, loom_module_value_type(module, ids[1])),
      "operation requires matching i32 or i64 operands");
}
using Emit = std::function<iree_status_t(loom_module_t *, loom_builder_t *,
                                         const std::vector<loom_value_id_t> &,
                                         loom_value_id_t *)>;
loomc_status_t EmitValue(CommandProgramBuilder &program,
                         const std::vector<Value> &inputs, const Emit &emit,
                         Value *out) {
  if (!out)
    return Invalid("missing value output");
  *out = {};
  std::vector<Value> results;
  auto status = program.BuildNative(
      inputs,
      [&](loom_module_t *module, loom_builder_t *builder,
          const std::vector<loom_value_id_t> &operands,
          std::vector<loom_value_id_t> *outputs) {
        loom_value_id_t result = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(emit(module, builder, operands, &result));
        outputs->push_back(result);
        return iree_ok_status();
      },
      &results);
  if (loomc_status_is_ok(status))
    *out = results.front();
  return status;
}
} // namespace

loomc_status_t CommandValues::Constant(int64_t value, loom_scalar_type_t type,
                                       Value *out) {
  if ((type != LOOM_SCALAR_TYPE_INDEX && type != LOOM_SCALAR_TYPE_OFFSET &&
       type != LOOM_SCALAR_TYPE_I1 && type != LOOM_SCALAR_TYPE_I32 &&
       type != LOOM_SCALAR_TYPE_I64) ||
      (type == LOOM_SCALAR_TYPE_OFFSET && value < 0) ||
      (type == LOOM_SCALAR_TYPE_I1 && value != 0 && value != 1) ||
      (type == LOOM_SCALAR_TYPE_I32 &&
       (value < INT32_MIN || value > INT32_MAX))) {
    if (out)
      *out = {};
    return Invalid("constant must fit a supported integer/address scalar type");
  }
  return EmitValue(
      program_, {},
      [&](auto *, auto *builder, const auto &, auto *id) {
        loom_op_t *op = nullptr;
        if (type == LOOM_SCALAR_TYPE_INDEX || type == LOOM_SCALAR_TYPE_OFFSET) {
          IREE_RETURN_IF_ERROR(loom_index_constant_build(
              builder, loom_attr_i64(value), loom_type_scalar(type),
              LOOM_LOCATION_UNKNOWN, &op));
          *id = loom_index_constant_result(op);
        } else {
          IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
              builder, loom_attr_i64(value), loom_type_scalar(type),
              LOOM_LOCATION_UNKNOWN, &op));
          *id = loom_scalar_constant_result(op);
        }
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Config(std::string_view name, Value *out) {
  if (name.empty() || name.find('\0') != std::string_view::npos) {
    if (out)
      *out = {};
    return Invalid("config name must be nonempty");
  }
  auto status = Preflight(
      program_, {},
      [&](const auto *module, const auto &) {
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto &entry = module->symbols.entries[i];
          if (entry.name_id == LOOM_STRING_ID_INVALID ||
              entry.name_id >= module->strings.count)
            continue;
          const auto actual =
              loom_string_table_get(&module->strings, entry.name_id);
          if (actual.size == name.size() &&
              std::memcmp(actual.data, name.data(), name.size()) == 0) {
            const auto *op = entry.defining_op;
            return Require(
                op && (loom_config_def_isa(op) || loom_config_decl_isa(op)),
                "symbol is not a configuration");
          }
        }
        return Require(false, "configuration symbol not found");
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {},
      [&](auto *module, auto *builder, const auto &, auto *id) {
        loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto &entry = module->symbols.entries[i];
          if (entry.name_id == LOOM_STRING_ID_INVALID ||
              entry.name_id >= module->strings.count)
            continue;
          const auto actual =
              loom_string_table_get(&module->strings, entry.name_id);
          if (actual.size == name.size() &&
              std::memcmp(actual.data, name.data(), name.size()) == 0) {
            symbol = static_cast<loom_symbol_id_t>(i);
            break;
          }
        }
        IREE_RETURN_IF_ERROR(Require(symbol != LOOM_SYMBOL_ID_INVALID,
                                     "configuration symbol not found"));
        const auto *op = module->symbols.entries[symbol].defining_op;
        IREE_RETURN_IF_ERROR(
            Require(op && (loom_config_def_isa(op) || loom_config_decl_isa(op)),
                    "symbol is not a configuration"));
        auto type = loom_module_value_type(module, loom_op_results(op)[0]);
        loom_op_t *get = nullptr;
        IREE_RETURN_IF_ERROR(loom_config_get_build(
            builder, {0, symbol}, type, LOOM_LOCATION_UNKNOWN, &get));
        *id = loom_config_get_result(get);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Cast(Value input, loom_scalar_type_t result_type,
                                   Value *out) {
  if (result_type != LOOM_SCALAR_TYPE_INDEX &&
      result_type != LOOM_SCALAR_TYPE_OFFSET &&
      result_type != LOOM_SCALAR_TYPE_I1 &&
      result_type != LOOM_SCALAR_TYPE_I8 &&
      result_type != LOOM_SCALAR_TYPE_I16 &&
      result_type != LOOM_SCALAR_TYPE_I32 &&
      result_type != LOOM_SCALAR_TYPE_I64) {
    if (out)
      *out = {};
    return Invalid("invalid cast result scalar type");
  }
  auto status = Preflight(
      program_, {input},
      [=](const auto *module, const auto &ids) {
        auto input_type = loom_module_value_type(module, ids[0]);
        return Require(
            IntegerOrAddress(input_type) &&
                (Address(input_type) || Address(loom_type_scalar(result_type))),
            "index.cast requires integer/address input and an address-domain "
            "boundary");
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {input},
      [&](auto *module, auto *builder, const auto &ids, auto *result) {
        auto input_type = loom_module_value_type(module, ids[0]);
        auto output_type = loom_type_scalar(result_type);
        IREE_RETURN_IF_ERROR(
            Require(IntegerOrAddress(input_type) &&
                        (Address(input_type) || Address(output_type)),
                    "index.cast requires an address-domain boundary"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, ids[0], input_type,
                                                   output_type,
                                                   LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_cast_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Add(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, AddressPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[0]);
        IREE_RETURN_IF_ERROR(Require(
            Address(type) &&
                loom_type_equal(type, loom_module_value_type(module, ids[1])),
            "index.add requires matching index or offset operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_index_add_build(builder, ids[0], ids[1], type,
                                                  LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_add_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Multiply(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, IndexPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        IREE_RETURN_IF_ERROR(
            Require(ScalarIs(loom_module_value_type(module, ids[0]),
                             LOOM_SCALAR_TYPE_INDEX) &&
                        ScalarIs(loom_module_value_type(module, ids[1]),
                                 LOOM_SCALAR_TYPE_INDEX),
                    "index.mul requires index operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_index_mul_build(builder, ids[0], ids[1],
                                                  LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_mul_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Scale(Value index, Value byte_stride,
                                    Value *out) {
  auto status = Preflight(
      program_, {index, byte_stride},
      [](const auto *module, const auto &ids) {
        return Require(ScalarIs(loom_module_value_type(module, ids[0]),
                                LOOM_SCALAR_TYPE_INDEX) &&
                           ScalarIs(loom_module_value_type(module, ids[1]),
                                    LOOM_SCALAR_TYPE_OFFSET),
                       "index.scale requires index and offset operands");
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {index, byte_stride},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        IREE_RETURN_IF_ERROR(
            Require(ScalarIs(loom_module_value_type(module, ids[0]),
                             LOOM_SCALAR_TYPE_INDEX) &&
                        ScalarIs(loom_module_value_type(module, ids[1]),
                                 LOOM_SCALAR_TYPE_OFFSET),
                    "index.scale requires index and offset operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_index_scale_build(
            builder, ids[0], ids[1], loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
            LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_scale_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Equal(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, AddressPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[0]);
        IREE_RETURN_IF_ERROR(Require(
            Address(type) &&
                loom_type_equal(type, loom_module_value_type(module, ids[1])),
            "index.cmp requires matching index or offset operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(
            loom_index_cmp_build(builder, LOOM_INDEX_CMP_PREDICATE_EQ, ids[0],
                                 ids[1], LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_cmp_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Divide(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, IndexPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        IREE_RETURN_IF_ERROR(
            Require(ScalarIs(loom_module_value_type(module, ids[0]),
                             LOOM_SCALAR_TYPE_INDEX) &&
                        ScalarIs(loom_module_value_type(module, ids[1]),
                                 LOOM_SCALAR_TYPE_INDEX),
                    "index.div requires index operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_index_div_build(builder, ids[0], ids[1],
                                                  LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_index_div_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::ShiftLeft(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, WordPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[0]);
        IREE_RETURN_IF_ERROR(Require(
            (ScalarIs(type, LOOM_SCALAR_TYPE_I32) ||
             ScalarIs(type, LOOM_SCALAR_TYPE_I64)) &&
                loom_type_equal(type, loom_module_value_type(module, ids[1])),
            "scalar.shli requires matching i32 or i64 operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
            builder, 0, ids[0], ids[1], type, LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_scalar_shli_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::BitAnd(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, WordPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[0]);
        IREE_RETURN_IF_ERROR(Require(
            (ScalarIs(type, LOOM_SCALAR_TYPE_I32) ||
             ScalarIs(type, LOOM_SCALAR_TYPE_I64)) &&
                loom_type_equal(type, loom_module_value_type(module, ids[1])),
            "scalar.andi requires matching i32 or i64 operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
            builder, ids[0], ids[1], type, LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_scalar_andi_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::NotEqual(Value lhs, Value rhs, Value *out) {
  auto status = Preflight(program_, {lhs, rhs}, WordPair, out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {lhs, rhs},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[0]);
        IREE_RETURN_IF_ERROR(Require(
            (ScalarIs(type, LOOM_SCALAR_TYPE_I32) ||
             ScalarIs(type, LOOM_SCALAR_TYPE_I64)) &&
                loom_type_equal(type, loom_module_value_type(module, ids[1])),
            "scalar.cmpi requires matching i32 or i64 operands"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(
            loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_NE,
                                   ids[0], ids[1], LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_scalar_cmpi_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Select(Value condition, Value if_true,
                                     Value if_false, Value *out) {
  auto status = Preflight(
      program_, {condition, if_true, if_false},
      [](const auto *module, const auto &ids) {
        return Require(
            ScalarIs(loom_module_value_type(module, ids[0]),
                     LOOM_SCALAR_TYPE_I1) &&
                loom_type_equal(loom_module_value_type(module, ids[1]),
                                loom_module_value_type(module, ids[2])),
            "scf.select requires i1 condition and matching value types");
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {condition, if_true, if_false},
      [](auto *module, auto *builder, const auto &ids, auto *result) {
        auto type = loom_module_value_type(module, ids[1]);
        IREE_RETURN_IF_ERROR(Require(
            ScalarIs(loom_module_value_type(module, ids[0]),
                     LOOM_SCALAR_TYPE_I1) &&
                loom_type_equal(type, loom_module_value_type(module, ids[2])),
            "scf.select requires i1 condition and matching value types"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_scf_select_build(
            builder, ids[0], ids[1], ids[2], type, LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_scf_select_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Allocate(Value byte_length, int64_t alignment,
                                       Value *out) {
  if (alignment <= 0 || (alignment & (alignment - 1))) {
    if (out)
      *out = {};
    return Invalid("scratch alignment must be a positive power of two");
  }
  auto status = Preflight(
      program_, {byte_length},
      [](const auto *module, const auto &ids) {
        return Require(ScalarIs(loom_module_value_type(module, ids[0]),
                                LOOM_SCALAR_TYPE_OFFSET),
                       "scratch byte length must have offset type");
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, {byte_length},
      [=](auto *module, auto *builder, const auto &ids, auto *result) {
        IREE_RETURN_IF_ERROR(
            Require(ScalarIs(loom_module_value_type(module, ids[0]),
                             LOOM_SCALAR_TYPE_OFFSET),
                    "scratch byte length must have offset type"));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
            builder, 0, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
            LOOM_VALUE_ID_INVALID, alignment, ids[0], loom_type_buffer(),
            LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_buffer_alloca_result(op);
        return iree_ok_status();
      },
      out);
}

loomc_status_t CommandValues::Parameter(Value source,
                                        std::string_view key_pattern,
                                        const std::vector<Value> &substitutions,
                                        loom_scalar_type_t element_type,
                                        const std::vector<Value> &dimensions,
                                        Value *out) {
  size_t placeholders = 0;
  bool valid =
      !key_pattern.empty() && key_pattern.find('\0') == std::string_view::npos;
  for (size_t i = 0; i < key_pattern.size() && valid; ++i) {
    if (key_pattern[i] == '{') {
      valid = i + 1 < key_pattern.size() && key_pattern[++i] == '}';
      ++placeholders;
    } else if (key_pattern[i] == '}') {
      valid = false;
    }
  }
  if (!valid || placeholders != substitutions.size() || dimensions.empty() ||
      dimensions.size() > 2 || !loom_scalar_type_is_valid(element_type) ||
      element_type == LOOM_SCALAR_TYPE_INDEX ||
      element_type == LOOM_SCALAR_TYPE_OFFSET) {
    if (out)
      *out = {};
    return Invalid(
        "parameter needs a matching key pattern and rank-1/2 storage view");
  }
  std::vector<Value> inputs{source};
  inputs.insert(inputs.end(), substitutions.begin(), substitutions.end());
  inputs.insert(inputs.end(), dimensions.begin(), dimensions.end());
  auto status = Preflight(
      program_, inputs,
      [](const auto *module, const auto &ids) {
        IREE_RETURN_IF_ERROR(
            Require(loom_type_is_buffer(loom_module_value_type(module, ids[0])),
                    "parameter source must be buffer"));
        for (size_t i = 1; i < ids.size(); ++i) {
          IREE_RETURN_IF_ERROR(
              Require(ScalarIs(loom_module_value_type(module, ids[i]),
                               LOOM_SCALAR_TYPE_INDEX),
                      "parameter substitutions/extents must be index"));
        }
        return iree_ok_status();
      },
      out);
  if (!loomc_status_is_ok(status))
    return status;
  return EmitValue(
      program_, inputs,
      [&](auto *module, auto *builder, const auto &ids, auto *result) {
        IREE_RETURN_IF_ERROR(
            Require(loom_type_is_buffer(loom_module_value_type(module, ids[0])),
                    "parameter source must be buffer"));
        for (size_t i = 1; i < ids.size(); ++i) {
          IREE_RETURN_IF_ERROR(
              Require(ScalarIs(loom_module_value_type(module, ids[i]),
                               LOOM_SCALAR_TYPE_INDEX),
                      "parameter substitutions/extents must be index"));
        }
        const size_t dim_start = 1 + substitutions.size();
        auto type =
            dimensions.size() == 1
                ? loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
                                      loom_dim_pack_dynamic(ids[dim_start]), 0)
                : loom_type_shaped_2d(LOOM_TYPE_VIEW, element_type,
                                      loom_dim_pack_dynamic(ids[dim_start]),
                                      loom_dim_pack_dynamic(ids[dim_start + 1]),
                                      0);
        loom_string_id_t key = LOOM_STRING_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_module_intern_string(
            module,
            iree_make_string_view(key_pattern.data(), key_pattern.size()),
            &key));
        loom_op_t *op = nullptr;
        IREE_RETURN_IF_ERROR(loom_command_parameter_build(
            builder, ids[0], key, ids.data() + 1, substitutions.size(), type,
            LOOM_LOCATION_UNKNOWN, &op));
        *result = loom_command_parameter_result(op);
        return iree_ok_status();
      },
      out);
}
} // namespace hrx::loomify_gguf
