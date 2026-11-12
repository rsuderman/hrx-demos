// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Shared exact-pin raw-buffer ABI assembly; no numerical definitions.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "loomcxx-pipeline/support/mapping.h"
#include <vector>

#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/template/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"

namespace {
void Require(loomc_status_t status) {
  if (!loomc_status_is_ok(status)) {
    auto text = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(text.size), text.data);
    loomc_status_free(status);
    std::exit(1);
  }
  loomc_status_free(status);
}
void RequireResult(loomc_result_t *result) {
  if (!result || !loomc_result_succeeded(result)) {
    if (result) {
      for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        auto *d = loomc_result_diagnostic_at(result, i);
        std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
      }
    }
    std::exit(1);
  }
  loomc_result_release(result);
}
loom_symbol_ref_t Symbol(loom_module_t *module, const char *name) {
  loom_string_id_t string;
  Require(
      loom_module_intern_string(module, iree_make_cstring_view(name), &string));
  auto id = loom_module_find_symbol(module, string);
  if (id == LOOM_SYMBOL_ID_INVALID)
    Require(loom_module_add_symbol(module, string, &id));
  return {0, id};
}

// A raw buffer represents the zero-origin root, unlike a general C++ pointer.
// The bridge does not weaken bounds, aliasing or arithmetic contracts; it only
// supplies explicit zero-origin pointer components when requested by mapping.
void Bridge(loom_module_t *module, const char *family_name,
            const char *provider_name, const char *cxx_family_name,
            std::vector<loom_type_t> types, bool pointer_origins) {
  auto family = Symbol(module, family_name);
  auto provider = Symbol(module, provider_name);
  auto cxx_family = Symbol(module, cxx_family_name);
  loom_builder_t top;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &top);
  if (!module->symbols.entries[family.symbol_id].defining_op) {
    loom_op_t *declaration = nullptr;
    Require(loom_template_decl_build(
        &top, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), {}, family,
        types.data(), types.size(), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &declaration));
  }
  loom_op_t *implementation = nullptr;
  Require(loom_template_def_build(
      &top, LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_CC, family, 0, 0,
      LOOM_TEMPLATE_CC_DEVICE, 0, 0, loom_symbol_ref_null(), {}, 0, provider,
      types.data(), types.size(), nullptr, 0, nullptr, 0, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &implementation));
  loom_builder_t body = top;
  auto *region = loom_template_def_body(implementation);
  loom_builder_enter_region(&body, implementation, region);
  auto *block = loom_region_entry_block(region);
  loom_op_t *zero = nullptr;
  Require(loom_index_constant_build(&body, loom_attr_i64(0),
                                    loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                    LOOM_LOCATION_UNKNOWN, &zero));
  std::vector<loom_value_id_t> args;
  for (size_t i = 0; i < types.size(); ++i) {
    args.push_back(block->arg_ids[i]);
    if (pointer_origins && loom_type_kind(types[i]) == LOOM_TYPE_BUFFER)
      args.push_back(loom_index_constant_result(zero));
  }
  loom_op_t *apply = nullptr;
  Require(loom_template_apply_build(&body, 0, cxx_family, args.data(),
                                    args.size(), 0, 0, nullptr, 0, nullptr, 0,
                                    LOOM_LOCATION_UNKNOWN, &apply));
  loom_op_t *ret = nullptr;
  Require(loom_template_return_build(&body, nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                     &ret));
}
loom_type_t Type(const std::string &name) {
  if (name == "buffer")
    return loom_type_buffer();
  if (name == "index")
    return loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (name == "offset")
    return loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  if (name == "i1")
    return loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  return loom_type_scalar(
      LOOM_SCALAR_TYPE_F32); // Parser admits only these types.
}
loom_op_t *Find(loom_module_t *module, const std::string &name) {
  loom_string_id_t string;
  Require(loom_module_intern_string(
      module, iree_make_cstring_view(name.c_str()), &string));
  auto id = loom_module_find_symbol(module, string);
  return id == LOOM_SYMBOL_ID_INVALID ? nullptr
                                      : module->symbols.entries[id].defining_op;
}
bool Matches(loom_module_t *module, loom_op_t *op,
             const std::vector<loom_type_t> &types) {
  if (!op || !loom_template_decl_isa(op) ||
      loom_template_decl_results(op).count != 0)
    return false;
  auto args = loom_template_decl_args(op);
  if (args.count != types.size())
    return false;
  for (size_t i = 0; i < types.size(); ++i)
    if (!loom_type_equal(loom_module_value_type(module, args.values[i]),
                         types[i]))
      return false;
  return true;
}
bool Validate(loom_module_t *module, const loomcxx_pipeline::Mapping &m) {
  std::vector<loom_type_t> public_types, imported_types;
  for (const auto &name : m.types) {
    public_types.push_back(Type(name));
    imported_types.push_back(Type(name));
    if (m.pointer_origins && name == "buffer")
      imported_types.push_back(Type("offset"));
  }
  auto *imported = Find(module, m.imported_family);
  auto *family = Find(module, m.family);
  if (!Matches(module, imported, imported_types) ||
      loom_template_decl_cc(imported) != LOOM_TEMPLATE_CC_DEVICE ||
      (family && !Matches(module, family, public_types)) ||
      Find(module, m.provider)) {
    std::fprintf(stderr,
                 "ABI mapping for %s: imported/public signature, device "
                 "convention or provider identity mismatch\n",
                 m.family.c_str());
    return false;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr,
                 "usage: assemble INPUT.loombc OUTPUT.loombc [MAPPINGS.abi]\n");
    return 1;
  }
  std::vector<loomcxx_pipeline::Mapping> mappings;
  if (argc == 4) {
    std::ifstream input(argv[3]);
    std::string error;
    if (!loomcxx_pipeline::ReadMappings(input, &mappings, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  }
  auto allocator = loomc_allocator_system();
  loomc_target_environment_t *target_environment = nullptr;
  Require(
      loomc_target_environment_create_amdgpu(allocator, &target_environment));
  loomc_context_target_options_t target_options = {};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.target_environment = target_environment;
  loomc_context_options_t context_options = {};
  context_options.next = &target_options;
  loomc_context_t *context = nullptr;
  Require(loomc_context_create(&context_options, allocator, &context));
  loomc_workspace_t *workspace = nullptr;
  Require(loomc_workspace_create(nullptr, allocator, &workspace));
  loomc_module_t *module = nullptr;
  loomc_result_t *result = nullptr;
  Require(loomc_module_deserialize_from_path(
      context, workspace, loomc_make_cstring_view(argv[1]), nullptr, allocator,
      &module, &result));
  RequireResult(result);
  auto *native = loomc_module_get_mutable_interop_view(module).module;
  // The original entries explicitly export their symbol spelling. The C++
  // symbol annotation supplies the callee; retain the export metadata too.
  for (auto *op = loom_module_block(native)->first_op; op; op = op->next_op) {
    if (loom_kernel_def_isa(op)) {
      auto symbol = loom_kernel_def_callee(op).symbol_id;
      loom_op_attrs(op)[3] =
          loom_attr_string(native->symbols.entries[symbol].name_id);
    }
  }
  // Admit the complete mapping set before mutating the module.
  for (const auto &mapping : mappings)
    if (!Validate(native, mapping))
      return 1;
  for (const auto &mapping : mappings) {
    std::vector<loom_type_t> types;
    for (const auto &name : mapping.types)
      types.push_back(Type(name));
    Bridge(native, mapping.family.c_str(), mapping.provider.c_str(),
           mapping.imported_family.c_str(), types, mapping.pointer_origins);
  }
  Require(loomc_module_verify(module, nullptr, allocator, &result));
  RequireResult(result);
  loomc_module_serialize_options_t options = {};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  Require(loomc_module_serialize_to_path(
      module, &options, loomc_make_cstring_view(argv[2]), allocator));
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
}
