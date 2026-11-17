// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <map>
#include <set>
#include <sstream>

#include "lib/loomify/benchmark/reset_bytes_source.h"

#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/interop.h"
#include "loomc/link.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
using Module = Owner<loomc_module_t, loomc_module_release>;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool Status(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok && error) {
    const auto text = loomc_status_message(status);
    error->assign(text.data ? text.data : "", text.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(const loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  std::string message = "native standalone benchmark composition failed";
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto *diagnostic = loomc_result_diagnostic_at(result, i);
    message.append("\n").append(diagnostic->message.data,
                                diagnostic->message.size);
  }
  return Fail(error, message);
}
bool SymbolSpelling(const std::string &name) {
  if (name.empty())
    return false;
  return std::all_of(name.begin(), name.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '.' || c == '$' || c == '-';
  });
}

bool ValidatePrepared(const Invocation &invocation,
                      const PreparedCase &prepared, std::string *error) {
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return Fail(error,
                "source replay has no declared indirect-count realization");
  if (prepared.setup_recipe_id.empty() ||
      prepared.arithmetic_check != "unchecked" ||
      invocation.arguments.size() != invocation.command.argument_count ||
      invocation.arguments.size() != prepared.source_argument_types.size())
    return Fail(
        error,
        "prepared case requires exact ABI and honest unchecked arithmetic");
  size_t buffer_ordinal = 0;
  for (size_t i = 0; i < invocation.arguments.size(); ++i) {
    const auto &argument = invocation.arguments[i];
    const auto &type = prepared.source_argument_types[i];
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
      if (type != "buffer" || buffer_ordinal >= prepared.buffer_extents.size())
        return Fail(error, "prepared case is missing a source buffer extent");
      const auto &extent = prepared.buffer_extents[buffer_ordinal++];
      const auto &ref = argument.value.buffer;
      if (extent.binding.role != ref.role ||
          extent.binding.root_index != ref.root_index ||
          extent.binding.byte_offset != ref.byte_offset ||
          !extent.byte_length ||
          (ref.byte_length != UINT64_MAX &&
           extent.byte_length > ref.byte_length))
        return Fail(error,
                    "prepared buffer view changes the selected native binding");
      const auto *root = FindInputRoot(prepared.roots, ref);
      if (!root || ref.byte_offset < root->original_byte_offset ||
          ref.byte_offset - root->original_byte_offset > root->bytes.size() ||
          extent.byte_length >
              root->bytes.size() -
                  (ref.byte_offset - root->original_byte_offset))
        return Fail(error, "prepared buffer view exceeds its root");
    } else if (!((argument.kind == LOOMC_CMD_ARGUMENT_KIND_INDEX &&
                  type == "index") ||
                 (argument.kind == LOOMC_CMD_ARGUMENT_KIND_OFFSET &&
                  type == "offset") ||
                 (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B32 &&
                  (type == "f32" || type == "i32")) ||
                 (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B8 &&
                  type == "i1"))) {
      return Fail(error, "source scalar ABI has no declared emission recipe");
    }
  }
  for (size_t i = 0; i < invocation.arguments.size(); ++i) {
    const auto &argument = invocation.arguments[i];
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B8 &&
        argument.value.scalar_bits > 1)
      return Fail(error, "i1 scalar carrier is not exactly zero or one");
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B32 &&
        argument.value.scalar_bits > UINT32_MAX)
      return Fail(error, "B32 scalar carrier has nonzero upper bits");
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B32 &&
        prepared.source_argument_types[i] == "f32") {
      const uint32_t bits = static_cast<uint32_t>(argument.value.scalar_bits);
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      if (!std::isfinite(value))
        return Fail(error,
                    "nonfinite f32 has no exact direct-case literal recipe");
    }
  }
  if (buffer_ordinal != prepared.buffer_extents.size())
    return Fail(error, "prepared case has extra native buffer extents");
  if (prepared.resets.empty()) {
    if (prepared.measurement_subject != "kernel" ||
        prepared.reset_policy != "none_overwrite_only")
      return Fail(
          error, "nonreset case must declare its actual single-kernel subject");
  } else if (prepared.measurement_subject != "reset_plus_kernel" ||
             prepared.reset_policy != "copy_seed_each_invocation") {
    return Fail(error, "mutable replay must declare the timed reset composite");
  }
  for (const auto &reset : prepared.resets) {
    const auto *root = FindInputRoot(prepared.roots, reset.target);
    if (!root || reset.target.role != LOOMC_CMD_BUFFER_ROLE_REBINDABLE ||
        !reset.target.byte_length || reset.target.byte_length > INT32_MAX ||
        reset.seed.bytes.size() != reset.target.byte_length ||
        reset.seed.original_byte_offset != 0 ||
        reset.target.byte_offset < root->original_byte_offset ||
        reset.target.byte_offset - root->original_byte_offset >
            root->bytes.size() ||
        reset.target.byte_length >
            root->bytes.size() -
                (reset.target.byte_offset - root->original_byte_offset))
      return Fail(error,
                  "reset seed/live span violates the exact-copy contract");
  }
  return true;
}

bool MatchesType(loom_type_t type, const std::string &spelling) {
  if (spelling == "buffer")
    return loom_type_is_buffer(type);
  if (spelling == "index")
    return loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  if (spelling == "offset")
    return loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET));
  if (spelling == "i32")
    return loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  if (spelling == "i1")
    return loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  return spelling == "f32" &&
         loom_type_equal(type, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
}

// The recipe supplies only its known operation-level logical geometry. Native
// replay must prove exact selected ABI/counts/implementation before
// publication.
struct LoadedImplementation {
  Module module{nullptr, loomc_module_release};
  std::vector<std::string> root_names;
};

bool LoadImplementation(const Implementation &implementation,
                        loomc_context_t *context, loomc_workspace_t *workspace,
                        LoadedImplementation *out, std::string *error) {
  if (!implementation.request)
    return Fail(error, "selected kernel request is unavailable");
  std::vector<loomc_string_view_t> names(
      loomc_request_root_count(implementation.request.get()));
  loomc_module_t *raw_module = nullptr;
  loomc_result_t *raw_result = nullptr;
  const auto status = loomc_module_deserialize_request(
      context, workspace, implementation.request.get(),
      loomc_allocator_system(), &raw_module, names.data(), &raw_result);
  Module module(raw_module, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result,
                                                     loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error))
    return false;
  LoadedImplementation loaded;
  loaded.module = std::move(module);
  for (const auto name : names)
    loaded.root_names.emplace_back(name.data, name.size);
  *out = std::move(loaded);
  return true;
}

bool CheckSignature(loomc_module_t *module, const std::string &name,
                    const PreparedCase &prepared, std::string *error) {
  if (!SymbolSpelling(name))
    return Fail(error, "selected kernel root needs unsupported symbol quoting");
  loomc_module_interop_view_t native{};
  loomc_result_t *raw_result = nullptr;
  const auto projected = loomc_module_get_interop_view(
      module, loomc_allocator_system(), &native, &raw_result);
  Owner<loomc_result_t, loomc_result_release> projection(raw_result,
                                                         loomc_result_release);
  if (!Status(projected, error) || !Result(projection.get(), error))
    return false;
  if (!native.module)
    return Fail(error, "selected kernel request native module is unavailable");
  const loom_symbol_t *root = nullptr;
  for (size_t i = 0; i < native.module->symbols.count; ++i) {
    auto &symbol = native.module->symbols.entries[i];
    const auto text =
        loom_string_table_get(&native.module->strings, symbol.name_id);
    if (name == std::string(text.data, text.size)) {
      root = &symbol;
      break;
    }
  }
  if (!root || !root->defining_op || !loom_kernel_def_isa(root->defining_op))
    return Fail(error,
                "selected source root is not a source kernel definition");
  const auto *config = loom_kernel_def_config(root->defining_op);
  const auto *body = loom_kernel_def_body(root->defining_op);
  if (!config || config->block_count != 1 ||
      config->blocks[0]->arg_count != prepared.source_workloads.size() ||
      !body || body->block_count != 1 ||
      body->blocks[0]->arg_count != prepared.source_argument_types.size())
    return Fail(
        error,
        "selected source root does not match recipe workload/argument arities");
  for (size_t i = 0; i < prepared.source_workloads.size(); ++i)
    if (!MatchesType(loom_module_value_type(native.module,
                                            config->blocks[0]->arg_ids[i]),
                     "index"))
      return Fail(error, "selected source workload ABI is not index");
  const auto *block = body->blocks[0];
  for (size_t i = 0; i < prepared.source_argument_types.size(); ++i)
    if (!MatchesType(loom_module_value_type(native.module, block->arg_ids[i]),
                     prepared.source_argument_types[i]))
      return Fail(error,
                  "selected source runtime ABI differs from operation recipe");
  // Native kernel functions are exported symbol routes; helpers retain their
  // native visibility. Keep the selected request module unchanged.
  return true;
}

std::string KernelDeclaration(const PreparedCase &prepared,
                              const std::string &kernel) {
  std::ostringstream source;
  source << "kernel.decl @" << kernel << '(';
  for (size_t i = 0; i < prepared.source_workloads.size(); ++i) {
    if (i)
      source << ", ";
    source << "%workload" << i << ": index";
  }
  source << ") launch(";
  for (size_t i = 0; i < prepared.source_argument_types.size(); ++i) {
    if (i)
      source << ", ";
    source << "%value" << i << ": " << prepared.source_argument_types[i];
  }
  source << ")\n";
  return source.str();
}

std::string ScalarSetup(const PreparedCase &prepared,
                        const Invocation &invocation, bool case_literals) {
  std::ostringstream source;
  auto literal = [&](const std::string &name, const std::string &value,
                     const std::string &type) {
    source << "  %" << name << " = ";
    if (case_literals)
      source << "check.literal value(" << value << ")";
    else
      source << (type == "i32" || type == "i1" ? "scalar.constant "
                                               : "index.constant ")
             << value;
    source << " : " << type << "\n";
  };
  for (size_t i = 0; i < prepared.source_workloads.size(); ++i)
    literal("workload" + std::to_string(i),
            std::to_string(prepared.source_workloads[i]), "index");
  for (size_t i = 0; i < invocation.arguments.size(); ++i) {
    const auto &argument = invocation.arguments[i];
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_INDEX ||
        argument.kind == LOOMC_CMD_ARGUMENT_KIND_OFFSET) {
      literal("value" + std::to_string(i),
              std::to_string(static_cast<int64_t>(argument.value.scalar_bits)),
              prepared.source_argument_types[i]);
    } else if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B8) {
      literal("value" + std::to_string(i),
              argument.value.scalar_bits ? "true" : "false", "i1");
    } else if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B32 &&
               prepared.source_argument_types[i] == "i32") {
      const uint32_t bits = static_cast<uint32_t>(argument.value.scalar_bits);
      int32_t value;
      std::memcpy(&value, &bits, sizeof(value));
      literal("value" + std::to_string(i), std::to_string(value), "i32");
    } else if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_B32) {
      if (case_literals) {
        const uint32_t bits = static_cast<uint32_t>(argument.value.scalar_bits);
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        // Every finite f32 is exactly representable in double. Emit enough
        // digits to round-trip that double, retaining signed zero as well.
        std::ostringstream spelling;
        spelling.imbue(std::locale::classic());
        spelling << std::scientific
                 << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << static_cast<double>(value);
        literal("value" + std::to_string(i), spelling.str(), "f32");
      } else {
        source << "  %bits" << i << " = scalar.constant "
               << static_cast<int32_t>(argument.value.scalar_bits) << " : i32\n"
               << "  %value" << i << " = scalar.bitcast %bits" << i
               << " : i32 to f32\n";
      }
    }
  }
  if (!prepared.resets.empty()) {
    literal("reset_zero", "0", "offset");
    for (size_t i = 0; i < prepared.resets.size(); ++i)
      literal("reset_bytes" + std::to_string(i),
              std::to_string(prepared.resets[i].target.byte_length), "index");
  }
  return source.str();
}

// The proof-only command and actual check.case use the same launch generator.
// Tensor annotations are required by the direct check runner; the selected
// kernel ABI remains buffer. No runtime scalar is substituted or rounded.
std::string LaunchSequence(const PreparedCase &prepared,
                           const Invocation &invocation,
                           const std::string &kernel, bool tensor_arguments) {
  std::ostringstream source;
  for (size_t i = 0; i < prepared.resets.size(); ++i) {
    source << "  kernel.launch @loomify.benchmark.reset_bytes[%reset_bytes" << i
           << "](%reset_bytes" << i << ", %reset_zero, %reset_zero, %seed" << i
           << ", %reset_live" << i << ") : [index](index, offset, offset, ";
    const auto length = prepared.resets[i].target.byte_length;
    if (tensor_arguments)
      source << "tensor<" << length << "xi8>, tensor<" << length << "xi8>";
    else
      source << "buffer, buffer";
    source << ")\n";
  }
  source << "  kernel.launch @" << kernel << '[';
  for (size_t i = 0; i < prepared.source_workloads.size(); ++i) {
    if (i)
      source << ", ";
    source << "%workload" << i;
  }
  source << "](";
  size_t buffer_ordinal = 0;
  for (size_t i = 0; i < invocation.arguments.size(); ++i) {
    if (i)
      source << ", ";
    if (invocation.arguments[i].kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      source << "%arg" << buffer_ordinal++;
    else
      source << "%value" << i;
  }
  source << ") : [";
  for (size_t i = 0; i < prepared.source_workloads.size(); ++i) {
    if (i)
      source << ", ";
    source << "index";
  }
  source << "](";
  buffer_ordinal = 0;
  for (size_t i = 0; i < prepared.source_argument_types.size(); ++i) {
    if (i)
      source << ", ";
    const auto &type = prepared.source_argument_types[i];
    if (tensor_arguments && type == "buffer")
      source << "tensor<"
             << prepared.buffer_extents[buffer_ordinal++].byte_length << "xi8>";
    else
      source << type;
  }
  source << ")\n";
  return source.str();
}

std::string ProofSource(const PreparedCase &prepared,
                        const Invocation &invocation, const std::string &kernel,
                        const std::string &subject) {
  std::ostringstream source;
  source << KernelDeclaration(prepared, kernel) << "command.def public @"
         << subject << "() launch(";
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i) {
    if (i)
      source << ", ";
    source << "%arg" << i << ": buffer";
  }
  for (size_t i = 0; i < prepared.resets.size(); ++i)
    source << ", %seed" << i << ": buffer, %reset_live" << i << ": buffer";
  source << ") {\n" << ScalarSetup(prepared, invocation, false);
  if (!prepared.resets.empty())
    source << " command.serial {\n";
  source << LaunchSequence(prepared, invocation, kernel, false);
  if (!prepared.resets.empty())
    source << " }\n";
  source << " command.return\n}\n";
  return source.str();
}

std::string CaseSource(const PreparedCase &prepared,
                       const Invocation &invocation, const std::string &kernel,
                       const std::string &subject,
                       const std::string &benchmark_symbol,
                       bool include_declaration) {
  std::ostringstream source, checks;
  if (include_declaration)
    source << KernelDeclaration(prepared, kernel);
  source << "check.case @" << subject << " {\n";
  const auto tensor = [](uint64_t length) {
    return "tensor<" + std::to_string(length) + "xi8>";
  };
  auto read = [&](const std::string &name, const PreparedRoot &root) {
    source << "  %" << name << "_storage = check.file.read.npy path(\""
           << root.fixture_path
           << "\") : " << tensor(root.bytes.size() + root.fixture_suffix_bytes)
           << "\n"
           << "  %" << name << " = check.tensor.view %" << name
           << "_storage offset(0) : "
           << tensor(root.bytes.size() + root.fixture_suffix_bytes) << " -> "
           << tensor(root.bytes.size()) << "\n";
  };
  auto view = [&](const std::string &name, const std::string &root,
                  uint64_t total, uint64_t offset, uint64_t length) {
    source << "  %" << name << " = check.tensor.view %" << root << " offset("
           << offset << ") : " << tensor(total) << " -> " << tensor(length)
           << "\n";
  };
  auto expect = [&](const std::string &actual, const std::string &expected,
                    uint64_t length) {
    checks << "  check.expect.bitwise actual(%" << actual << ") expected(%"
           << expected << ") : " << tensor(length) << "\n";
  };
  source << "  %expected_tail = check.generate.fill value("
         << unsigned(kBenchmarkTailGuardValue)
         << ") : " << tensor(kBenchmarkTailGuardBytes) << "\n";
  auto guarded_read = [&](const std::string &name, const PreparedRoot &root) {
    read(name, root);
    view(name + "_tail", name + "_storage",
         root.bytes.size() + root.fixture_suffix_bytes, root.bytes.size(),
         kBenchmarkTailGuardBytes);
    expect(name + "_tail", "expected_tail", kBenchmarkTailGuardBytes);
  };
  for (size_t i = 0; i < prepared.roots.size(); ++i)
    guarded_read("root" + std::to_string(i), prepared.roots[i]);
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i) {
    const auto &extent = prepared.buffer_extents[i];
    const auto *root = FindInputRoot(prepared.roots, extent.binding);
    const auto ordinal = root - prepared.roots.data();
    view("arg" + std::to_string(i), "root" + std::to_string(ordinal),
         root->bytes.size(),
         extent.binding.byte_offset - root->original_byte_offset,
         extent.byte_length);
  }
  // Independent reads are intentional. Native check.file.read.npy is effectful
  // and materializes each source into independent storage, even for one path.
  std::set<size_t> expected_weight_roots;
  for (size_t i = 0; i < prepared.representative_weight_spans.size(); ++i) {
    const auto &span = prepared.representative_weight_spans[i];
    const auto *root = FindInputRoot(prepared.roots, span.binding);
    const size_t ordinal = root - prepared.roots.data();
    const std::string suffix = std::to_string(i);
    if (expected_weight_roots.insert(ordinal).second)
      read("weight_expected_root" + std::to_string(ordinal), *root);
    const uint64_t offset =
        span.binding.byte_offset - root->original_byte_offset;
    view("weight_actual" + suffix, "root" + std::to_string(ordinal),
         root->bytes.size(), offset, span.byte_length);
    view("weight_expected" + suffix,
         "weight_expected_root" + std::to_string(ordinal), root->bytes.size(),
         offset, span.byte_length);
    expect("weight_actual" + suffix, "weight_expected" + suffix,
           span.byte_length);
  }
  for (size_t i = 0; i < prepared.resets.size(); ++i) {
    const auto &reset = prepared.resets[i];
    const std::string suffix = std::to_string(i);
    const auto *root = FindInputRoot(prepared.roots, reset.target);
    const auto ordinal = root - prepared.roots.data();
    guarded_read("seed" + suffix, reset.seed);
    read("seed_expected" + suffix, reset.seed);
    expect("seed" + suffix, "seed_expected" + suffix, reset.seed.bytes.size());
    view("reset_live" + suffix, "root" + std::to_string(ordinal),
         root->bytes.size(),
         reset.target.byte_offset - root->original_byte_offset,
         reset.target.byte_length);
  }
  source << ScalarSetup(prepared, invocation, true)
         << LaunchSequence(prepared, invocation, kernel, true) << checks.str()
         << "  check.return\n}\n"
         << "check.benchmark<@" << subject << "> @" << benchmark_symbol << "\n";
  return source.str();
}

bool LinkCase(loomc_context_t *context, loomc_workspace_t *workspace,
              const std::string &source,
              const std::vector<loomc_module_t *> &implementations,
              const std::vector<std::string> &benchmark_symbols, Module *out,
              std::string *error) {
  loomc_source_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.identifier = loomc_make_cstring_view("benchmark-case.loom");
  options.contents = loomc_make_byte_span(source.data(), source.size());
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t *raw_source = nullptr;
  if (!Status(
          loomc_source_create(&options, loomc_allocator_system(), &raw_source),
          error))
    return false;
  Owner<loomc_source_t, loomc_source_release> owned_source(
      raw_source, loomc_source_release);
  loomc_module_t *raw_wrapper = nullptr;
  loomc_result_t *raw_result = nullptr;
  auto status = loomc_module_deserialize_from_source(
      context, workspace, owned_source.get(), nullptr, loomc_allocator_system(),
      &raw_wrapper, &raw_result);
  Module wrapper(raw_wrapper, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> parsed(raw_result,
                                                     loomc_result_release);
  if (!Status(status, error) || !Result(parsed.get(), error))
    return false;
  loomc_linker_t *raw_linker = nullptr;
  if (!Status(loomc_linker_create(context, nullptr, loomc_allocator_system(),
                                  &raw_linker),
              error))
    return false;
  Owner<loomc_linker_t, loomc_linker_release> linker(raw_linker,
                                                     loomc_linker_release);
  std::vector<std::string> labels;
  labels.reserve(implementations.size());
  std::vector<loomc_link_module_provider_t> providers{
      {wrapper.get(), loomc_make_cstring_view("benchmark-case"),
       LOOMC_LINK_PROVIDER_ROLE_INPUT}};
  for (size_t i = 0; i < implementations.size(); ++i) {
    labels.push_back("selected-kernel-request-" + std::to_string(i));
    providers.push_back({implementations[i],
                         loomc_make_cstring_view(labels.back().c_str()),
                         LOOMC_LINK_PROVIDER_ROLE_LIBRARY});
  }
  std::vector<loomc_string_view_t> roots;
  for (const auto &symbol : benchmark_symbols)
    roots.push_back(loomc_make_cstring_view(symbol.c_str()));
  loomc_link_options_t link{};
  link.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  link.structure_size = sizeof(link);
  link.mode = LOOMC_LINK_MODE_LINK;
  link.module_providers = providers.data();
  link.module_provider_count = providers.size();
  link.root_symbols = roots.data();
  link.root_symbol_count = roots.size();
  link.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  loomc_module_t *raw_linked = nullptr;
  raw_result = nullptr;
  status = loomc_link_module(linker.get(), workspace, &link, &raw_linked,
                             &raw_result);
  Module linked(raw_linked, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result,
                                                     loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error))
    return false;
  *out = std::move(linked);
  return true;
}

// Native LINK is the semantic boundary for retained implementation assembly.
// It preserves public definitions and predicates while legally renaming
// private dependencies and their references. Byte equality after that operation
// would reject legal links, so provenance is the immutable request + exact
// root, not the output spelling of native-private symbols.
bool MatchBufferSlot(const loomc_product_t *product, uint32_t slot,
                     const loomc_cmd_argument_t &argument) {
  loomc_cmd_launch_binding_t launch{};
  return loomc_cmd_program_product_launch_binding_at(product, 0, slot,
                                                     &launch) &&
         argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER &&
         argument.value.buffer.role == launch.role &&
         argument.value.buffer.root_index == launch.root_index &&
         argument.value.buffer.byte_offset == 0;
}

bool VerifyReplay(const Invocation &invocation, const PreparedCase &prepared,
                  const std::string &subject, const loomc_module_t *module,
                  loomc_workspace_t *workspace,
                  std::vector<EmittedDispatch> *out_dispatches,
                  std::string *error) {
  const auto root = loomc_make_cstring_view(subject.c_str());
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  // No request sink: this proof checks final launch ABI/counts only. It must
  // not reclassify the retained selected kernel with singleton runtime facts.
  loomc_product_t *raw_product = nullptr;
  loomc_result_t *raw_result = nullptr;
  const auto status = loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &raw_product,
      &raw_result);
  Owner<loomc_product_t, loomc_product_release> product(raw_product,
                                                        loomc_product_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result,
                                                     loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error))
    return false;
  loomc_cmd_program_layout_t layout{};
  if (!loomc_cmd_program_product_layout(product.get(), 0, &layout))
    return Fail(error, "cannot inspect standalone replay program");
  std::vector<EmittedDispatch> dispatches;
  bool ordered = false;
  for (uint32_t i = 0; i < layout.command_count; ++i) {
    loomc_cmd_command_t command{};
    if (!loomc_cmd_program_product_command_at(product.get(), 0, i, &command))
      return Fail(error, "cannot inspect standalone replay command");
    if (command.kind == LOOMC_CMD_COMMAND_KIND_BARRIER_EXECUTION) {
      ordered = true;
      continue;
    }
    const size_t ordinal = dispatches.size();
    if (command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
        ordinal > prepared.resets.size())
      return Fail(error,
                  "standalone replay changes the selected dispatch schedule");
    if (ordinal && !(ordered || command.begins_barrier_wave))
      return Fail(error, "reset composite lost its serial execution barrier");
    ordered = false;
    loomc_cmd_entry_requirement_t requirement{};
    if (!loomc_cmd_program_product_entry_requirement_at(
            product.get(), command.payload.dispatch.requirement_ordinal,
            &requirement))
      return Fail(error, "cannot inspect replay entry requirement");
    EmittedDispatch emitted;
    emitted.kernel_symbol.assign(requirement.symbol.data,
                                 requirement.symbol.size);
    emitted.reset = ordinal < prepared.resets.size();
    std::copy(std::begin(command.payload.dispatch.workgroup_count),
              std::end(command.payload.dispatch.workgroup_count),
              emitted.workgroup_count.begin());
    if (emitted.reset) {
      const auto length = prepared.resets[ordinal].target.byte_length;
      if (emitted.kernel_symbol != "loomify.benchmark.reset_bytes" ||
          command.argument_count != 5 ||
          emitted.workgroup_count[0] != (length + 255) / 256 ||
          emitted.workgroup_count[1] != 1 || emitted.workgroup_count[2] != 1)
        return Fail(error, "reset replay changed the exact-copy helper/grid");
      for (uint32_t a = 0; a < 5; ++a) {
        loomc_cmd_argument_t argument{};
        if (!loomc_cmd_program_product_argument_at(product.get(), 0, i, a,
                                                   &argument))
          return Fail(error, "cannot inspect reset replay ABI");
        if (a == 0) {
          if (argument.kind != LOOMC_CMD_ARGUMENT_KIND_INDEX ||
              argument.value.scalar_bits != length)
            return Fail(error, "reset replay changed exact byte length");
        } else if (a < 3) {
          if (argument.kind != LOOMC_CMD_ARGUMENT_KIND_OFFSET ||
              argument.value.scalar_bits)
            return Fail(error, "reset replay changed rebased copy origins");
        } else if (!MatchBufferSlot(product.get(),
                                    prepared.buffer_extents.size() +
                                        2 * ordinal + a - 3,
                                    argument)) {
          return Fail(error,
                      "reset replay changed immutable seed/live view bindings");
        }
      }
      dispatches.push_back(std::move(emitted));
      continue;
    }
    if (command.argument_count != invocation.arguments.size())
      return Fail(error, "standalone replay changes selected ABI arity");
    for (unsigned axis = 0; axis < 3; ++axis)
      if (command.payload.dispatch.workgroup_count[axis] !=
          invocation.command.payload.dispatch.workgroup_count[axis])
        return Fail(error,
                    "standalone replay changes selected workgroup counts");
    uint32_t buffer_ordinal = 0;
    for (uint32_t a = 0; a < command.argument_count; ++a) {
      loomc_cmd_argument_t argument{};
      if (!loomc_cmd_program_product_argument_at(product.get(), 0, i, a,
                                                 &argument) ||
          argument.kind != invocation.arguments[a].kind)
        return Fail(error,
                    "standalone replay changes the selected argument ABI");
      if (argument.kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
        if (argument.value.scalar_bits !=
            invocation.arguments[a].value.scalar_bits)
          return Fail(error, "standalone replay changes selected scalar bits");
      } else {
        // Each command argument receives its exact prepared view. The scenario
        // retains physical alias roots; this entry maps final ABI buffer slots.
        if (!MatchBufferSlot(product.get(), buffer_ordinal++, argument))
          return Fail(error,
                      "standalone replay changes its prepared buffer binding");
      }
    }
    dispatches.push_back(std::move(emitted));
  }
  if (dispatches.size() != prepared.resets.size() + 1)
    return Fail(error, "standalone replay has no selected dispatch");
  *out_dispatches = std::move(dispatches);
  return true;
}

bool WriteModule(const loomc_module_t *module,
                 const std::filesystem::path &path,
                 loomc_source_format_t format, std::string *error) {
  loomc_module_serialize_options_t options{};
  options.format = format;
  return Status(loomc_module_serialize_to_path(
                    module, &options, loomc_make_cstring_view(path.c_str()),
                    loomc_allocator_system()),
                error);
}
} // namespace

const model::TensorRecord *FindBenchmarkTensor(const Plan &plan,
                                               const std::string &key) {
  for (const auto &binding : plan.model.globals)
    if (binding.tensor.name == key)
      return &binding.tensor;
  for (const auto &layer : plan.model.layers)
    for (const auto &binding : layer.tensors)
      if (binding.tensor.name == key)
        return &binding.tensor;
  return nullptr;
}

const Parameter *FindBenchmarkParameter(const Plan &plan,
                                        const Invocation &invocation,
                                        const loomc_cmd_buffer_ref_t &binding,
                                        uint64_t byte_length,
                                        std::string *error) {
  if (binding.role != LOOMC_CMD_BUFFER_ROLE_FIXED || !byte_length ||
      invocation.program_ordinal >= plan.programs.size() ||
      (binding.byte_length != UINT64_MAX &&
       byte_length > binding.byte_length)) {
    Fail(error, "weight input is not a selected immutable parameter span");
    return nullptr;
  }
  const Parameter *found = nullptr;
  for (const auto &candidate :
       plan.programs[invocation.program_ordinal].parameters)
    if (candidate.fixed_buffer_index == binding.root_index &&
        binding.byte_offset >= candidate.byte_offset &&
        binding.byte_offset - candidate.byte_offset <= candidate.byte_length &&
        byte_length <= candidate.byte_length -
                           (binding.byte_offset - candidate.byte_offset)) {
      if (found) {
        Fail(error, "weight span has ambiguous immutable parameter placements");
        return nullptr;
      }
      found = &candidate;
    }
  if (!found)
    Fail(error, "weight input has no selected immutable parameter placement");
  else if (error)
    error->clear();
  return found;
}

bool EmitPreparedBenchmarkBundle(
    const Plan &plan, std::vector<PreparedInvocationCase> cases,
    const std::filesystem::path &empty_staging_directory,
    loomc_context_t *context, loomc_workspace_t *workspace,
    std::vector<EmittedCase> *out, std::string *error) {
  if (!context || !workspace || !out || empty_staging_directory.empty() ||
      cases.empty())
    return Fail(
        error,
        "benchmark bundle requires native context, cases and staging directory");
  std::error_code ec;
  if (!std::filesystem::is_directory(empty_staging_directory, ec) || ec ||
      !std::filesystem::is_empty(empty_staging_directory, ec) || ec)
    return Fail(error,
                "benchmark staging directory must already exist and be empty");
  struct Record {
    size_t case_index;
    CaseIdentity identity;
    std::string benchmark;
    std::string subject;
  };
  std::vector<Record> records;
  std::map<std::string, size_t> identities, implementation_indices;
  std::vector<LoadedImplementation> implementations;
  std::map<std::string, std::string> root_owners;
  std::set<std::string> declared_roots;
  std::string source;
  std::string helper_source;
  const bool has_resets =
      std::any_of(cases.begin(), cases.end(),
                  [](const auto &item) { return !item.inputs.resets.empty(); });
  if (has_resets) {
    const auto *helper = loomify_benchmark_reset_bytes_source_create();
    if (loomify_benchmark_reset_bytes_source_size() != 1 || !helper)
      return Fail(error, "benchmark reset helper source is unavailable");
    helper_source.assign(reinterpret_cast<const char *>(helper[0].data),
                         helper[0].size);
    source = helper_source + '\n';
  }
  for (size_t index = 0; index < cases.size(); ++index) {
    auto &item = cases[index];
    if (item.invocation_index >= plan.invocations.size())
      return Fail(error,
                  "prepared case names an unavailable selected invocation");
    const auto &invocation = plan.invocations[item.invocation_index];
    auto &prepared = item.inputs;
    if (invocation.implementation_index >= plan.implementations.size())
      return Fail(error, "selected invocation implementation is unavailable");
    if (!ValidatePrepared(invocation, prepared, error))
      return false;
    CaseIdentity identity;
    if (!BuildCaseIdentity(plan, invocation, prepared, &identity, error))
      return false;
    if (!identities.emplace(identity.case_id, records.size()).second)
      continue;
    if (!WriteInputFixtures(empty_staging_directory, &prepared.roots, error,
                            kBenchmarkTailGuardBytes))
      return false;
    std::vector<PreparedRoot> seeds;
    for (const auto &reset : prepared.resets)
      seeds.push_back(reset.seed);
    if (!seeds.empty()) {
      if (!WriteInputFixtures(empty_staging_directory, &seeds, error,
                              kBenchmarkTailGuardBytes))
        return false;
      for (size_t i = 0; i < seeds.size(); ++i)
        prepared.resets[i].seed = std::move(seeds[i]);
    }
    const auto &implementation =
        plan.implementations[invocation.implementation_index];
    auto [slot, added] = implementation_indices.emplace(
        implementation.implementation_id, implementations.size());
    if (added) {
      LoadedImplementation loaded;
      if (!LoadImplementation(implementation, context, workspace, &loaded,
                              error))
        return false;
      implementations.push_back(std::move(loaded));
    }
    auto &loaded = implementations[slot->second];
    if (invocation.request_root_ordinal >= loaded.root_names.size())
      return Fail(
          error,
          "selected request root is unavailable in retained implementation");
    const auto &kernel = loaded.root_names[invocation.request_root_ordinal];
    if (!CheckSignature(loaded.module.get(), kernel, prepared, error))
      return false;
    // A source declaration must resolve to exactly this retained public root.
    // Never select the first provider when distinct requests share that name.
    auto [owner, unique] =
        root_owners.emplace(kernel, implementation.implementation_id);
    if ((!unique && owner->second != implementation.implementation_id) ||
        (has_resets && kernel == "loomify.benchmark.reset_bytes"))
      return Fail(
          error,
          "distinct selected implementations share a kernel symbol in aggregate: " +
              kernel);
    const std::string benchmark = "gguf_case_" + identity.case_id;
    const std::string subject = benchmark + "_subject";
    source += CaseSource(prepared, invocation, kernel, subject, benchmark,
                         declared_roots.insert(kernel).second);
    records.push_back({index, std::move(identity), benchmark, subject});
  }
  std::vector<loomc_module_t *> modules;
  for (const auto &loaded : implementations)
    modules.push_back(loaded.module.get());
  std::vector<std::string> benchmarks;
  for (const auto &record : records)
    benchmarks.push_back(record.benchmark);
  Module linked(nullptr, loomc_module_release);
  if (!LinkCase(context, workspace, source, modules, benchmarks, &linked,
                error))
    return false;
  std::vector<EmittedCase> emitted;
  for (const auto &record : records) {
    auto &item = cases[record.case_index];
    const auto &invocation = plan.invocations[item.invocation_index];
    auto &prepared = item.inputs;
    EmittedCase result;
    const auto &implementation =
        plan.implementations[invocation.implementation_index];
    auto &loaded = implementations[implementation_indices.at(
        implementation.implementation_id)];
    const auto &kernel = loaded.root_names[invocation.request_root_ordinal];
    // The product shipped to the runner contains only direct check.case
    // launches. This temporary command is an ABI/grid proof, never published.
    Module proof(nullptr, loomc_module_release);
    const std::string proof_source =
        helper_source + "\n" +
        ProofSource(prepared, invocation, kernel, record.subject);
    if (!LinkCase(context, workspace, proof_source, {loaded.module.get()},
                  {record.subject}, &proof, error) ||
        !VerifyReplay(invocation, prepared, record.subject, proof.get(),
                      workspace, &result.timed_dispatches, error)) {
      if (error)
        *error = "benchmark replay " + invocation.invocation_id + " @" +
                 kernel + ": " + *error;
      return false;
    }
    result.safety_checks = {"allocation_tail_canaries"};
    if (!prepared.representative_weight_spans.empty())
      result.safety_checks.push_back("immutable_weight_spans");
    if (!prepared.resets.empty())
      result.safety_checks.push_back("immutable_reset_seeds");
    result.invocation_id = invocation.invocation_id;
    result.case_id = record.identity.case_id;
    result.logical_case_key = record.identity.logical_case_key;
    result.input_policy_digest = record.identity.input_policy_digest;
    result.benchmark_symbol = record.benchmark;
    result.setup_recipe_id = prepared.setup_recipe_id;
    result.measurement_subject = prepared.measurement_subject;
    result.reset_policy = prepared.reset_policy;
    result.arithmetic_check = prepared.arithmetic_check;
    result.weight_sources = std::move(prepared.weight_sources);
    result.roots = std::move(prepared.roots);
    result.resets = std::move(prepared.resets);
    emitted.push_back(std::move(result));
  }
  if (!WriteModule(linked.get(), empty_staging_directory / "benchmark.loombc",
                   LOOMC_SOURCE_FORMAT_BYTECODE, error) ||
      !WriteModule(linked.get(), empty_staging_directory / "benchmark.loom",
                   LOOMC_SOURCE_FORMAT_TEXT, error))
    return false;
  *out = std::move(emitted);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::benchmark
