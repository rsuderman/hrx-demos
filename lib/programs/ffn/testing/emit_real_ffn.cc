// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual GGUF facts plus maintained wrappers; arithmetic/state oracles live in
// the existing kernel tests. Old-versus-new parity is external migration
// evidence.
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "tools/model/model_descriptor.h"
#include "loomc/target/amdgpu.h"
#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using namespace hrx;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
// Test-only operation facts are verified separately against immutable GGUF
// headers. Unrelated model tensor formats and auxiliary blocks are not
// admitted.
bool PartialSource(const char *path, std::string *source, std::string *error) {
  uint64_t profile, hidden, ffn, gate_tag, up_tag, gate_bytes, up_bytes;
  std::string gate_name, up_name, trailing;
  std::ifstream input(path);
  if (!(input >> profile >> hidden >> ffn >> gate_tag >> up_tag >> gate_bytes >>
        up_bytes >> gate_name >> up_name) ||
      (input >> trailing)) {
    *error = "partial facts require profile H F gt ut gb ub gate_name up_name";
    return false;
  }
  const auto supported = [](uint64_t tag) {
    return tag == 2 || tag == 3 || tag == 6 || tag == 8 || tag == 12 ||
           tag == 13 || tag == 14 || tag == 23;
  };
  const auto valid_name = [](const std::string &name, const char *suffix) {
    const std::string end = suffix;
    return name.rfind("blk.", 0) == 0 && name.size() > end.size() &&
           name.compare(name.size() - end.size(), end.size(), end) == 0 &&
           name.find_first_not_of(
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._") ==
               std::string::npos;
  };
  uint64_t expected_gate = 0, expected_up = 0;
  if (profile > 1 || hidden < 128 || hidden > 32768 || hidden % 128 ||
      ffn < 256 || ffn > 32768 || ffn % 256 || !supported(gate_tag) ||
      !supported(up_tag) ||
      ((gate_tag == 14 || up_tag == 14) && hidden > 17408) ||
      !valid_name(gate_name, ".ffn_gate.weight") ||
      !valid_name(up_name, ".ffn_up.weight") ||
      !model::EncodedTensorByteSize(gate_tag, hidden, ffn, &expected_gate) ||
      !model::EncodedTensorByteSize(up_tag, hidden, ffn, &expected_up) ||
      expected_gate != gate_bytes || expected_up != up_bytes) {
    *error = "invalid partial gate/up facts, names or encoded extents";
    return false;
  }
  std::ostringstream text;
  text
      << "template.decl @llm.projection.policy_rows(%profile: index, %rows: index, %input_width: index, %output_width: index, %tag: index, %weight_bytes: index, %count: buffer, %input: buffer, %weight: view<[%weight_bytes]xi8>, %output: buffer)\n"
      << "kernel.decl @llm.projection.active_swiglu(%rows: index, %width: index) launch(%rows: index, %width: index, %count: buffer, %gate: buffer, %up: buffer, %output: buffer)\n"
      << "kernel.decl @llm.projection.prepare_f32_prefix(%rows: index, %width: index) launch(%rows: index, %width: index, %count: buffer, %input: buffer, %output: buffer)\n";
  for (const char *format : {"q4k", "q5k", "q6k", "q5_0", "q8_0", "q4_0",
                             "q4_1", "iq4_xs", "iq4_nl", "iq3_s", "q3k"})
    text << "config.def @ggml.linear_" << format
         << "_q8_1_x4.token_capacity = 512 : index\n"
         << "config.def @ggml.linear_" << format
         << "_q8_1_x4.output_capacity = " << ffn << " : index\n";
  text << "config.def @ggml.quantize_q8_1_x4.group_capacity = "
       << 512 * (hidden / 128) << " : index\n";
  for (int rows : {1, 17}) {
    text
        << "command.def public @partial.ffn.gate_up.t" << rows
        << "() launch(%parameters: buffer, %input: buffer, %count: buffer, %output: buffer) {\n";
    for (const auto &[key, value] :
         std::vector<std::pair<const char *, uint64_t>>{
             {"profile", profile},
             {"rows", uint64_t(rows)},
             {"hidden", hidden},
             {"ffn", ffn},
             {"gt", gate_tag},
             {"ut", up_tag},
             {"gb", gate_bytes},
             {"ub", up_bytes}})
      text << "  %" << key << " = index.constant " << value << " : index\n";
    text
        << "  %gate_weight = command.parameter %parameters, \"" << gate_name
        << "\" : view<[%gb]xi8>\n"
        << "  %up_weight = command.parameter %parameters, \"" << up_name
        << "\" : view<[%ub]xi8>\n"
        << "  %scratch_bytes = index.constant " << uint64_t(rows) * ffn * 4
        << " : offset\n"
        << "  %gate = buffer.alloca<global> align(256) %scratch_bytes : buffer\n"
        << "  %up = buffer.alloca<global> align(256) %scratch_bytes : buffer\n"
        << "  template.apply<@llm.projection.policy_rows>(%profile, %rows, %hidden, %ffn, %gt, %gb, %count, %input, %gate_weight, %gate) : (index, index, index, index, index, index, buffer, buffer, view<[%gb]xi8>, buffer)\n"
        << "  template.apply<@llm.projection.policy_rows>(%profile, %rows, %hidden, %ffn, %ut, %ub, %count, %input, %up_weight, %up) : (index, index, index, index, index, index, buffer, buffer, view<[%ub]xi8>, buffer)\n"
        << "  kernel.launch @llm.projection.active_swiglu[%rows, %ffn](%rows, %ffn, %count, %gate, %up, %output) : [index, index](index, index, buffer, buffer, buffer, buffer)\n"
        << "  kernel.launch @llm.projection.prepare_f32_prefix[%rows, %ffn](%rows, %ffn, %count, %output, %output) : [index, index](index, index, buffer, buffer, buffer)\n"
        << "  command.return\n}\n";
  }
  *source = text.str();
  return true;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    std::fprintf(
        stderr,
        "usage: emitter LIBRARY_ROOT GGUF llama|qwen OUTPUT_STEM [PREFILL_ROWS]\n"
        "       emitter LIBRARY_ROOT VERIFIED_FACTS partial OUTPUT_STEM\n");
    return 1;
  }
  uint64_t rows = 4;
  if (argc == 6) {
    const auto parsed =
        std::from_chars(argv[5], argv[5] + std::strlen(argv[5]), rows);
    if (parsed.ec != std::errc() || *parsed.ptr || rows < 1 || rows > 512)
      return 1;
  }
  const std::string family = argv[3];
  if (family == "partial" && argc != 5)
    return 1;
  if (family != "llama" && family != "qwen" && family != "partial")
    return 1;
  model::ModelDescriptor descriptor;
  std::string error;
  ConfigurationFacts config;
  if (family == "partial") {
    if (!PartialSource(argv[2], &config.source, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  } else {
    std::unique_ptr<FILE, decltype(&std::fclose)> input(
        std::fopen(argv[2], "rb"), std::fclose);
    if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0))
      return 1;
    model::GgufCatalog catalog;
    if (!model::ReadGgufCatalog(input.get(), &catalog, &error) ||
        !model::LoadModelDescriptor(catalog, &descriptor, &error) ||
        descriptor.family != (family == "llama" ? "llama" : "qwen35") ||
        !DeriveConfiguration(descriptor, argv[2], {rows, 512}, &config,
                             &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  }
  loomc_target_environment_t *env = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &env)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = env;
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  std::vector<std::string> names;
  if (family == "partial") {
    names = {"partial.ffn.gate_up.t1", "partial.ffn.gate_up.t17"};
  } else {
    for (int layer : family == "llama" ? std::vector<int>{0, 1, 3}
                                       : (descriptor.embedding_width == 2560
                                              ? std::vector<int>{0, 4}
                                              : std::vector<int>{0, 1}))
      for (const char *phase : {"prefill", "decode"})
        names.push_back("real." + family + ".ffn.layer" +
                        std::to_string(layer) + "." + phase);
  }
  if (family == "llama")
    for (int layer : {0, 3})
      for (const char *suffix :
           {"configured_prefill", "configured_prefill_twice"})
        names.push_back("real.llama.ffn.layer" + std::to_string(layer) + "." +
                        suffix);
  std::vector<const char *> roots;
  for (const auto &name : names)
    roots.push_back(name.c_str());
  auto product = (std::filesystem::path(argv[1]) / "lib/programs/ffn/testing" /
                  (family == "partial" ? "partial_provider.loombc"
                                       : family + "_fixture.loombc"))
                     .string();
  loomc_module_t *module = nullptr;
  if (!LinkConfiguredProducts(context, workspace, config.source, {product},
                              roots, true, &module, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  bool ok = true;
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialize{};
    serialize.format = format;
    auto path = std::string(argv[4]) +
                (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc");
    ok = Ok(loomc_module_serialize_to_path(
             module, &serialize, loomc_make_cstring_view(path.c_str()),
             loomc_allocator_system())) &&
         ok;
  }
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(env);
  return ok ? 0 : 1;
}
