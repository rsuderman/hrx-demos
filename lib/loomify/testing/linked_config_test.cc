// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "lib/loomify/testing/qwen3_fixture.h"
#include "loomc/artifact.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/interop.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace hrx::loomify_gguf;
namespace fs = std::filesystem;
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
using Module = Owner<loomc_module_t, loomc_module_release>;
constexpr const char *kAlignment =
    "llm.platform.projection.scaled_f16.row_alignment";
::testing::AssertionResult Ok(loomc_status_t status) {
  auto m = loomc_status_message(status);
  std::string message(m.data ? m.data : "", m.size);
  const bool ok = loomc_status_is_ok(status);
  loomc_status_free(status);
  return ok ? ::testing::AssertionSuccess()
            : ::testing::AssertionFailure() << message;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size).append("\n");
  }
  return text;
}
std::string Read(const fs::path &path) {
  std::ifstream stream(path);
  EXPECT_TRUE(stream) << path;
  return {std::istreambuf_iterator<char>(stream), {}};
}
class LinkedConfigTest : public ::testing::TestWithParam<bool> {
protected:
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment{nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr,
                                                        loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{
      nullptr, loomc_workspace_release};
  fs::path root;
  std::string parse_error;
  void SetUp() override {
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    root = fs::path(std::getenv("TEST_SRCDIR")) / std::getenv("TEST_WORKSPACE");
    loomc_target_environment_t *e = nullptr;
    ASSERT_TRUE(Ok(
        loomc_target_environment_create_amdgpu(loomc_allocator_system(), &e)));
    environment.reset(e);
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.structure_size = sizeof(target);
    target.target_environment = e;
    loomc_context_options_t options{};
    options.next = &target;
    loomc_context_t *c = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &c)));
    context.reset(c);
    loomc_workspace_t *w = nullptr;
    ASSERT_TRUE(
        Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)));
    workspace.reset(w);
  }
  std::string Product(const std::string &relative) {
    if (const auto *baseline = std::getenv("LOOMIFY_LINKED_CONFIG_BASELINE"))
      return (fs::path(baseline) / relative).string();
    return (root / relative).string();
  }
  std::string Fragment(const char *directory, const char *name) {
    return Product(std::string(directory) + "/" + name +
                   (GetParam() ? ".loombc" : "_text.loom"));
  }
  std::string SaveVariant(const std::string &text, const std::string &name) {
    const auto path = fs::path(std::getenv("TEST_TMPDIR")) /
                      (name + (GetParam() ? ".loombc" : ".loom"));
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t *s = nullptr;
    EXPECT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &s)));
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    const auto status = loomc_module_deserialize_from_source(
        context.get(), workspace.get(), s, nullptr, loomc_allocator_system(),
        &m, &r);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
    const auto parsed_ok = Ok(status);
    parse_error = Diagnostics(r);
    if (!parsed_ok || !r || !loomc_result_succeeded(r)) {
      std::fprintf(stderr, "VARIANT_PARSE %s: %s\n", name.c_str(),
                   parse_error.c_str());
      return {}; // Invalid typed definitions may fail before native linking.
    }
    loomc_module_serialize_options_t serialize{};
    serialize.format =
        GetParam() ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    EXPECT_TRUE(Ok(loomc_module_serialize_to_path(
        m, &serialize, loomc_make_cstring_view(path.c_str()),
        loomc_allocator_system())));
    return path.string();
  }
  Module Link(const std::string &source,
              const std::vector<std::string> &products, bool expected = true) {
    loomc_module_t *module = nullptr;
    std::string error;
    const bool success =
        LinkConfiguredProducts(context.get(), workspace.get(), source, products,
                               {"proof.root"}, true, &module, &error);
    EXPECT_EQ(success, expected) << error;
    if (!expected) {
      EXPECT_EQ(module, nullptr);
      EXPECT_FALSE(error.empty());
    }
    return Module(module, loomc_module_release);
  }
  std::string Text(loomc_module_t *module) {
    loomc_source_t *s = nullptr;
    EXPECT_TRUE(Ok(loomc_module_serialize_text_to_source(
        module, nullptr, loomc_allocator_system(), &s)));
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    const auto bytes = loomc_source_contents(s);
    return {reinterpret_cast<const char *>(bytes.data), bytes.data_length};
  }
  void CheckArchitectureCaller(loomc_module_t *module, const std::string &root,
                               bool ffn) {
    loomc_module_interop_view_t view{};
    loomc_result_t *r = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &r)));
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    ASSERT_TRUE(r && loomc_result_succeeded(r)) << Diagnostics(r);
    const auto constant = [&](loom_value_id_t value) -> int64_t {
      const auto *op = loom_value_def_op(loom_module_value(view.module, value));
      EXPECT_TRUE(op && loom_index_constant_isa(op));
      return op && loom_index_constant_isa(op)
                 ? loom_index_constant_value(op).i64
                 : -1;
    };
    const auto allocation = [&](loom_value_id_t value) -> int64_t {
      const auto *op = loom_value_def_op(loom_module_value(view.module, value));
      EXPECT_TRUE(op && loom_buffer_alloca_isa(op));
      return op && loom_buffer_alloca_isa(op)
                 ? constant(loom_buffer_alloca_byte_length(op))
                 : -1;
    };
    std::vector<std::vector<int64_t>> prepared, selected;
    bool found = false;
    std::function<void(const loom_region_t *)> walk =
        [&](const loom_region_t *region) {
          for (size_t b = 0; b < region->block_count; ++b)
            for (const auto *op = region->blocks[b]->first_op; op;
                 op = op->next_op) {
              if (loom_kernel_launch_isa(op)) {
                const auto name_view = loom_string_table_get(
                    &view.module->strings,
                    view.module->symbols
                        .entries[loom_kernel_launch_callee(op).symbol_id]
                        .name_id);
                const std::string name(name_view.data, name_view.size);
                const bool prep = name == "llm.projection.prepare_scaled_f16";
                const bool store =
                    name == "llm.platform.q4k_scaled_f16_store" ||
                    name == "llm.platform.q6k_scaled_f16_store";
                const bool gated = name == "llm.platform.q4k_scaled_f16_swiglu";
                if (prep || store || gated) {
                  const auto workloads = loom_kernel_launch_workloads(op);
                  ASSERT_EQ(workloads.count, 3u) << name;
                  std::vector<int64_t> shape;
                  for (size_t i = 0; i < workloads.count; ++i)
                    shape.push_back(constant(workloads.values[i]));
                  const auto args = loom_kernel_launch_arguments(op);
                  ASSERT_EQ(args.count, gated ? 9u : 8u) << name;
                  if (prep) {
                    prepared.push_back(shape);
                    EXPECT_EQ(allocation(args.values[6]), shape[1] * 8);
                    EXPECT_EQ(allocation(args.values[7]),
                              shape[1] * shape[2] * 2);
                  } else {
                    selected.push_back(shape);
                    EXPECT_EQ(allocation(args.values[args.count - 1]),
                              shape[0] * shape[2] * 4);
                    for (size_t i = 0; i < 3; ++i)
                      EXPECT_EQ(constant(args.values[i]), shape[i]);
                  }
                }
              }
              for (size_t i = 0; i < op->region_count; ++i)
                walk(loom_op_regions(op)[i]);
            }
        };
    for (size_t i = 0; i < view.module->symbols.count; ++i) {
      const auto &symbol = view.module->symbols.entries[i];
      const auto name =
          loom_string_table_get(&view.module->strings, symbol.name_id);
      if (std::string(name.data, name.size) == root) {
        ASSERT_TRUE(loom_command_def_isa(symbol.defining_op));
        found = true;
        walk(loom_command_def_body(symbol.defining_op));
      }
    }
    ASSERT_TRUE(found) << root;
    // Shapes come from the existing Qwen3ModelFacts fixture, while every
    // allocation/launch above is in the actual maintained architecture caller.
    EXPECT_EQ(prepared, (std::vector<std::vector<int64_t>>{
                            {17, 64, 1024}, {17, 64, ffn ? 3072 : 2048}}));
    EXPECT_EQ(selected,
              ffn ? (std::vector<std::vector<int64_t>>{{64, 1024, 3072},
                                                       {64, 3072, 1024}})
                  : (std::vector<std::vector<int64_t>>{{64, 1024, 2048},
                                                       {64, 1024, 1024},
                                                       {64, 1024, 1024},
                                                       {64, 2048, 1024}}));
  }
  int64_t FirstAllocation(loomc_module_t *module) {
    loomc_module_interop_view_t view{};
    loomc_result_t *r = nullptr;
    EXPECT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &r)));
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    EXPECT_TRUE(r && loomc_result_succeeded(r)) << Diagnostics(r);
    for (size_t i = 0; i < view.module->symbols.count; ++i) {
      const auto &symbol = view.module->symbols.entries[i];
      const auto name =
          loom_string_table_get(&view.module->strings, symbol.name_id);
      if (std::string(name.data, name.size) != "proof.root")
        continue;
      const auto *block = loom_region_const_entry_block(
          loom_command_def_body(symbol.defining_op));
      for (auto *op = block->first_op; op; op = op->next_op) {
        if (!loom_buffer_alloca_isa(op))
          continue;
        const auto value = loom_buffer_alloca_byte_length(op);
        const auto *constant =
            loom_value_def_op(loom_module_value(view.module, value));
        EXPECT_TRUE(constant && loom_index_constant_isa(constant));
        if (!constant || !loom_index_constant_isa(constant))
          return -1;
        return loom_index_constant_value(constant).i64;
      }
    }
    ADD_FAILURE() << "Missing actual root allocation";
    return -1;
  }
  std::string MatrixRoot(int rows) {
    std::ostringstream s;
    for (const auto *format : {"q4k", "q5k", "q6k", "q8_0", "q5_0", "q4_0",
                               "q4_1", "iq4_xs", "iq4_nl", "iq3_s", "q3k"})
      s << "config.def @ggml.linear_" << format
        << "_q8_1_x4.token_capacity = 512 : index\n"
        << "config.def @ggml.linear_" << format
        << "_q8_1_x4.output_capacity = 256 : index\n";
    s << "config.def @ggml.quantize_q8_1_x4.group_capacity = 1024 : index\n"
      << "func.decl pure @llm.projection.matrix_output_rows(%rows:index)->(index)\n"
      << "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%B:index,%count:buffer,%input:buffer,%weight:view<[%B]xi8>,%output:buffer)\n"
      << "command.def public @proof.root() launch(%count:buffer,%input:buffer,%weight:buffer) {\n"
      << "%T = index.constant " << rows << " : index\n"
      << "%K = index.constant 256 : index\n%N = index.constant 256 : index\n"
      << "%tag = index.constant 12 : index\n%B = index.constant 36864 : index\n"
      << "%origin = index.constant 0 : offset\n%word = index.constant 4 : offset\n"
      << "%P = func.call pure @llm.projection.matrix_output_rows(%T):(index)->(index)\n"
      << "%elements = index.mul %P, %N : index\n"
      << "%bytes = index.scale %elements,%word : index,offset -> offset\n"
      << "%output = buffer.alloca<global> align(256) %bytes : buffer\n"
      << "%w = buffer.view %weight[%origin] : buffer -> view<[%B]xi8>\n"
      << "template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%B,%count,%input,%w,%output):(index,index,index,index,index,buffer,buffer,view<[%B]xi8>,buffer)\n"
      << "command.return\n}\n";
    return s.str();
  }
  std::vector<std::string> MatrixProducts(const std::string &variant = {}) {
    return {
        Fragment("lib/programs/projection", "packed"),
        Fragment("lib/programs/projection", "prepared_rows"),
        Fragment("lib/programs/projection", "matrix"),
        Product("lib/platforms/amdgpu/gfx11/targets.loombc"),
        variant.empty()
            ? Product(
                  "lib/platforms/amdgpu/gfx11/projection/implementation.loombc")
            : variant};
  }
  std::string MatrixVariant(const std::string &config) {
    // Mutate only the fact in the actual merged product, retaining all leaf
    // definitions and native-MERGE declaration coalescing.
    auto text =
        Read(root /
             "lib/platforms/amdgpu/gfx11/projection/implementation_text.loom");
    const auto begin =
        text.find(std::string("config.def @") + kAlignment + " = ");
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos)
      return {};
    const auto end = text.find('\n', begin);
    text.replace(begin, end - begin, config);
    return SaveVariant(text, "matrix-variant");
  }
  std::string AttentionRoot(int rows, bool spans) {
    ConfigurationFacts facts;
    std::string error;
    EXPECT_TRUE(
        DeriveConfiguration(hrx::loomify_gguf::testing::QwenModelFacts(),
                            "fixture", {uint64_t(rows), 512}, &facts, &error))
        << error;
    std::ostringstream s;
    s << facts.source;
    if (spans)
      s << "kernel.decl @qwen38_attention_spans_f16_wmma(%T:index,%S:index) launch(%epoch:buffer,%rows:buffer,%origin:offset,%query:buffer,%gate:buffer,%state:buffer,%output:buffer)\n";
    else
      s << "kernel.decl @qwen38_attention_prefill_f16_wmma(%T:index) launch(%control:buffer,%query:buffer,%gate:buffer,%key:buffer,%value:buffer,%output:buffer)\n";
    s << "command.def public @proof.root() launch(%a:buffer,%b:buffer,%c:buffer,%d:buffer,%e:buffer,%f:buffer) {\n"
      << "%T = index.constant " << rows << " : index\n";
    if (spans)
      s << "%S = index.constant 1 : index\n%origin = index.constant 0 : offset\n"
        << "kernel.launch @qwen38_attention_spans_f16_wmma[%T,%S](%a,%b,%origin,%c,%d,%e,%f):[index,index](buffer,buffer,offset,buffer,buffer,buffer,buffer)\n";
    else
      s << "kernel.launch @qwen38_attention_prefill_f16_wmma[%T](%a,%b,%c,%d,%e,%f):[index](buffer,buffer,buffer,buffer,buffer,buffer)\n";
    s << "command.return\n}\n";
    return s.str();
  }
  std::vector<std::string> AttentionProducts(const std::string &variant = {}) {
    return {
        Fragment("lib/architectures/qwen35", "implementation"),
        Product("lib/platforms/amdgpu/gfx11/targets.loombc"),
        variant.empty()
            ? Product(
                  "lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc")
            : variant};
  }
  std::string AttentionVariant(const std::string &config) {
    return SaveVariant(
        config + "\n" +
            Read(root /
                 "lib/platforms/amdgpu/gfx11/attention_h256_schedule.loom"),
        "attention-variant");
  }
  void Observe(loomc_module_t *module, const std::string &name, bool compile,
               const std::vector<uint32_t> &expected_grid = {},
               const std::string &root_name = "proof.root") {
    const char *directory = std::getenv("LOOMIFY_LINKED_CONFIG_PROOF_DIR");
    const auto output =
        directory
            ? fs::path(directory) / (name + (GetParam() ? "-bc" : "-source"))
            : fs::path();
    if (directory) {
      fs::create_directories(output);
      for (const auto format :
           {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
        loomc_module_serialize_options_t options{};
        options.format = format;
        const auto path =
            output / (format == LOOMC_SOURCE_FORMAT_TEXT ? "module.loom"
                                                         : "module.loombc");
        ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
            module, &options, loomc_make_cstring_view(path.c_str()),
            loomc_allocator_system())));
      }
    }
    // Compile the reloaded final product, not merely its in-memory precursor.
    loomc_module_serialize_options_t roundtrip_options{};
    roundtrip_options.format =
        GetParam() ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *roundtrip_source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(module, &roundtrip_options,
                                                    loomc_allocator_system(),
                                                    &roundtrip_source)));
    Owner<loomc_source_t, loomc_source_release> retained(roundtrip_source,
                                                         loomc_source_release);
    loomc_module_t *roundtrip_module = nullptr;
    loomc_result_t *roundtrip_result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), roundtrip_source, nullptr,
        loomc_allocator_system(), &roundtrip_module, &roundtrip_result)));
    Module reloaded(roundtrip_module, loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> reloaded_result(
        roundtrip_result, loomc_result_release);
    ASSERT_TRUE(roundtrip_result && loomc_result_succeeded(roundtrip_result))
        << Diagnostics(roundtrip_result);
    module = reloaded.get();
    if (root_name.rfind("validated.qwen3.", 0) == 0)
      CheckArchitectureCaller(module, root_name,
                              root_name == "validated.qwen3.ffn0");
    const auto text = Text(module);
    EXPECT_EQ(text.find("config.get @llm.platform."), std::string::npos);
    loomc_target_profile_t *p = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(
        environment.get(), loomc_make_cstring_view("amdgpu:gfx1100"),
        loomc_allocator_system(), &p)));
    Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
        p, loomc_target_profile_release);
    const auto symbol = loomc_make_cstring_view(root_name.c_str());
    std::vector<loomc_request_t *> raw_requests;
    loomc_cmd_program_module_options_t options{};
    options.root_symbols = &symbol;
    options.root_symbol_count = 1;
    options.target_profile = p;
    options.artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING;
    options.request_sink.publish = [](void *user, loomc_request_t *request) {
      static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
      return loomc_ok_status();
    };
    options.request_sink.user_data = &raw_requests;
    loomc_product_t *product = nullptr;
    loomc_result_t *r = nullptr;
    const auto status = loomc_cmd_program_product_build_module(
        workspace.get(), module, &options, loomc_allocator_system(), &product,
        &r);
    Owner<loomc_product_t, loomc_product_release> program(
        product, loomc_product_release);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    std::vector<Owner<loomc_request_t, loomc_request_release>> requests;
    for (auto *request : raw_requests)
      requests.emplace_back(request, loomc_request_release);
    ASSERT_TRUE(Ok(status));
    ASSERT_TRUE(r && loomc_result_succeeded(r)) << Diagnostics(r);
    ASSERT_FALSE(requests.empty());
    ASSERT_EQ(loomc_cmd_program_product_program_count(product), 1u);
    loomc_cmd_program_t executable{};
    ASSERT_TRUE(loomc_cmd_program_product_program_at(product, 0, &executable));
    EXPECT_EQ(std::string(executable.symbol.data, executable.symbol.size),
              root_name);
    EXPECT_GT(executable.entry_requirement_count, 0u);
    if (directory) {
      const auto path = output / "command-program.bin";
      ASSERT_TRUE(Ok(loomc_artifact_write_to_path(
          &executable.artifact, loomc_make_cstring_view(path.c_str()),
          loomc_allocator_system())));
    }
    // Public diagnostic listing replaces removed arbitrary-dispatch reflection.
    // Preserve the complete listing and command image for exact B0/C ABI,
    // operand, buffer binding, ordering and transient-storage comparisons.
    std::string listing;
    size_t listing_count = 0;
    for (size_t i = 0; i < loomc_result_artifact_count(r); ++i) {
      const auto *artifact = loomc_result_artifact_at(r, i);
      if (std::string(artifact->format.data, artifact->format.size) !=
          LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM_LISTING)
        continue;
      ++listing_count;
      loomc_byte_sequence_callback_t callback{};
      callback.fn = [](void *user, loomc_byte_span_t bytes) {
        static_cast<std::string *>(user)->append(
            reinterpret_cast<const char *>(bytes.data), bytes.data_length);
        return loomc_ok_status();
      };
      callback.user_data = &listing;
      ASSERT_TRUE(
          Ok(loomc_byte_sequence_enumerate(artifact->contents, callback)));
      if (directory) {
        const auto path = output / "command-program.listing";
        ASSERT_TRUE(Ok(loomc_artifact_write_to_path(
            artifact, loomc_make_cstring_view(path.c_str()),
            loomc_allocator_system())));
      }
    }
    ASSERT_EQ(listing_count, 1u);
    ASSERT_FALSE(listing.empty());
    if (!expected_grid.empty()) {
      ASSERT_EQ(expected_grid.size(), 3u);
      // The public cmd-assembly diagnostic spells direct counts as references
      // to u32 constants. This bounded check does not decode command bytecode
      // or reconstruct an executable/benchmark from its diagnostic output.
      const std::regex constant(
          R"((%[A-Za-z0-9_.$]+) = cmd\.constant\.u32 ([0-9]+))");
      std::map<std::string, uint32_t> values;
      for (std::sregex_iterator it(listing.begin(), listing.end(), constant),
           end;
           it != end; ++it)
        values[(*it)[1]] = uint32_t(std::stoul((*it)[2]));
      const std::regex dispatch(
          R"(cmd\.dispatch\.direct(?:\.barrier)?<[^>]+>\[([^\]]+)\])");
      const std::regex any_dispatch(R"(cmd\.dispatch\.[^\n]+)");
      std::string last_dispatch;
      for (std::sregex_iterator
               it(listing.begin(), listing.end(), any_dispatch),
           end;
           it != end; ++it)
        last_dispatch = it->str();
      std::smatch direct;
      ASSERT_TRUE(std::regex_search(last_dispatch, direct, dispatch))
          << listing;
      std::istringstream grid(direct[1]);
      for (const auto expected : expected_grid) {
        std::string value;
        ASSERT_TRUE(bool(grid >> value)) << listing;
        if (value.back() == ',')
          value.pop_back();
        ASSERT_EQ(values.count(value), 1u) << value << '\n' << listing;
        EXPECT_EQ(values.at(value), expected);
      }
    }
    if (!compile)
      return;
    loomc_compiler_t *c = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(context.get(), nullptr,
                                         loomc_allocator_system(), &c)));
    Owner<loomc_compiler_t, loomc_compiler_release> compiler(
        c, loomc_compiler_release);
    for (size_t i = 0; i < requests.size(); ++i) {
      auto *source = loomc_request_source(requests[i].get());
      if (directory) {
        const auto bytes = loomc_source_contents(source);
        std::ofstream stream(output /
                                 ("request-" + std::to_string(i) + ".request"),
                             std::ios::binary);
        stream.write(reinterpret_cast<const char *>(bytes.data),
                     bytes.data_length);
      }
      loomc_module_t *m = nullptr;
      loomc_result_t *parsed = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context.get(), workspace.get(), source, nullptr,
          loomc_allocator_system(), &m, &parsed)));
      Module leaf(m, loomc_module_release);
      Owner<loomc_result_t, loomc_result_release> parse_result(
          parsed, loomc_result_release);
      ASSERT_TRUE(parsed && loomc_result_succeeded(parsed))
          << Diagnostics(parsed);
      if (directory) {
        std::ofstream stream(output /
                             ("request-" + std::to_string(i) + ".loom"));
        stream << Text(m);
      }
      loomc_compile_artifact_options_t artifact_options{};
      artifact_options.target_profile = p;
      loomc_result_t *compiled = nullptr;
      ASSERT_TRUE(Ok(loomc_compile_artifact(
          c, workspace.get(), nullptr, m, &artifact_options,
          loomc_allocator_system(), &compiled)));
      Owner<loomc_result_t, loomc_result_release> compiled_result(
          compiled, loomc_result_release);
      ASSERT_TRUE(compiled && loomc_result_succeeded(compiled))
          << Diagnostics(compiled);
      EXPECT_GT(loomc_result_artifact_count(compiled), 0u);
      if (directory)
        for (size_t j = 0; j < loomc_result_artifact_count(compiled); ++j) {
          const auto *a = loomc_result_artifact_at(compiled, j);
          const auto path = output / ("request-" + std::to_string(i) +
                                      "-artifact-" + std::to_string(j));
          ASSERT_TRUE(Ok(loomc_artifact_write_to_path(
              a, loomc_make_cstring_view(path.c_str()),
              loomc_allocator_system())));
        }
    }
  }
};
TEST_P(LinkedConfigTest, MatrixAllocationsAndActualRequests) {
  for (int rows : {1, 16, 17, 65, 128, 129, 257, 512}) {
    SCOPED_TRACE(rows);
    auto module = Link(MatrixRoot(rows), MatrixProducts());
    ASSERT_NE(module, nullptr);
    const auto text = Text(module.get());
    const auto padded = rows <= 16 ? rows : (rows + 63) / 64 * 64;
    // Actual first caller allocation uses the selected query, before dispatch.
    EXPECT_EQ(FirstAllocation(module.get()), padded * 256 * 4) << text;
    Observe(module.get(), "matrix-t" + std::to_string(rows), rows == 17,
            rows > 16 ? std::vector<uint32_t>{4u, uint32_t(padded / 64), 1u}
                      : std::vector<uint32_t>{});
  }
}
TEST_P(LinkedConfigTest, AttentionActualDenseAndSpanRequests) {
  for (int rows : {1, 17, 128})
    for (bool spans : {false, true}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(spans);
      auto module = Link(AttentionRoot(rows, spans), AttentionProducts());
      ASSERT_NE(module, nullptr);
      Observe(module.get(),
              std::string(spans ? "span-t" : "dense-t") + std::to_string(rows),
              rows == 17, {uint32_t((rows + 15) / 16), 8u, 1u});
    }
}
TEST_P(LinkedConfigTest, ActualQwen3CallersAllocateAndPreparePaddedRows) {
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(hrx::loomify_gguf::testing::Qwen3ModelFacts(),
                                  "fixture", {17, 128}, &facts, &error))
      << error;
  for (const bool ffn : {false, true}) {
    const auto name =
        std::string("validated.qwen3.") + (ffn ? "ffn0" : "attention0");
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(LinkConfiguredProducts(
        context.get(), workspace.get(), facts.source,
        {Product("lib/architectures/qwen3/layer_fixture.loombc"),
         Product("lib/platforms/amdgpu/gfx11/targets.loombc"),
         Product(
             "lib/platforms/amdgpu/gfx11/projection/implementation.loombc")},
        {name.c_str()}, true, &raw, &error))
        << error;
    Module module(raw, loomc_module_release);
    ASSERT_NE(module, nullptr);
    Observe(module.get(), ffn ? "qwen3-ffn-t17" : "qwen3-qkv-t17", false, {},
            name);
  }
}
TEST_P(LinkedConfigTest, MatrixAlignmentComesFromSelectedProduct) {
  const auto config =
      std::string("config.def @") + kAlignment + " = 128 : index\n";
  const auto variant = MatrixVariant(config);
  ASSERT_FALSE(variant.empty()) << parse_error;
  auto module = Link(MatrixRoot(17), MatrixProducts(variant));
  ASSERT_NE(module, nullptr);
  const auto text = Text(module.get());
  EXPECT_EQ(FirstAllocation(module.get()), 131072) << text;
  Observe(module.get(), "matrix-t17-overaligned", true, {4u, 2u, 1u});
}
TEST_P(LinkedConfigTest, MissingAndInvalidMatrixDefinitionsFail) {
  for (const auto &definition :
       {std::string(),
        std::string("config.def @") + kAlignment + " = 0 : index\n",
        std::string("config.def @") + kAlignment + " = -1 : index\n",
        std::string("config.def @") + kAlignment + " = 513 : index\n",
        std::string("config.def @") + kAlignment + " = 32 : index\n",
        std::string("config.def @") + kAlignment + " = 64.0 : f32\n"}) {
    SCOPED_TRACE(definition);
    const auto variant = MatrixVariant(definition);
    if (!variant.empty()) {
      Link(MatrixRoot(17), MatrixProducts(variant), false);
    } else {
      EXPECT_NE(parse_error.find("llm.platform."), std::string::npos)
          << parse_error;
    }
  }
}
TEST_P(LinkedConfigTest, MissingAndInconsistentAttentionDefinitionsFail) {
  const auto current =
      Read(root / "lib/platforms/amdgpu/gfx11/attention_h256_config.loom");
  const auto valid = AttentionVariant(current);
  ASSERT_FALSE(valid.empty());
  ASSERT_NE(Link(AttentionRoot(17, true), AttentionProducts(valid)), nullptr);
  std::vector<std::string> variants{std::string()};
  for (const auto &pair :
       {std::pair<std::string, std::string>{"= 16 :", "= 32 :"},
        {"= 256 :", "= 128 :"},
        {"= 16 :", "= 0 :"}}) {
    auto text = current;
    const auto at = text.find(pair.first);
    ASSERT_NE(at, std::string::npos);
    text.replace(at, pair.first.size(), pair.second);
    variants.push_back(text);
  }
  for (const auto &text : variants) {
    SCOPED_TRACE(text);
    const auto variant = AttentionVariant(text);
    if (!variant.empty()) {
      Link(AttentionRoot(17, true), AttentionProducts(variant), false);
    } else {
      EXPECT_NE(parse_error.find("llm.platform."), std::string::npos)
          << parse_error;
    }
  }
}
TEST_P(LinkedConfigTest, ConflictingDefinitionFailsNativeLink) {
  const auto source = MatrixRoot(17) + std::string("config.def @") +
                      kAlignment + " = 128 : index\n";
  loomc_module_t *raw = nullptr;
  std::string error;
  const auto success = LinkConfiguredProducts(
      context.get(), workspace.get(), source, MatrixProducts(), {"proof.root"},
      true, &raw, &error);
  Module module(raw, loomc_module_release);
  EXPECT_FALSE(success);
  EXPECT_EQ(raw, nullptr);
  EXPECT_NE(error.find(kAlignment), std::string::npos) << error;
  EXPECT_NE(error.find("incompatible contract field 'value'"),
            std::string::npos)
      << error;
}
TEST_P(LinkedConfigTest, CanonicalSharedRootNeedsNoMatrixOrAttentionFacts) {
  auto source = MatrixRoot(1);
  source.resize(
      source.find("func.decl pure @llm.projection.matrix_output_rows"));
  source +=
      "template.decl @llm.projection.pack_input(%policy:index,%rows:index,%capacity:index,%K:index,%begin:index,%fixed:i32,%origin:offset,%count:buffer,%input:buffer,%packed:buffer)\n"
      "command.def public @proof.root() launch(%count:buffer,%input:buffer,%output:buffer) {\n"
      "%one = index.constant 1 : index\n%zero = index.constant 0 : index\n"
      "%K = index.constant 256 : index\n%origin = index.constant 0 : offset\n"
      "%dynamic = scalar.constant -1 : i32\n"
      "template.apply<@llm.projection.pack_input>(%one,%one,%one,%K,%zero,%dynamic,%origin,%count,%input,%output):(index,index,index,index,index,i32,offset,buffer,buffer,buffer)\n"
      "command.return\n}\n";
  auto module =
      Link(source, {Fragment("lib/programs/projection", "packed"),
                    Product("lib/platforms/amdgpu/gfx11/targets.loombc")});
  ASSERT_NE(module, nullptr);
  Observe(module.get(), "sparse-canonical", false);
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, LinkedConfigTest,
                         ::testing::Bool());
} // namespace
