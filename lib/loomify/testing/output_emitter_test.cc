// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/template_analysis.h"
#include "tools/model/text_metadata.h"
#include "iree/base/internal/json.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/text/jinja/output_emitter.h"
#include "lib/loomify/text/jinja/request_emitter.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja::testing {
namespace {
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}
std::string Required(const char *name) {
  const char *value = std::getenv(name);
  EXPECT_TRUE(value && *value) << name;
  return value ? value : "";
}

struct OutputFixture : GeneratedInputRenderer {
  std::string original_source, response, prompt_suffix, frame_suffix, header;
  bool native_tools = false, exact_prompt = false;
};

// This wrapper packages actual scanner results through PreparedTemplate's one
// buffer ABI. Its scalar checks are independent expectations, not a scanner.
GeneratedInputRenderer Scanner(std::string_view open, std::string_view close,
                               std::string_view tool, int64_t previous,
                               int64_t state, int phase, int64_t kind,
                               int64_t end, int64_t next,
                               std::string_view after_open = "\n",
                               std::string_view after_close = "\n\n") {
  GeneratedInputRenderer product;
  product.entrypoint = "test.scan";
  product.helper_requirements = {"lib/text/common/strings.loom",
                                 "lib/text/output/stream.loom"};
  product.source = R"(
func.decl @text.output_check(%condition: i1)
func.decl @text.output_scan(%response: buffer, %previous: i64, %state: i64, %phase: i32, %open: buffer, %close: buffer, %tool: buffer, %after_open: buffer, %after_close: buffer) -> (i64, i64, i64, i64)
func.decl @text.range(%text: buffer, %begin: i64, %length: i64) -> (buffer)
)";
  for (const auto &[name, value] :
       {std::pair<std::string_view, std::string_view>{"open", open},
        {"close", close},
        {"tool", tool},
        {"after_open", after_open},
        {"after_close", after_close}})
    product.source += "global.rodata.def @test." + std::string(name) +
                      " = bytes(\"" + Hex(value) + "\")\n";
  product.source +=
      "func.def public @test.scan(%response: buffer) -> (buffer) {\n";
  for (const auto *name :
       {"open", "close", "tool", "after_open", "after_close"})
    product.source += "  %" + std::string(name) + " = global.load @test." +
                      name + " : buffer\n";
  product.source +=
      "  %previous = scalar.constant " + std::to_string(previous) + " : i64\n";
  product.source +=
      "  %state = scalar.constant " + std::to_string(state) + " : i64\n";
  product.source +=
      "  %phase = scalar.constant " + std::to_string(phase) + " : i32\n";
  product.source += R"(
  %kind, %begin, %end, %next = func.call @text.output_scan(%response, %previous, %state, %phase, %open, %close, %tool, %after_open, %after_close) : (buffer, i64, i64, i32, buffer, buffer, buffer, buffer, buffer) -> (i64, i64, i64, i64)
)";
  for (const auto &[name, value] :
       {std::pair<std::string_view, int64_t>{"kind", kind},
        {"begin", previous},
        {"end", end},
        {"next", next}}) {
    const std::string n(name);
    product.source += "  %expected_" + n + " = scalar.constant " +
                      std::to_string(value) + " : i64\n";
    product.source += "  %same_" + n + " = scalar.cmpi eq, %" + n +
                      ", %expected_" + n + " : i64\n";
    product.source +=
        "  func.call @text.output_check(%same_" + n + ") : (i1)\n";
  }
  product.source += R"(
  %length = scalar.subi %end, %begin : i64
  %result = func.call @text.range(%response, %begin, %length) : (buffer, i64, i64) -> (buffer)
  func.return %result : buffer
}
)";
  return product;
}

GeneratedInputRenderer JsonParser(const TextToolFormat &format) {
  GeneratedInputRenderer result;
  result.entrypoint = "test.parse";
  result.helper_requirements = {
      "lib/text/common/strings.loom", "lib/text/common/json.loom",
      "lib/text/output/stream.loom", "lib/text/output/json_calls.loom",
      "lib/text/output/schema.loom"};
  result.source = R"(
func.decl @output_json.parse_json_calls(%schemas: buffer, %response: buffer, %open: buffer, %close: buffer, %name: buffer, %arguments: buffer, %json_arguments: i1) -> (buffer)
)";
  for (const auto &[name, value] :
       {std::pair<std::string_view, std::string_view>{"open", format.open},
        {"close", format.close},
        {"name", format.name_field},
        {"arguments", format.arguments_field}})
    result.source += "global.rodata.def @test." + std::string(name) +
                     " = bytes(\"" + Hex(value) + "\")\n";
  result.source += R"(
func.def public @test.parse(%response: buffer) -> (buffer) {
  %schemas = buffer.constant utf8("[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]") : buffer
  %open = global.load @test.open : buffer
  %close = global.load @test.close : buffer
  %name = global.load @test.name : buffer
  %arguments = global.load @test.arguments : buffer
  %raw_json_wire = scalar.constant false : i1
  %calls = func.call @output_json.parse_json_calls(%schemas, %response, %open, %close, %name, %arguments, %raw_json_wire) : (buffer, buffer, buffer, buffer, buffer, buffer, i1) -> (buffer)
  func.return %calls : buffer
}
)";
  return result;
}

constexpr std::string_view kConditionalSource =
    "{% for m in messages %}{% if m.role == 'assistant' %}"
    "{{ '<turn>assistant:' }}"
    "{% if m.reasoning_content %}{{ '<reason>' + m.reasoning_content + '</reason>' }}{% endif %}"
    "{{ m.content }}"
    "{% if m.tool_calls %}{% for c in m.tool_calls %}"
    "{% if alternate_fields | default(false) %}"
    "{{ '<invoke>{\"operation\":' }}{{ c.function.name | tojson }}"
    "{{ ',\"inputs\":' }}{{ c.function.arguments | tojson }}"
    "{% else %}"
    "{{ '<invoke>{\"name\":' }}{{ c.function.name | tojson }}"
    "{{ ',\"arguments\":' }}{{ c.function.arguments | tojson }}"
    "{% endif %}{{ '}</invoke>' }}{% endfor %}{% endif %}"
    "{{ '</turn>' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<turn>assistant:' }}"
    "{% if enable_thinking | default(false) %}{{ '<reason>' }}{% endif %}"
    "{% endif %}";

// Ordinary callback fixtures use one source-proven field grammar. The original
// request-dependent grammar remains below as an explicit fixed-ABI rejection.
const std::string kFixedFieldsSource =
    "{% set alternate_fields = false %}" + std::string(kConditionalSource);

constexpr std::string_view kUnicodeRequest =
    R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"查找-name.点","parameters":{"type":"object","properties":{"启用-flag.位":{"type":"boolean"},"空值":{"type":"null"},"text":{"type":"string"},"lines":{"type":"string"},"opaque":{"type":"string"}}}}}]})";

constexpr std::string_view kXmlRequest =
    R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"b":{"type":"boolean"},"c":{"type":"boolean"},"s":{"type":"string"},"obj":{"type":"object"},"arr":{"type":"array"}},"required":["b"]}}},{"type":"function","function":{"name":"g","parameters":{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}}}]})";

bool OutputProgram(std::string_view source, std::string_view response,
                   OutputFixture *out, std::string *error,
                   const TextFormatBindings &bindings = {}) {
  TextFormatAnalysis analysis;
  Diagnostic diagnostic;
  GeneratedInputRenderer selector, adaptation, adapters, output;
  if (!AnalyzeTextFormat(source, "output-consumer.jinja", bindings, &analysis,
                         &diagnostic) ||
      !CompileTemplateSelector(source, "output-consumer.jinja",
                               analysis.selector_template, &selector,
                               &diagnostic) ||
      !CompileTemplateSelector(source, "output-consumer.jinja",
                               analysis.adaptation_selector_template,
                               &adaptation, &diagnostic,
                               TemplateSelectorKind::Adaptation) ||
      !EmitRequestAdapters("output-consumer.jinja", analysis, adaptation,
                           bindings, {}, &adapters, &diagnostic) ||
      !EmitOutputParser("output-consumer.jinja", analysis, &output,
                        &diagnostic)) {
    *error = diagnostic.message;
    return false;
  }
  output.entrypoint = "test.output_callbacks";
  output.source +=
      "global.rodata.def @test.response = bytes(\"" + Hex(response) + "\")\n";
  output.source += R"(
func.decl @text.range(%text: buffer, %begin: i64, %length: i64) -> (buffer)
func.decl @text.lookup(%json: buffer, %key: buffer) -> (buffer, i64, i1)
func.decl @text.join8(%a: buffer, %b: buffer, %c: buffer, %d: buffer, %e: buffer, %f: buffer, %g: buffer, %h: buffer) -> (buffer)
func.decl @text.json_quote(%text: buffer) -> (buffer)
func.decl @text.request_environment(%request: buffer, %operation: i32) -> (buffer)
func.def public @test.output_callbacks(%request: buffer) -> (buffer) {
  %response = global.load @test.response : buffer
  %live = scalar.constant 0 : i32
  %env = func.call @text.request_environment(%request, %live) : (buffer, i32) -> (buffer)
  %format, %startup = func.call @jinja.select_format(%env) : (buffer) -> (i64, i64)
  func.call @text.output_check_environment(%env, %format) : (buffer, i64)
  %initial = func.call @text.output_initial(%format, %startup) : (i64, i64) -> (i64)
  %zero = scalar.constant 0 : i64
  %one = scalar.constant 1 : i64
  %two = scalar.constant 2 : i64
  %phase = scalar.constant 1 : i32
  %empty = buffer.constant utf8("") : buffer
  %true = scalar.constant true : i1
  %size = buffer.length %response
  %size64 = index.cast %size : offset to i64
  %cursor, %state, %content, %reason, %active = scf.while(%cursor = %zero : i64, %state = %initial : i64, %content = %empty : buffer, %reason = %empty : buffer, %active = %true : i1) -> (i64, i64, buffer, buffer, i1) {
    %within = scalar.cmpi ult, %cursor, %size64 : i64
    %more = scalar.andi %within, %active : i1
    scf.condition %more, %cursor, %state, %content, %reason, %active : i1, i64, i64, buffer, buffer, i1
  } do(%cursor: i64, %state: i64, %content: buffer, %reason: buffer, %active: i1) {
    %kind, %begin, %end, %next = func.call @output_next(%response, %cursor, %state, %phase) : (buffer, i64, i64, i32) -> (i64, i64, i64, i64)
    %length = scalar.subi %end, %begin : i64
    %span = func.call @text.range(%response, %begin, %length) : (buffer, i64, i64) -> (buffer)
    %is_content = scalar.cmpi eq, %kind, %one : i64
    %is_reason = scalar.cmpi eq, %kind, %two : i64
    %c = scf.select %is_content, %span, %empty : buffer
    %r = scf.select %is_reason, %span, %empty : buffer
    %next_content = func.call @text.join8(%content, %c, %empty, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
    %next_reason = func.call @text.join8(%reason, %r, %empty, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
    %advanced = scalar.cmpi ugt, %end, %cursor : i64
    scf.yield %end, %next, %next_content, %next_reason, %advanced : i64, i64, buffer, buffer, i1
  }
  %remaining = scalar.subi %size64, %cursor : i64
  %suffix = func.call @text.range(%response, %cursor, %remaining) : (buffer, i64, i64) -> (buffer)
  %tools_key = buffer.constant utf8("tools") : buffer
  %schemas, %schema_kind, %has_schemas = func.call @text.lookup(%request, %tools_key) : (buffer, buffer) -> (buffer, i64, i1)
  %calls = func.call @parse_tools(%schemas, %suffix, %zero) : (buffer, buffer, i64) -> (buffer)
  %cq = func.call @text.json_quote(%content) : (buffer) -> (buffer)
  %rq = func.call @text.json_quote(%reason) : (buffer) -> (buffer)
  %a = buffer.constant utf8("{\"content\":") : buffer
  %b = buffer.constant utf8(",\"reasoning\":") : buffer
  %c = buffer.constant utf8(",\"calls\":") : buffer
  %z = buffer.constant utf8("}") : buffer
  %result = func.call @text.join8(%a, %cq, %b, %rq, %c, %calls, %z, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
  func.return %result : buffer
}
)";
  if (!CombineGeneratedTemplates({selector, adapters, output},
                                 output.entrypoint, out, &diagnostic)) {
    *error = diagnostic.message;
    return false;
  }
  out->original_source = source;
  out->response = response;
  out->frame_suffix = analysis.alternatives.front().frame_suffix;
  out->header = analysis.alternatives.front().assistant_header;
  out->native_tools =
      std::any_of(analysis.alternatives.begin(), analysis.alternatives.end(),
                  [](const auto &format) {
                    return format.tools.kind == TextToolKind::TaggedNamed ||
                           format.tools.kind == TextToolKind::TaggedKeyValue;
                  });
  return true;
}

// Couple the original compiled prompt bytes and the first output transition
// through the same actual request environment. The assertion is an independent
// source expectation; neither a rendered probe nor a profile supplies it.
bool CheckPromptPrefix(std::string_view source, std::string_view prefix,
                       OutputFixture *program, std::string *error,
                       bool exact = false) {
  GeneratedInputRenderer renderer, combined;
  Diagnostic diagnostic;
  if (!CompileChatTemplate(source, "output-prompt.jinja", &renderer,
                           &diagnostic) ||
      !CombineGeneratedTemplates({renderer, *program}, "test.prompt_and_output",
                                 &combined, &diagnostic)) {
    *error = diagnostic.message;
    return false;
  }
  combined.source += "global.rodata.def @test.expected_prefix = bytes(\"" +
                     Hex(prefix) + "\")\n";
  if (combined.source.find("func.decl @text.at(") == std::string::npos)
    combined.source +=
        "func.decl @text.at(%text: buffer, %at: offset, %marker: buffer) -> (i1)\n";
  if (exact &&
      combined.source.find("func.decl @text.equal(") == std::string::npos)
    combined.source +=
        "func.decl @text.equal(%a: buffer, %b: buffer) -> (i1)\n";
  combined.source += R"(
func.def public @test.prompt_and_output(%request: buffer) -> (buffer) {
  %live = scalar.constant 0 : i32
  %env = func.call @text.request_environment(%request, %live) : (buffer, i32) -> (buffer)
  %prompt = func.call @jinja.evaluate(%env) : (buffer) -> (buffer)
  %prefix = global.load @test.expected_prefix : buffer
  %length = buffer.length %prompt
  %size = buffer.length %prefix
  %fits = index.cmp uge, %length, %size : offset
  func.call @text.output_check(%fits) : (i1)
  %at = index.sub %length, %size : offset
  %matches = func.call @text.at(%prompt, %at, %prefix) : (buffer, offset, buffer) -> (i1)
  func.call @text.output_check(%matches) : (i1)
)";
  if (exact)
    combined.source += R"(
  %exact = func.call @text.equal(%prompt, %prefix) : (buffer, buffer) -> (i1)
  func.call @text.output_check(%exact) : (i1)
)";
  combined.source += R"(
  %result = func.call @test.output_callbacks(%request) : (buffer) -> (buffer)
  func.return %result : buffer
}
)";
  static_cast<GeneratedInputRenderer &>(*program) = std::move(combined);
  program->prompt_suffix = prefix;
  program->exact_prompt = exact;
  return true;
}

std::string JsonQuote(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"')
      result += '\\';
    if (c < 32) {
      result += "\\u00";
      result += digits[c >> 4];
      result += digits[c & 15];
    } else
      result += static_cast<char>(c);
  }
  return result + '"';
}

struct NativeOutput {
  int exit;
  std::string output, error;
};

// A real production text-policy fixture, not replacement hook implementations.
// Its tiny GGUF declares the actual tokenizer pieces used by this custom
// source; native modules and the entire callback binding remain owned by pinned
// Serve.
std::optional<NativeOutput> RunNativeOutput(const OutputFixture &fixture,
                                            std::string_view request,
                                            bool bytecode,
                                            int response_chunk_size = 0) {
  namespace fs = std::filesystem;
  const auto root =
      fs::path(Required("TEST_SRCDIR")) / Required("TEST_WORKSPACE");
  const auto checker = Required("LOOMIFY_TEXT_REQUEST_CHECK");
  if (checker.empty() || !fs::is_regular_file(checker)) {
    ADD_FAILURE() << "actual public text_request_check is required";
    return std::nullopt;
  }
  std::string pattern =
      (fs::path(Required("TEST_TMPDIR")) / "native-output-XXXXXX").string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back(0);
  const char *created = ::mkdtemp(writable.data());
  if (!created) {
    ADD_FAILURE() << "cannot create output scratch";
    return std::nullopt;
  }
  struct Cleanup {
    fs::path path;
    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  } scratch{created};
  auto write = [&](const char *name, std::string_view value) {
    const auto path = scratch.path / name;
    std::ofstream out(path, std::ios::binary);
    out.write(value.data(), value.size());
    return path.string();
  };
  auto read = [&](const char *name) {
    std::ifstream in(scratch.path / name, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };
  auto number = [](std::string &bytes, uint64_t value, int width) {
    for (int i = 0; i < width; ++i)
      bytes += static_cast<char>(value >> (i * 8));
  };
  auto string = [&](std::string &bytes, std::string_view value) {
    number(bytes, value.size(), 8);
    bytes += value;
  };
  const std::vector<std::string> pieces{"a",
                                        "<|fixture_bos|>",
                                        "<|fixture_end|>",
                                        "<|fixture_message|>",
                                        fixture.frame_suffix,
                                        fixture.header};
  std::string bytes = "GGUF";
  number(bytes, 3, 4);
  number(bytes, 0, 8);
  number(bytes, 7, 8);
  for (auto [key, value] : std::vector<std::pair<std::string, std::string>>{
           {"general.architecture", "llama"},
           {"tokenizer.ggml.model", "gpt2"},
           {"tokenizer.ggml.pre", "llama-bpe"}}) {
    string(bytes, key);
    number(bytes, 8, 4);
    string(bytes, value);
  }
  string(bytes, "tokenizer.ggml.tokens");
  number(bytes, 9, 4);
  number(bytes, 8, 4);
  number(bytes, pieces.size(), 8);
  for (const auto &piece : pieces)
    string(bytes, piece);
  string(bytes, "tokenizer.ggml.eos_token_id");
  number(bytes, 4, 4);
  number(bytes, 4, 4);
  string(bytes, "tokenizer.ggml.bos_token_id");
  number(bytes, 4, 4);
  number(bytes, 1, 4);
  string(bytes, "tokenizer.chat_template");
  number(bytes, 8, 4);
  string(bytes, fixture.original_source);
  while (bytes.size() % 32)
    bytes += '\0';
  std::unique_ptr<FILE, decltype(&std::fclose)> gguf(std::tmpfile(),
                                                     std::fclose);
  if (!gguf || std::setvbuf(gguf.get(), nullptr, _IONBF, 0) ||
      std::fwrite(bytes.data(), 1, bytes.size(), gguf.get()) != bytes.size()) {
    ADD_FAILURE() << "cannot create declared GGUF fixture";
    return std::nullopt;
  }
  std::string tokenizer = R"({"version":"1.0","model":{"type":"BPE","vocab":{)";
  for (size_t i = 0; i < pieces.size(); ++i) {
    if (i)
      tokenizer += ',';
    tokenizer += JsonQuote(pieces[i]) + ':' + std::to_string(i);
  }
  tokenizer += R"(},"merges":[]},"added_tokens":[)";
  for (size_t i = 1; i < pieces.size(); ++i) {
    if (i != 1)
      tokenizer += ',';
    tokenizer +=
        "{\"id\":" + std::to_string(i) +
        ",\"content\":" + JsonQuote(pieces[i]) +
        R"(,"special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false})";
  }
  tokenizer += "]}";
  model::GgufCatalog catalog;
  model::TextMetadata metadata;
  std::string error;
  if (!model::ReadGgufCatalog(gguf.get(), &catalog, &error) ||
      !model::ReadTextMetadata(gguf.get(), catalog, &metadata, &error)) {
    ADD_FAILURE() << error;
    return std::nullopt;
  }
  PreparedTextControl prepared;
  if (!PrepareGeneratedText({"output-fixture", {}, fixture.original_source},
                            gguf.get(), metadata, {}, tokenizer,
                            {root.string(), scratch.path.string(), bytecode},
                            &prepared, &error)) {
    ADD_FAILURE() << error;
    return std::nullopt;
  }
  auto compiler_ok = [&](loomc_status_t status) {
    const bool ok = loomc_status_is_ok(status);
    if (!ok) {
      const auto m = loomc_status_message(status);
      ADD_FAILURE() << std::string(m.data, m.size);
    }
    loomc_status_free(status);
    return ok;
  };
  struct Compiler {
    loomc_target_environment_t *target = nullptr;
    loomc_context_t *context = nullptr;
    loomc_workspace_t *workspace = nullptr;
    loomc_module_t *module = nullptr;
    loomc_source_t *source = nullptr;
    ~Compiler() {
      loomc_source_release(source);
      loomc_module_release(module);
      loomc_workspace_release(workspace);
      loomc_context_release(context);
      loomc_target_environment_release(target);
    }
  } compiler;
  if (!compiler_ok(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                      &compiler.target)))
    return std::nullopt;
  loomc_context_target_options_t target_options{};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.target_environment = compiler.target;
  loomc_context_options_t options{};
  options.next = &target_options;
  if (!compiler_ok(loomc_context_create(&options, loomc_allocator_system(),
                                        &compiler.context)) ||
      !compiler_ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                          &compiler.workspace)))
    return std::nullopt;
  auto helpers = prepared.helper_requirements;
  for (const char *path :
       {"lib/text/common/identity.loom", "lib/serving/greedy.loom",
        "lib/serving/request_check.loom"})
    helpers.push_back(path);
  for (auto &path : helpers)
    path = (root / path).string();
  const std::string configuration =
      prepared.source +
      "\nglobal.rodata.def public @loomify.text.model_name = bytes(\"" +
      Hex("output-fixture") + "\")\n";
  const std::vector<const char *> roots{
      "model_name",     "chat_reasoning", "chat_image_limits",
      "chat_begin",     "chat_message",   "chat_end",
      "parse_tools",    "prepare_input",  "output_next",
      "complete_text",  "render_tool",    "request_defaults",
      "request_options"};
  if (!LinkConfiguredProducts(compiler.context, compiler.workspace,
                              configuration, helpers, roots, false,
                              &compiler.module, &error)) {
    ADD_FAILURE() << error;
    return std::nullopt;
  }
  loomc_module_serialize_options_t serialize{};
  serialize.format =
      bytecode ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
  if (!compiler_ok(loomc_module_serialize_to_source(compiler.module, &serialize,
                                                    loomc_allocator_system(),
                                                    &compiler.source)))
    return std::nullopt;
  const auto contents = loomc_source_contents(compiler.source);
  const auto control =
      write("control.loom", {reinterpret_cast<const char *>(contents.data),
                             contents.data_length});
  std::string body(request);
  const auto close = body.find_last_of('}');
  if (close == std::string::npos) {
    ADD_FAILURE() << "native output request must be an object";
    return std::nullopt;
  }
  body.insert(close, ",\"model\":\"output-fixture\"");
  const auto request_path = write("request.json", body),
             tokenizer_path = write("tokenizer.json", tokenizer);
  const auto response_path = write("response.txt", fixture.response);
  const std::vector<std::string> arguments{
      checker,
      "--control=" + control,
      "--tokenizer=" + tokenizer_path,
      "--request=" + request_path,
      "--response=" + response_path,
      "--response_chunk_size=" + std::to_string(response_chunk_size)};
  std::vector<char *> argv;
  for (const auto &argument : arguments)
    argv.push_back(const_cast<char *>(argument.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = ::fork();
  if (pid < 0) {
    ADD_FAILURE() << "cannot start public checker";
    return std::nullopt;
  }
  if (!pid) {
    if (!std::freopen((scratch.path / "stdout").c_str(), "wb", stdout) ||
        !std::freopen((scratch.path / "stderr").c_str(), "wb", stderr))
      ::_exit(126);
    ::execv(checker.c_str(), argv.data());
    ::_exit(127);
  }
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) {
      ADD_FAILURE() << "waitpid failed";
      return std::nullopt;
    }
  return NativeOutput{WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                      read("stdout"), read("stderr")};
}

std::string JsonString(iree_string_view_t object, const char *key) {
  iree_string_view_t value{};
  auto status = iree_json_try_lookup_object_value(
      object, iree_make_cstring_view(key), &value);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  EXPECT_NE(value.data, nullptr) << key;
  std::string text(value.size, '\0');
  iree_host_size_t size = 0;
  status = iree_json_unescape_string(value, text.size(), text.data(), &size);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  text.resize(size);
  return text;
}

class OutputEmitterTest : public ::testing::TestWithParam<bool> {
protected:
  void Expect(const OutputFixture &fixture, std::string_view request,
              std::string_view expected) {
    if (!fixture.native_tools) {
      Expect(static_cast<const GeneratedInputRenderer &>(fixture), request,
             expected);
      return;
    }
    const auto result = RunNativeOutput(fixture, request, GetParam());
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->exit, 0) << result->error;
    const auto view = [](std::string_view text) {
      return iree_make_string_view(text.data(), text.size());
    };
    const auto actual = view(result->output), wanted = view(expected);
    EXPECT_EQ(JsonString(actual, "content"), JsonString(wanted, "content"));
    EXPECT_EQ(JsonString(actual, "reasoning_content"),
              JsonString(wanted, "reasoning"));
    auto calls = [&](iree_string_view_t object, const char *key) {
      iree_string_view_t list{};
      auto status = iree_json_try_lookup_object_value(
          object, iree_make_cstring_view(key), &list);
      EXPECT_TRUE(iree_status_is_ok(status));
      iree_status_ignore(status);
      std::vector<iree_string_view_t> values;
      status = iree_json_enumerate_array_typed(
          list,
          [](void *data, iree_host_size_t, iree_json_value_type_t type,
             iree_string_view_t value) {
            if (type != IREE_JSON_VALUE_TYPE_OBJECT)
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
            static_cast<std::vector<iree_string_view_t> *>(data)->push_back(
                value);
            return iree_ok_status();
          },
          &values);
      EXPECT_TRUE(iree_status_is_ok(status));
      iree_status_ignore(status);
      std::vector<std::pair<std::string, std::string>> records;
      for (auto value : values) {
        EXPECT_EQ(JsonString(value, "type"), "function");
        iree_string_view_t function{};
        status = iree_json_try_lookup_object_value(value, IREE_SV("function"),
                                                   &function);
        EXPECT_TRUE(iree_status_is_ok(status));
        iree_status_ignore(status);
        records.emplace_back(JsonString(function, "name"),
                             JsonString(function, "arguments"));
      }
      return records;
    };
    // Native owns assigned IDs. These parser/source witnesses compare the
    // ordered function names and exact argument strings, not generated IDs.
    EXPECT_EQ(calls(actual, "tool_calls"), calls(wanted, "calls"));
    if (!fixture.prompt_suffix.empty()) {
      const auto prompt = JsonString(actual, "prompt");
      ASSERT_GE(prompt.size(), fixture.prompt_suffix.size());
      EXPECT_EQ(prompt.substr(prompt.size() - fixture.prompt_suffix.size()),
                fixture.prompt_suffix);
      if (fixture.exact_prompt)
        EXPECT_EQ(prompt, fixture.prompt_suffix);
    }
  }

  void ExpectRejected(const OutputFixture &fixture, std::string_view request) {
    if (fixture.native_tools) {
      const auto result = RunNativeOutput(fixture, request, GetParam());
      ASSERT_TRUE(result.has_value());
      EXPECT_EQ(result->exit, 1) << result->error;
      EXPECT_TRUE(result->output.empty()) << result->output;
      EXPECT_FALSE(result->error.empty());
      return;
    }
    const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                       Required("TEST_WORKSPACE"))
                          .string();
    std::unique_ptr<PreparedTemplate> prepared;
    std::string error, result = "unchanged";
    ASSERT_TRUE(PreparedTemplate::Prepare(
        fixture, root, Required("TEST_TMPDIR"), GetParam(), &prepared, &error))
        << error;
    EXPECT_FALSE(prepared->Render(request, &result, &error));
    EXPECT_EQ(result, "unchanged");
  }

  void Expect(const GeneratedInputRenderer &product, std::string_view response,
              std::string_view expected) {
    std::unique_ptr<PreparedTemplate> prepared;
    std::string error;
    const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                       Required("TEST_WORKSPACE"))
                          .string();
    ASSERT_TRUE(PreparedTemplate::Prepare(
        product, root, Required("TEST_TMPDIR"), GetParam(), &prepared, &error))
        << error;
    Expect(*prepared, response, expected);
  }

  void Expect(PreparedTemplate &prepared, std::string_view response,
              std::string_view expected) {
    std::string error, result;
    ASSERT_TRUE(prepared.Render(response, &result, &error)) << error;
    EXPECT_EQ(result, expected);
  }
};

TEST(OutputEmitterFacts, RejectsMalformedJsonWireWithoutChangingOutput) {
  TextFormatAnalysis analysis;
  analysis.alternatives.emplace_back().tools.kind = TextToolKind::JsonEnvelope;
  GeneratedInputRenderer output;
  output.source = "preserved output";
  output.entrypoint = "preserved.entrypoint";
  Diagnostic diagnostic;
  for (int defect : {0, 1, 2}) {
    auto &tools = analysis.alternatives[0].tools;
    tools.kind = TextToolKind::JsonEnvelope;
    tools.json_argument_wire.reset();
    if (defect == 1) {
      tools.kind = TextToolKind::None;
      tools.json_argument_wire = TextJsonArgumentWire::RawJson;
    } else if (defect == 2) {
      tools.json_argument_wire = static_cast<TextJsonArgumentWire>(99);
    }
    EXPECT_FALSE(EmitOutputParser("malformed-facts.jinja", analysis, &output,
                                  &diagnostic));
    EXPECT_EQ(diagnostic.message,
              defect == 2 ? "invalid finalized JSON argument wire"
                          : "output emission needs finalized JSON wire facts");
    EXPECT_EQ(output.source, "preserved output");
    EXPECT_EQ(output.entrypoint, "preserved.entrypoint");
  }
}

TEST_P(OutputEmitterTest, BoundedMarkersUtf8AndTerminalToolSuffix) {
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 1, 0, 1, 7, 1),
         "Answer <inv", "Answer ");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 7, 1, 0, 0, 7, 1),
         "Answer <inv", "");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 7, 1, 0, 0, 7, 1),
         "Answer <invoke>{}", "");
  Expect(Scanner("", "", "", 0, 1, 0, 1, 1, 1), std::string("a\xe2\x82", 3),
         "a");
  Expect(Scanner("", "", "", 1, 1, 0, 0, 1, 1), std::string("a\xe2\x82", 3),
         "");
  Expect(Scanner("", "", "", 1, 1, 0, 1, 4, 1), "a\xe2\x82\xac",
         "\xe2\x82\xac");
  Expect(Scanner("", "", "", 1, 1, 1, 1, 3, 1), std::string("a\xe2\x82", 3),
         std::string("\xe2\x82", 2));
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 1, 1, 1, 4, 1), "<inv",
         "<inv");
}

TEST_P(OutputEmitterTest, OptionalOpenRequiresExplicitReasoningFrame) {
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 3, 0, 1, 5, 1),
         "plain", "plain");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 3, 0, 1, 5, 1),
         "<reaX", "<reaX");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 3, 1, 1, 4, 1), "<rea",
         "<rea");
  // An already-open reasoning prefix remains reasoning without a new marker.
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 2, 0, 2, 5, 2),
         "plain", "plain");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 3, 0, 0, 0, 3), "<rea",
         "");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 3, 0, 0, 8, 4),
         "<reason>\nbody", "<reason>");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 8, 4, 0, 0, 9, 260),
         "<reason>\nbody", "\n");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 9, 260, 0, 2, 13, 2),
         "<reason>\nbody", "body");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 2, 0, 0, 9, 5),
         "</reason>\n\ntext", "</reason>");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 9, 5, 0, 0, 10, 261),
         "</reason>\n\ntext", "\n");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 10, 261, 0, 0, 11, 517),
         "</reason>\n\ntext", "\n");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 11, 517, 0, 1, 15, 1),
         "</reason>\n\ntext", "text");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 261, 0, 1, 1, 1), "x",
         "x");
  // No-progress preserves framing progress and the generated format identity.
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 4294967557, 0, 0, 0,
                 4294967557),
         "<inv", "");
  // Both immutable completion flags survive advancement and no-progress.
  Expect(Scanner("", "", "", 0, 25, 0, 1, 5, 25), "plain", "plain");
  Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, 4294967581, 0, 0, 0,
                 4294967581),
         "<inv", "");
}

TEST_P(OutputEmitterTest, EveryMarkerAndUtf8SplitPreservesCursorAndFinalTail) {
  struct Marker {
    std::string_view bytes;
    int64_t initial, visible, final_state, complete_end, complete_state;
  };
  for (const auto &marker :
       {Marker{"<reason>", 3, 1, 1, 8, 4}, Marker{"</reason>", 2, 2, 2, 9, 5},
        Marker{"<invoke>", 1, 1, 1, 0, 1}}) {
    for (size_t size = 1; size < marker.bytes.size(); ++size) {
      SCOPED_TRACE(std::string(marker.bytes) +
                   " split=" + std::to_string(size));
      const auto prefix = marker.bytes.substr(0, size);
      Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, marker.initial, 0,
                     0, 0, marker.initial),
             prefix, "");
      Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, marker.initial, 1,
                     marker.visible, size, marker.final_state),
             prefix, prefix);
    }
    // The growing buffer can now complete the held marker using the identical
    // cursor/state. A complete tool opener still belongs to parse_tools.
    Expect(Scanner("<reason>", "</reason>", "<invoke>", 0, marker.initial, 0, 0,
                   marker.complete_end, marker.complete_state),
           marker.bytes, marker.bytes.substr(0, marker.complete_end));
  }
  for (std::string_view codepoint :
       {std::string_view("\xc3\xa9"), std::string_view("\xe2\x82\xac"),
        std::string_view("\xf0\x9f\x98\x80")}) {
    for (size_t size = 1; size < codepoint.size(); ++size) {
      SCOPED_TRACE("UTF8 width=" + std::to_string(codepoint.size()) +
                   " split=" + std::to_string(size));
      const auto tail = codepoint.substr(0, size);
      const std::string response = "a" + std::string(tail);
      Expect(Scanner("", "", "", 1, 1, 0, 0, 1, 1), response, "");
      Expect(Scanner("", "", "", 1, 1, 1, 1, response.size(), 1), response,
             tail);
    }
    Expect(Scanner("", "", "", 1, 1, 0, 1, 1 + codepoint.size(), 1),
           "a" + std::string(codepoint), codepoint);
  }
}

TEST_P(OutputEmitterTest, NeutralPrefixPreservesExplicitFramingAndCheckpoint) {
  // A neutral generation prefix must not invent reasoning around plain output.
  // The same source can still render explicitly framed assistant reasoning.
  constexpr std::string_view source =
      "{% for m in messages %}{% if m.role == 'assistant' %}{{ '<reply>' }}"
      "{% if m.reasoning_content %}{{ '<reason>' + m.reasoning_content + '</reason>' }}{% endif %}"
      "{{ m.content }}{{ '</reply>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<reply>' }}{% endif %}";
  constexpr std::string_view request =
      R"({"messages":[{"role":"user","content":"a"}]})";
  struct Case {
    const char *response;
    const char *content;
    const char *reasoning;
    const char *checkpoint;
  };
  for (const auto &test : {Case{"plain", "plain", "", "<reply>plain</reply>"},
                           Case{"<reason>why</reason>plain", "plain", "why",
                                "<reply><reason>why</reason>plain</reply>"},
                           Case{"<reaX", "<reaX", "", "<reply><reaX</reply>"},
                           Case{"<rea", "<rea", "", "<reply><rea</reply>"}}) {
    SCOPED_TRACE(test.response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, test.response, &program, &error))
        << error;
    for (int chunk_size : {0, 1}) {
      SCOPED_TRACE(chunk_size);
      const auto result =
          RunNativeOutput(program, request, GetParam(), chunk_size);
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result->exit, 0) << result->error;
      const auto report =
          iree_make_string_view(result->output.data(), result->output.size());
      EXPECT_EQ(JsonString(report, "prompt"), "<reply>");
      EXPECT_EQ(JsonString(report, "content"), test.content);
      EXPECT_EQ(JsonString(report, "reasoning_content"), test.reasoning);
      EXPECT_EQ(JsonString(report, "checkpoint"), test.checkpoint);
    }
  }
}

TEST_P(OutputEmitterTest, InlineReasoningTransformsPreserveCanonicalReplay) {
  for (bool macro_content : {false, true}) {
    SCOPED_TRACE(macro_content ? "macro content" : "direct content");
    const std::string source =
        std::string(
            macro_content
                ? "{% macro normalize(value) %}{{ value }}{% endmacro %}"
                : "") +
        "{% for m in messages %}{% if m.role == 'assistant' %}"
        "{% set content = " +
        (macro_content ? "normalize(m.content)" : "m.content") +
        " | trim %}{% set reason = '' %}"
        "{% if m.reasoning_content is string %}"
        "{% set reason = m.reasoning_content %}"
        "{% elif '</think>' in content %}"
        "{% set reason = content.split('</think>')[0].rstrip('\\n')"
        ".split('<think>')[-1].lstrip('\\n') %}"
        "{% set content = content.split('</think>')[-1].lstrip('\\n') %}"
        "{% endif %}"
        "{{ '<reply><think>' + reason + '</think>' + content + '</reply>' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<reply><think>' }}"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "{{ '</think>' }}{% endif %}{% endif %}";
    struct Case {
      bool thinking;
      const char *response;
      const char *content;
      const char *reasoning;
      const char *checkpoint;
    };
    const Case cases[] = {
        {false, "<think>\nINLINE_WHY\n</think>\n\nINLINE_ANSWER",
         "<think>\nINLINE_WHY\n</think>\n\nINLINE_ANSWER", "",
         "<reply><think>INLINE_WHY</think>INLINE_ANSWER</reply>"},
        {false, "PLAIN_ANSWER", "PLAIN_ANSWER", "",
         "<reply><think></think>PLAIN_ANSWER</reply>"},
        {true, "EXPLICIT_WHY</think><think>BODY_LITERAL</think>FINAL_ANSWER",
         "<think>BODY_LITERAL</think>FINAL_ANSWER", "EXPLICIT_WHY",
         "<reply><think>EXPLICIT_WHY</think><think>BODY_LITERAL</think>FINAL_ANSWER</reply>"},
    };
    for (const auto &test : cases) {
      SCOPED_TRACE(test.response);
      OutputFixture program;
      std::string error;
      ASSERT_TRUE(OutputProgram(source, test.response, &program, &error))
          << error;
      const auto request =
          std::string(
              "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],") +
          "\"enable_thinking\":" + (test.thinking ? "true}" : "false}");
      const auto result = RunNativeOutput(program, request, GetParam());
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result->exit, 0) << result->error;
      const auto report =
          iree_make_string_view(result->output.data(), result->output.size());
      EXPECT_EQ(JsonString(report, "content"), test.content);
      EXPECT_EQ(JsonString(report, "reasoning_content"), test.reasoning);
      EXPECT_EQ(JsonString(report, "checkpoint"), test.checkpoint);
    }
  }
}

TEST_P(OutputEmitterTest, SourceDerivedJsonEnvelopeAndMemberNames) {
  constexpr std::string_view source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<turn>assistant:' }}{{ m.content }}"
      "{% if m.tool_calls %}{% for c in m.tool_calls %}"
      "{{ '<invoke>{\"operation\":' }}{{ c.function.name | tojson }}"
      "{{ ',\"inputs\":' }}{{ c.function.arguments if c.function.arguments is string else c.function.arguments | tojson }}"
      "{{ '}</invoke>' }}{% endfor %}{% endif %}"
      "{{ '</turn>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
  TextFormatAnalysis analysis;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "changed-wire.jinja", {}, &analysis,
                                &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(analysis.alternatives.empty());
  // Generation-prefix presence creates distinct request routes with the same
  // source-derived call grammar. Do not conflate routes with tool grammars.
  const auto &tools = analysis.alternatives[0].tools;
  ASSERT_EQ(tools.kind, TextToolKind::JsonEnvelope);
  EXPECT_EQ(tools.open, "<invoke>");
  EXPECT_EQ(tools.close, "</invoke>");
  EXPECT_EQ(tools.name_field, "operation");
  EXPECT_EQ(tools.arguments_field, "inputs");
  auto product = JsonParser(tools);
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error, result = "unchanged";
  const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                     Required("TEST_WORKSPACE"))
                        .string();
  ASSERT_TRUE(PreparedTemplate::Prepare(product, root, Required("TEST_TMPDIR"),
                                        GetParam(), &prepared, &error))
      << error;
  Expect(
      *prepared,
      R"(<invoke>{"operation":"lookup","inputs":{"x":1.2500e+02,"s":"</invoke>"}}</invoke>)",
      R"([{"type":"function","function":{"name":"lookup","arguments":"{\"x\":1.2500e+02,\"s\":\"</invoke>\"}"}}])");
  Expect(
      *prepared,
      R"(<invoke>{"operation":"lookup","inputs":{"s":"a\"</invoke>b"}}</invoke>)",
      R"([{"type":"function","function":{"name":"lookup","arguments":"{\"s\":\"a\\\"</invoke>b\"}"}}])");
  Expect(*prepared, "<invoke>{", "[]");
  Expect(*prepared, "", "[]");
  Expect(
      *prepared,
      R"(<invoke>{"operation":"lookup","inputs":{}}</invoke><invoke>{"operation":"lookup","inputs":{"x":2}}</invoke>)",
      R"([{"type":"function","function":{"name":"lookup","arguments":"{}"}},{"type":"function","function":{"name":"lookup","arguments":"{\"x\":2}"}}])");
  Expect(
      *prepared, R"(<invoke>{"operation":"lookup","inputs":"{}"}</invoke>)",
      R"([{"type":"function","function":{"name":"lookup","arguments":"\"{}\""}}])");
  for (
      const auto *bad :
      {R"(<invoke>{"name":"lookup","arguments":{}}</invoke>)",
       R"(<invoke>{"operation":"unknown","inputs":{}}</invoke>)",
       R"(<invoke>{"operation":"lookup","inputs":{}}</invoke><invoke>{bad}</invoke>)",
       R"(<invoke>{"operation":"lookup","inputs":{"x":1,"x":2}}</invoke>)",
       R"(<invoke>{"operation":"lookup","inputs":not JSON}</invoke>)"}) {
    EXPECT_FALSE(prepared->Render(bad, &result, &error)) << bad;
    EXPECT_EQ(result, "unchanged");
  }
}

TEST(OutputEmitterFacts, RequestDependentParserNeedsUnavailableFixedHookInput) {
  TextFormatAnalysis analysis;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(kConditionalSource, "request-grammar.jinja", {},
                                &analysis, &diagnostic))
      << diagnostic.message;
  OutputFixture program;
  program.source = "preserved output";
  program.entrypoint = "preserved.entrypoint";
  EXPECT_FALSE(EmitOutputParser("request-grammar.jinja", analysis, &program,
                                &diagnostic));
  EXPECT_NE(
      diagnostic.message.find("fixed a401 parse_tools has no request or state"),
      std::string::npos);
  EXPECT_EQ(diagnostic.source_name, "request-grammar.jinja");
  EXPECT_EQ(program.source, "preserved output");
  EXPECT_EQ(program.entrypoint, "preserved.entrypoint");
}

TEST_P(OutputEmitterTest,
       JsonArgumentCapturePreservesNativeStringPayloadExactly) {
  struct Fixture {
    const char *response;
    const char *expected;
  };
  const Fixture fixtures[] = {
      {R"(<invoke>{"name":"查找-name.点","arguments":"not JSON: </invoke>\n  "}</invoke>)",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"not JSON: </invoke>\n  "}}]})"},
      {R"(<invoke>{"name":"查找-name.点","arguments":"{ \"x\" : 1.2500e+02 }"}</invoke>)",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{ \"x\" : 1.2500e+02 }"}}]})"},
      {R"(<invoke>{"name":"查找-name.点","arguments":""}</invoke>)",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":""}}]})"},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(
        OutputProgram(kFixedFieldsSource, fixture.response, &program, &error))
        << error;
    Expect(program, kUnicodeRequest, fixture.expected);
  }
}

TEST_P(OutputEmitterTest, StringElseJsonPreservesTheRawJsonFieldQuoteLevel) {
  std::string source(kFixedFieldsSource);
  const std::string original = "c.function.arguments | tojson";
  const std::string replacement =
      "c.function.arguments if c.function.arguments is string else c.function.arguments | tojson";
  size_t at = 0;
  while ((at = source.find(original, at)) != std::string::npos) {
    source.replace(at, original.size(), replacement);
    at += replacement.size();
  }
  OutputFixture object_program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      R"(<invoke>{"name":"查找-name.点","arguments":{"text":"hello"}}</invoke>)",
      &object_program, &error))
      << error;
  Expect(
      object_program, kUnicodeRequest,
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"hello\"}"}}]})");

  struct Fixture {
    const char *wire_argument;
    const char *field;
    const char *native_argument;
  };
  const Fixture fixtures[] = {
      {R"("5")", "5", R"("5")"},
      {R"("[1, 2]")", "[1, 2]", R"("[1,2]")"},
      {R"("1.2500e+02")", "1.2500e+02", R"("1.2500e+02")"},
      {R"("\"x\"")", R"("x")", R"("\"x\"")"},
      {R"("\"\\u0078\"")", R"("\u0078")", R"("\"\\u0078\"")"},
      {R"("\"{}\"")", R"("{}")", R"("\"{}\"")"},
      {R"("\"not JSON\"")", R"("not JSON")", R"("\"not JSON\"")"},
      {R"("\"\"")", R"("")", R"("\"\"")"},
      {R"("true")", "true", R"("true")"},
      {R"("null")", "null", R"("null")"},
  };
  const std::string request_begin =
      R"({"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":)";
  const std::string request_end =
      R"(}}]}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]})";
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.field);
    const std::string response = "<invoke>{\"name\":\"lookup\",\"arguments\":" +
                                 std::string(fixture.field) + "}</invoke>";
    const std::string request =
        request_begin + fixture.wire_argument + request_end;
    OutputFixture program;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    // Native history's argument is STRING. The unmodified source emits that
    // STRING verbatim as a JSON field; its quotes are payload when applicable.
    const std::string prompt =
        "<turn>assistant:" + response + "</turn><turn>assistant:";
    ASSERT_TRUE(CheckPromptPrefix(source, prompt, &program, &error, true))
        << error;
    const std::string expected =
        R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"lookup","arguments":)" +
        std::string(fixture.native_argument) + "}}]}";
    Expect(program, request, expected);
  }

  // An unquoted opaque argument makes the actual raw source field malformed;
  // keep normal native JSON failure instead of inventing a fallback decoder.
  const std::string response =
      "<invoke>{\"name\":\"lookup\",\"arguments\":not JSON}</invoke>";
  OutputFixture invalid;
  ASSERT_TRUE(OutputProgram(source, response, &invalid, &error)) << error;
  ASSERT_TRUE(CheckPromptPrefix(
      source, "<turn>assistant:" + response + "</turn><turn>assistant:",
      &invalid, &error, true))
      << error;
  const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                     Required("TEST_WORKSPACE"))
                        .string();
  std::unique_ptr<PreparedTemplate> prepared;
  ASSERT_TRUE(PreparedTemplate::Prepare(invalid, root, Required("TEST_TMPDIR"),
                                        GetParam(), &prepared, &error))
      << error;
  std::string result = "unchanged";
  EXPECT_FALSE(prepared->Render(request_begin + R"("not JSON")" + request_end,
                                &result, &error));
  EXPECT_EQ(result, "unchanged");
}

TEST_P(OutputEmitterTest, ActualLlamaPlainJsonRemainsContentWithoutToolMode) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/llama/templates/5816fce1.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  TextFormatBindings bindings;
  bindings.strings["bos_token"] = "";
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(source, R"({"name":"lookup","parameters":{"x":1}})",
                            &program, &error, bindings))
      << error;
  std::unique_ptr<PreparedTemplate> prepared;
  ASSERT_TRUE(PreparedTemplate::Prepare(program, root.string(),
                                        Required("TEST_TMPDIR"), GetParam(),
                                        &prepared, &error))
      << error;
  constexpr std::string_view expected =
      R"({"content":"{\"name\":\"lookup\",\"parameters\":{\"x\":1}}","reasoning":"","calls":[]})";
  for (const auto *request :
       {R"({"messages":[{"role":"user","content":"question"}]})",
        R"({"messages":[{"role":"user","content":"question"}],"tools":null})",
        R"({"messages":[{"role":"user","content":"question"}],"tools":[]})"}) {
    std::string result;
    ASSERT_TRUE(prepared->Render(request, &result, &error)) << error;
    EXPECT_EQ(result, expected);
  }
  std::string result = "unchanged";
  EXPECT_FALSE(prepared->Render(
      R"({"messages":[{"role":"user","content":"question"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]})",
      &result, &error));
  EXPECT_EQ(result, "unchanged");
  ASSERT_TRUE(
      prepared->Render(R"({"messages":[{"role":"user","content":"question"}]})",
                       &result, &error))
      << error;
  EXPECT_EQ(result, expected);
}

// Tagged callbacks execute through the actual pinned Serve checker, which owns
// its private tools module. Source-only fixtures use the production text
// producer.
std::string TaggedSource(bool named, std::string_view value) {
  return "{% for m in messages %}{% if m.role == 'assistant' %}"
         "{{ '<turn>assistant:' }}{{ m.content }}"
         "{% if m.tool_calls %}{% for c in m.tool_calls %}" +
         std::string(named ? "{{ '<run><action=' + c.function.name + '>' }}"
                           : "{{ '<run>' + c.function.name }}") +
         "{% for k,v in c.function.arguments.items() %}" +
         (named ? "{{ '<input=' + k + '>\\n' }}"
                : "{{ '<key>' + k + '</key><value>' }}") +
         std::string(value) +
         (named ? "{{ '\\n</input>' }}" : "{{ '</value>' }}") + "{% endfor %}" +
         (named ? "{{ '</action></run>' }}" : "{{ '</run>' }}") +
         "{% endfor %}{% endif %}{{ '</turn>' }}{% endif %}{% endfor %}"
         "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
}

TEST_P(OutputEmitterTest, TaggedNameTransformCannotReachOutputEmission) {
  auto source = TaggedSource(true, "{{ v }}");
  const auto name = source.find("c.function.name");
  ASSERT_NE(name, std::string::npos);
  source.replace(name, std::string("c.function.name").size(),
                 "(c.function.name | tojson)");
  OutputFixture program;
  program.source = "preserved output";
  program.entrypoint = "preserved.entrypoint";
  std::string error;
  EXPECT_FALSE(OutputProgram(source, "", &program, &error));
  EXPECT_EQ(error, "tagged names require a direct source capture");
  EXPECT_EQ(program.source, "preserved output");
  EXPECT_EQ(program.entrypoint, "preserved.entrypoint");
}

TEST_P(OutputEmitterTest,
       TaggedNamedSourceMutationsPreserveUnicodeAndScalarKinds) {
  const auto source = TaggedSource(
      true,
      "{{ v | tojson if v is mapping or (v is sequence and v is not string) else v | string }}");
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      "<run><action=查找-name.点><input=启用-flag.位>\nTrue\n</input>"
      "<input=空值>\nNone\n</input><input=text>\nNone\n</input>"
      "<input=lines>\n\nline\n\n</input>"
      "<input=opaque>\n</function><tool_call>\n</input></action></run>",
      &program, &error))
      << error;
  Expect(
      program, kUnicodeRequest,
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"启用-flag.位\":true,\"空值\":null,\"text\":\"None\",\"lines\":\"\\nline\\n\",\"opaque\":\"</function><tool_call>\"}"}}]})");
}

TEST_P(OutputEmitterTest,
       TaggedKeyValueSourceMutationRepeatsAndEmptyArguments) {
  const auto source =
      TaggedSource(false, "{{ v if v is string else v | tojson }}");
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      "<run>查找-name.点<key>lines</key><value>\nline\n</value></run>"
      "<run>查找-name.点</run>",
      &program, &error))
      << error;
  Expect(
      program, kUnicodeRequest,
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"lines\":\"\\nline\\n\"}"}},{"type":"function","function":{"name":"查找-name.点","arguments":"{}"}}]})");
}

TEST_P(OutputEmitterTest,
       CompactCanonicalLiteralsDoNotEnablePythonScalarRewrite) {
  const auto source = TaggedSource(
      true,
      "{% if v is string %}{{ v }}"
      "{% elif v is boolean %}{{ 'true' if v else 'false' }}"
      "{% elif v is none %}{{ 'null' }}"
      "{% elif v is mapping or (v is sequence and v is not string) %}{{ v | tojson }}"
      "{% else %}{{ v }}{% endif %}");
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      "<run><action=查找-name.点><input=启用-flag.位>\nTrue\n</input>"
      "<input=空值>\nNone\n</input></action></run>",
      &program, &error))
      << error;
  // Native XML represents these non-JSON bytes as strings. The source's
  // canonical literal codec must not rewrite them into Boolean/null values.
  Expect(
      program, kUnicodeRequest,
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"启用-flag.位\":\"True\",\"空值\":\"None\"}"}}]})");
}

TEST_P(OutputEmitterTest,
       TaggedJsonCaptureDecodesOnlyQualifiedStringRepresentation) {
  const auto source = TaggedSource(true, "{{ v | tojson }}");
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      "<run><action=查找-name.点><input=text>\n\"literal\\ntext\"\n</input>"
      "<input=空值>\nnull\n</input></action></run>",
      &program, &error))
      << error;
  Expect(
      program, kUnicodeRequest,
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"literal\\ntext\",\"空值\":null}"}}]})");
}

TEST_P(OutputEmitterTest, TaggedMalformedSuffixNeverPublishesPartialCalls) {
  const auto source =
      TaggedSource(true, "{{ v if v is string else v | tojson }}");
  for (
      const auto *response :
      {"<run><action=查找-name.点><input=text>missing</action></run>",
       "<run><action=查找-name.点></action></run><run><action=查找-name.点>",
       "<run><action=查找-name.点><input=text>\n</parameter>\n</input></action></run>",
       "<run><action=查找-name.点></action></run>late visible content"}) {
    SCOPED_TRACE(response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    ExpectRejected(program, kUnicodeRequest);
  }
}

TEST_P(OutputEmitterTest, ActualTaggedSourcesLowerWithoutProfileConstants) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  for (
      const auto &[path, response] :
      {std::pair<const char *, const char *>{
           "lib/text/qwen35/templates/7f0e5290.jinja",
           "<tool_call><function=查找-name.点><parameter=空值>\nNone\n</parameter>"
           "<parameter=text>\nNone\n</parameter></function></tool_call>"},
       {"lib/text/qwen35/templates/c3cf9e34.jinja",
        "<tool_call><function=查找-name.点><parameter=lines>\n\nline\n\n</parameter></function></tool_call>"},
       {"lib/text/compact_chatml/templates/f45005c5.jinja",
        "<tool_call><function=查找-name.点><parameter=空值>\nnull\n</parameter></function></tool_call>"},
       {"lib/text/spark/templates/93d6aa90.jinja",
        "<tool_call>查找-name.点<arg_key>空值</arg_key><arg_value>null</arg_value></tool_call>"}}) {
    SCOPED_TRACE(path);
    std::ifstream file(root / path);
    ASSERT_TRUE(file.good()) << path;
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    OutputFixture program;
    std::string error;
    TextFormatBindings bindings;
    bindings.strings["bos_token"] = "";
    ASSERT_TRUE(OutputProgram(source, response, &program, &error, bindings))
        << error;
    const auto expected =
        std::string(path).find("7f0e5290") != std::string::npos
            ? R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"空值\":null,\"text\":\"None\"}"}}]})"
        : std::string(path).find("c3cf9e34") != std::string::npos
            ? R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"lines\":\"\\nline\\n\"}"}}]})"
            : R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"空值\":null}"}}]})";
    Expect(program, kUnicodeRequest, expected);
  }
}

TEST_P(OutputEmitterTest,
       RemainingLiteralAndJsonSourcesUseActualSelectedChannels) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  constexpr std::string_view plain_request =
      R"({"messages":[{"role":"user","content":"question"}]})";
  struct Fixture {
    const char *path;
    const char *response;
    std::string_view request;
    const char *expected;
    const char *prefix = nullptr;
  };
  const Fixture fixtures[] = {
      {"lib/text/plain_chatml/templates/58c1a1f0.jinja",
       "<think>literal</think><tool_call>literal</tool_call>", plain_request,
       R"({"content":"<think>literal</think><tool_call>literal</tool_call>","reasoning":"","calls":[]})"},
      {"lib/text/gemma/templates/7de1c58e.jinja",
       "<think>literal</think><tool_call>literal</tool_call>", plain_request,
       R"({"content":"<think>literal</think><tool_call>literal</tool_call>","reasoning":"","calls":[]})"},
      {"lib/text/qwen25/templates/cd8e9439.jinja",
       R"(answer<tool_call>{"name":"查找-name.点","arguments":{"text":"hello"}}</tool_call>)",
       kUnicodeRequest,
       R"({"content":"answer","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"hello\"}"}}]})"},
      {"lib/text/qwen3/templates/5da44855.jinja",
       R"(answer<tool_call>{"name":"查找-name.点","arguments":{"text":"hello"}}</tool_call>)",
       kUnicodeRequest,
       R"({"content":"answer","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"hello\"}"}}]})",
       "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.path);
    std::ifstream file(root / fixture.path);
    ASSERT_TRUE(file.good());
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, fixture.response, &program, &error))
        << error;
    if (fixture.prefix) {
      ASSERT_TRUE(CheckPromptPrefix(source, fixture.prefix, &program, &error))
          << error;
    }
    Expect(program, fixture.request, fixture.expected);
  }
}

TEST_P(OutputEmitterTest,
       ActualSourcePrefixesAndResponseOpenersHaveDistinctStartup) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/qwen3/templates/5da44855.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  std::string request(kUnicodeRequest);
  const std::string disabled = "\"enable_thinking\":false";
  const auto option = request.find(disabled);
  ASSERT_NE(option, std::string::npos);
  request.replace(option, disabled.size(), "\"enable_thinking\":true");
  // This actual source's thinking=true generation suffix is neutral. Only an
  // opening marker in the response establishes reasoning; a closing marker
  // cannot retroactively turn the preceding unframed bytes into reasoning.
  for (
      const auto &[response, expected] :
      {std::pair<const char *, const char *>{
           "<think>\nreason</think>\n\nanswer<tool_call>{\"name\":\"查找-name.点\",\"arguments\":{\"text\":\"hello\"}}</tool_call>",
           R"({"content":"answer","reasoning":"reason","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"hello\"}"}}]})"},
       {"reason</think>\n\nanswer<tool_call>{\"name\":\"查找-name.点\",\"arguments\":{\"text\":\"hello\"}}</tool_call>",
        R"({"content":"reason</think>\n\nanswer","reasoning":"","calls":[{"type":"function","function":{"name":"查找-name.点","arguments":"{\"text\":\"hello\"}"}}]})"}}) {
    SCOPED_TRACE(response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    ASSERT_TRUE(
        CheckPromptPrefix(source, "<|im_start|>assistant\n", &program, &error))
        << error;
    Expect(program, request, expected);
  }
  // A different actual source opens reasoning in its prompt. A close-only
  // response must still close that already-open frame and expose final text.
  std::ifstream opened_file(root / "lib/text/spark/templates/93d6aa90.jinja");
  ASSERT_TRUE(opened_file.good());
  const std::string opened_source((std::istreambuf_iterator<char>(opened_file)),
                                  {});
  OutputFixture opened_program;
  std::string error;
  ASSERT_TRUE(OutputProgram(opened_source, "reason</think>answer",
                            &opened_program, &error))
      << error;
  ASSERT_TRUE(CheckPromptPrefix(opened_source,
                                "<｜start▁of▁sentence｜><|Bot|><think>",
                                &opened_program, &error))
      << error;
  Expect(opened_program, request,
         R"({"content":"answer","reasoning":"reason","calls":[]})");
}

TEST_P(OutputEmitterTest,
       ActualJsonSourcesPreserveRepeatedCallsAndIncompleteTail) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  constexpr std::string_view request =
      R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]})";
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{"text":"é </tool_call>","nested":{"ok":true},"n":1}}</tool_call>)";
  const std::string record =
      R"({"type":"function","function":{"name":"f","arguments":"{\"text\":\"é </tool_call>\",\"nested\":{\"ok\":true},\"n\":1}"}})";
  for (const auto *path : {"lib/text/qwen25/templates/cd8e9439.jinja",
                           "lib/text/qwen3/templates/5da44855.jinja"}) {
    SCOPED_TRACE(path);
    std::ifstream file(root / path);
    ASSERT_TRUE(file.good());
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    for (const auto &[response, calls] :
         {std::pair<std::string, std::string>{
              call + "\n" + call, "[" + record + "," + record + "]"},
          {"<tool_call>{\"name\":\"f\"", "[]"},
          {call + "\n<tool_call>{\"name\":\"f\"", "[" + record + "]"}}) {
      SCOPED_TRACE(response);
      OutputFixture program;
      std::string error;
      ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
      Expect(program, request,
             "{\"content\":\"\",\"reasoning\":\"\",\"calls\":" + calls + "}");
    }

    // Native history arguments is STRING. The two unmodified sources spell
    // that payload differently: Qwen25 tojson quotes it, Qwen3 emits it raw.
    constexpr std::string_view history =
        R"({"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":"[]"}}]}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]})";
    const bool raw =
        std::string_view(path).find("qwen3/") != std::string_view::npos;
    const std::string response =
        raw ? "<tool_call>\n{\"name\": \"f\", \"arguments\": []}\n</tool_call>"
            : "<tool_call>\n{\"name\": \"f\", \"arguments\": \"[]\"}\n</tool_call>";
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    const std::string suffix = response +
                               "<|im_end|>\n<|im_start|>assistant\n" +
                               (raw ? "<think>\n\n</think>\n\n" : "");
    ASSERT_TRUE(CheckPromptPrefix(source, suffix, &program, &error)) << error;
    Expect(
        program, history,
        R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"[]"}}]})");
  }
}

TEST_P(OutputEmitterTest, ActualJsonMalformedCallsNeverPublishPartialResults) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  constexpr std::string_view request =
      R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]})";
  for (const auto *path : {"lib/text/qwen25/templates/cd8e9439.jinja",
                           "lib/text/qwen3/templates/5da44855.jinja"}) {
    SCOPED_TRACE(path);
    std::ifstream file(root / path);
    ASSERT_TRUE(file.good());
    const std::string source((std::istreambuf_iterator<char>(file)), {});
    for (
        const auto *response :
        {R"(<tool_call>{"name":"other","arguments":{}}</tool_call>)",
         R"(<tool_call>{"name":"f","name":"f","arguments":{}}</tool_call>)",
         R"(<tool_call>{"name":"f","arguments":[],}</tool_call>)",
         R"(<tool_call>{"name":"f","arguments":{"x":1,"x":2}}</tool_call>)",
         R"(<tool_call>{"name":"f","arguments":not JSON}</tool_call>)",
         R"(<tool_call>{"name":"f","arguments":{}}</tool_call><tool_call>{bad}</tool_call>)"}) {
      SCOPED_TRACE(response);
      OutputFixture program;
      std::string error, result = "unchanged";
      ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
      std::unique_ptr<PreparedTemplate> prepared;
      ASSERT_TRUE(PreparedTemplate::Prepare(program, root.string(),
                                            Required("TEST_TMPDIR"), GetParam(),
                                            &prepared, &error))
          << error;
      EXPECT_FALSE(prepared->Render(request, &result, &error));
      EXPECT_EQ(result, "unchanged");
    }
  }
}

TEST_P(OutputEmitterTest, ActualXmlMixedSchemasKeepOpaqueAndNestedValues) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/qwen35/templates/7f0e5290.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  struct Fixture {
    const char *response;
    const char *expected;
  };
  const Fixture fixtures[] = {
      {"<tool_call><function=f><parameter=b>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}]})"},
      {"<tool_call><function=f><parameter=b>\ntrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nFalse\n</parameter><parameter=c>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":false,\"c\":true}"}}]})"},
      {"<tool_call><function=g><parameter=b>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"g","arguments":"{\"b\":\"True\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nTrue\n</parameter>"
       "<parameter=s>\nTrue\n</parameter>"
       "<parameter=obj>\n{\"nested\":true,\"word\":\"False\"}\n</parameter>"
       "<parameter=arr>\n[true,\"True\"]\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"s\":\"True\",\"obj\":{\"nested\":true,\"word\":\"False\"},\"arr\":[true,\"True\"]}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nTrue\n</parameter>"
       "<parameter=s>\n<parameter=b>True\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"s\":\"<parameter=b>True\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nTrue\n</parameter></function></tool_call>\n"
       "<tool_call><function=g><parameter=b>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}},{"type":"function","function":{"name":"g","arguments":"{\"b\":\"True\"}"}}]})"},
      {" \r\n<tool_call> \t<function=f>\r\n<parameter=b>\r\n \tTrue \r\n</parameter>\r\n</function> \r\n</tool_call>\t ",
       R"({"content":" \r\n","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}]})"},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, fixture.response, &program, &error))
        << error;
    Expect(program, kXmlRequest, fixture.expected);
  }
}

TEST_P(OutputEmitterTest, ActualXmlSchemaHintsDoNotInventExecutionValidation) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/qwen35/templates/7f0e5290.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  // The native XML codec uses schema hints to choose a representation. These
  // are parser records, not permission/schema validation for executing a call.
  // They replace incorrect EXPECT_FALSEs in the legacy normalizer ledger.
  struct Fixture {
    const char *response;
    const char *expected;
  };
  const Fixture fixtures[] = {
      {"<tool_call><function=unknown><parameter=b>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"unknown","arguments":"{\"b\":\"True\"}"}}]})"},
      {"<tool_call><function=f><parameter=unknown>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"unknown\":\"True\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nTRUE\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":\"TRUE\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\n\"True\"\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":\"True\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\ntrue\n</parameter>"
       "<parameter=obj>\n{\"b\":True}\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"obj\":\"{\\\"b\\\":True}\"}"}}]})"},
      {"<tool_call><function=f><parameter=b>\nTrue\n</parameter>"
       "<parameter=b>\nFalse\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"b\":false}"}}]})"},
      {"<tool_call><function=f><parameter=s>\nTrue\n</parameter></function></tool_call>",
       R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"s\":\"True\"}"}}]})"},
  };
  for (const auto &fixture : fixtures) {
    SCOPED_TRACE(fixture.response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, fixture.response, &program, &error))
        << error;
    Expect(program, kXmlRequest, fixture.expected);
  }
  OutputFixture program;
  std::string error;
  ASSERT_TRUE(OutputProgram(
      source,
      "<tool_call><function=f><parameter=b>\nTrue\n</parameter></function></tool_call>",
      &program, &error))
      << error;
  Expect(
      program,
      R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":[]})",
      R"({"content":"","reasoning":"","calls":[{"type":"function","function":{"name":"f","arguments":"{\"b\":\"True\"}"}}]})");
  // The native request boundary owns invalid tool inventory admission. It does
  // not pass an invented request argument or a null descriptor to parse_tools.
  ExpectRejected(
      program,
      R"({"messages":[{"role":"user","content":"question"}],"enable_thinking":false,"tools":null})");
}

TEST_P(OutputEmitterTest,
       ActualXmlNamesUseNativeBytesWithoutAsciiOrLengthBans) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/qwen35/templates/7f0e5290.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  for (const std::string &name :
       {std::string("bad key"), std::string("café"), std::string(129, 'x')}) {
    SCOPED_TRACE(name);
    const std::string request =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"}],\"enable_thinking\":false,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"" +
        name + "\",\"parameters\":{\"type\":\"object\",\"properties\":{\"" +
        name + "\":{\"type\":\"boolean\"}}}}}]}";
    const std::string response = "<tool_call><function=" + name +
                                 "><parameter=" + name +
                                 ">\nTrue\n</parameter></function></tool_call>";
    const std::string expected =
        "{\"content\":\"\",\"reasoning\":\"\",\"calls\":[{\"type\":\"function\",\"function\":{\"name\":\"" +
        name + "\",\"arguments\":\"{\\\"" + name + "\\\":true}\"}}]}";
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    Expect(program, request, expected);
  }
}

TEST_P(OutputEmitterTest, ActualXmlMalformedClosuresNeverPublishPartialCalls) {
  const auto root = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE");
  std::ifstream file(root / "lib/text/qwen35/templates/7f0e5290.jinja");
  ASSERT_TRUE(file.good());
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  for (
      const auto *response :
      {"<tool_call><function=f><parameter=b>True</parameter></function>",
       "<tool_call><function=f><parameter=b>True</parameter><parameter=c>False</function></tool_call>",
       "<tool_call><function=f><parameter=b>True</parameter></function></tool_call>trailing",
       "<tool_call><function=f><parameter=b>True</parameter></function></tool_call>"
       "<tool_call><function=f><parameter=b>False"}) {
    SCOPED_TRACE(response);
    OutputFixture program;
    std::string error;
    ASSERT_TRUE(OutputProgram(source, response, &program, &error)) << error;
    ExpectRejected(program, kXmlRequest);
  }
}

INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, OutputEmitterTest,
                         ::testing::Bool());
} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
