// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "iree/base/tooling/flags.h"
#include "lib/loomify/benchmark/export.h"
#include "lib/loomify/benchmark/report_io.h"
#include "loom/util/digest.h"
#include "loomc/target/amdgpu.h"
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <string_view>

IREE_FLAG(string, gguf, nullptr, "Input GGUF checkpoint.");
IREE_FLAG(string, output, nullptr,
          "New bundle directory, or new JSON file with --plan-only/report.");
IREE_FLAG(string, phase, "prefill", "Numerical phase: prefill or decode.");
IREE_FLAG(string, tokens, nullptr,
          "Prefill token capacity (default128); decode capacity is one.");
IREE_FLAG_NAMED(string, active_tokens, "active-tokens", nullptr,
                "Active token count (default: token capacity).");
IREE_FLAG_NAMED(string, context_capacity, "context-capacity", "512",
                "Numerical context capacity.");
IREE_FLAG_NAMED(string, history_tokens, "history-tokens", "0",
                "History tokens before this invocation.");
IREE_FLAG(string, platform, "amdgpu:gfx11-generic",
          "Maintained generation platform, not a live device selector.");
IREE_FLAG(string, seed, "0", "Deterministic input seed.");
IREE_FLAG(string, weights, "representative",
          "Real-weight policy: representative or all.");
IREE_FLAG_NAMED(string, library_root, "library-root", nullptr,
                "Explicit maintained Loom resource root (default: runfiles).");
IREE_FLAG_NAMED(
    bool, plan_only, "plan-only", false,
    "Write selected numerical plan JSON without preparing fixtures.");
IREE_FLAG_NAMED(bool, allow_partial, "allow-partial", false,
                "Export runnable cases and retain every unsupported origin.");
IREE_FLAG_NAMED(bool, emit_source, "emit-source", true,
                "Include benchmark.loom alongside benchmark.loombc.");
IREE_FLAG_NAMED(
    string, max_input_bytes, "max-input-bytes", nullptr,
    "Optional prepared-root byte budget per case; no cap by default.");
IREE_FLAG(string, manifest, nullptr, "Export manifest for the report command.");
IREE_FLAG(string, index, nullptr,
          "Explicit result index for the report command.");

namespace {
namespace benchmark = hrx::loomify_gguf::benchmark;
namespace fs = std::filesystem;
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;

int Error(const std::string &message) {
  std::fprintf(stderr, "loomify-benchmark: %s\n", message.c_str());
  return 1;
}
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    std::fprintf(stderr, "loomify-benchmark: %.*s\n", int(message.size),
                 message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Unsigned(const char *value, const char *name, uint64_t *out,
              std::string *error) {
  if (!value || !*value) {
    *error = std::string("--") + name + " requires an unsigned integer";
    return false;
  }
  const std::string_view text(value);
  uint64_t number = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), number);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    *error = std::string("--") + name + " requires an unsigned integer";
    return false;
  }
  *out = number;
  return true;
}
fs::path LibraryRoot(const char *executable) {
  if (const auto *root = std::getenv("RUNFILES_DIR"))
    if (*root)
      return fs::path(root) / "_main";
  const auto conventional =
      fs::path(std::string(executable) + ".runfiles") / "_main";
  std::error_code ec;
  if (fs::is_directory(conventional, ec))
    return conventional;
#if defined(__linux__)
  const auto actual = fs::read_symlink("/proc/self/exe", ec);
  if (!ec)
    return fs::path(actual.string() + ".runfiles") / "_main";
#endif
  return conventional;
}
std::string GeneratorIdentity() {
#if defined(__linux__)
  // The command is statically linked to its numerical compiler dependencies.
  // Name the actual running generator, never an assumed source pin or runner.
  std::ifstream executable("/proc/self/exe", std::ios::binary);
  if (!executable)
    return {};
  loom_sha256_state_t state;
  loom_sha256_initialize(&state);
  std::array<char, 65536> bytes;
  while (executable.read(bytes.data(), bytes.size()) || executable.gcount())
    loom_sha256_update(
        &state, iree_make_const_byte_span(bytes.data(), executable.gcount()));
  if (executable.bad())
    return {};
  loom_sha256_t digest;
  loom_sha256_finalize(&state, &digest);
  constexpr char digits[] = "0123456789abcdef";
  std::string result = "sha256:";
  for (uint8_t byte : digest.bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
#else
  return {};
#endif
}
bool ReportFlagsOnly(int argc, char **argv) {
  const std::set<std::string_view> allowed = {"manifest", "index", "output",
                                              "help"};
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument.substr(0, 2) != "--")
      continue;
    const auto key =
        argument.substr(2, argument.find('=') == std::string_view::npos
                               ? argument.size() - 2
                               : argument.find('=') - 2);
    if (!allowed.count(key))
      return false;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  const bool report_mode = argc > 1 && std::strcmp(argv[1], "report") == 0;
  if (report_mode) {
    for (int i = 1; i < argc; ++i)
      argv[i] = argv[i + 1];
    --argc;
    if (!ReportFlagsOnly(argc, argv))
      return Error("report accepts --manifest, --index and --output only");
  }
  iree_flags_set_usage(
      "loomify-benchmark",
      "Export standalone cases from a GGUF-selected numerical pipeline.\n"
      "  --gguf=MODEL --output=NEW_DIRECTORY [--phase=prefill|decode]\n"
      "  --gguf=MODEL --plan-only --output=NEW_PLAN.json\n"
      "  report --manifest=MANIFEST --index=RESULT_INDEX --output=NEW_REPORT.json\n"
      "Generation is CPU-only and requires neither tokenizer nor Serve.\n"
      "Complete runnable coverage is required unless --allow-partial is explicit.\n"
      "Generated cases run with the existing loom-benchmark tool; export does not run them.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1 || !FLAG_output || !*FLAG_output)
    return Error(
        "--output=NEW_PATH is required and positional arguments are not accepted");
  std::string error;
  if (report_mode) {
    if (!FLAG_manifest || !*FLAG_manifest || !FLAG_index || !*FLAG_index)
      return Error("report requires --manifest=PATH and --index=PATH");
    benchmark::BenchmarkReport report;
    if (!benchmark::BuildBenchmarkReportFromFiles(FLAG_manifest, FLAG_index, {},
                                                  &report, &error) ||
        !benchmark::WriteBenchmarkReportFile(report, FLAG_output, &error))
      return Error(error);
    std::printf(
        "measured=%llu selected=%llu qualified=%llu\n",
        static_cast<unsigned long long>(report.coverage.measured_invocations),
        static_cast<unsigned long long>(report.coverage.selected_invocations),
        static_cast<unsigned long long>(report.coverage.qualified_invocations));
    return 0;
  }
  if (!FLAG_gguf || !*FLAG_gguf || FLAG_manifest || FLAG_index)
    return Error(
        "generation requires --gguf=PATH; --manifest/--index belong to report");
  benchmark::Workload workload;
  workload.phase = FLAG_phase;
  if (workload.phase != "prefill" && workload.phase != "decode")
    return Error("--phase must be prefill or decode");
  workload.token_capacity = workload.phase == "decode" ? 1 : 128;
  if (FLAG_tokens &&
      !Unsigned(FLAG_tokens, "tokens", &workload.token_capacity, &error))
    return Error(error);
  workload.active_tokens = workload.token_capacity;
  if (FLAG_active_tokens && !Unsigned(FLAG_active_tokens, "active-tokens",
                                      &workload.active_tokens, &error))
    return Error(error);
  if (!Unsigned(FLAG_context_capacity, "context-capacity",
                &workload.context_capacity, &error) ||
      !Unsigned(FLAG_history_tokens, "history-tokens", &workload.history_tokens,
                &error) ||
      !Unsigned(FLAG_seed, "seed", &workload.seed, &error))
    return Error(error);
  if (workload.phase == "decode" &&
      (workload.token_capacity != 1 || workload.active_tokens != 1))
    return Error(
        "decode requires --tokens=1 and --active-tokens=1 when supplied");
  workload.platform = FLAG_platform;
  workload.weight_policy = FLAG_weights;
  benchmark::ExportOptions options;
  options.output = FLAG_output;
  options.plan_only = FLAG_plan_only;
  options.allow_partial = FLAG_allow_partial;
  options.emit_source = FLAG_emit_source;
  options.generator_identity = GeneratorIdentity();
  if (FLAG_max_input_bytes && !Unsigned(FLAG_max_input_bytes, "max-input-bytes",
                                        &options.max_case_input_bytes, &error))
    return Error(error);
  const auto root =
      FLAG_library_root ? fs::path(FLAG_library_root) : LibraryRoot(argv[0]);
  if (root.empty())
    return Error("--library-root must not be empty");
  loomc_target_environment_t *raw_environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &raw_environment)))
    return 1;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t context_options{};
  context_options.next = &target;
  loomc_context_t *raw_context = nullptr;
  if (!Ok(loomc_context_create(&context_options, loomc_allocator_system(),
                               &raw_context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context(raw_context,
                                                        loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &raw_workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace, loomc_workspace_release);
  benchmark::ExportResult result;
  if (!benchmark::ExportGgufBenchmarks(FLAG_gguf, workload, root.string(),
                                       options, context.get(), workspace.get(),
                                       &result, &error))
    return Error(error);
  if (options.plan_only)
    std::printf("planned=%llu selection=%s\n",
                static_cast<unsigned long long>(result.selected_invocations),
                result.selection_id.c_str());
  else
    std::printf(
        "selected=%llu exported=%llu unsupported=%llu cases=%llu complete=%s\n",
        static_cast<unsigned long long>(result.selected_invocations),
        static_cast<unsigned long long>(result.exported_invocations),
        static_cast<unsigned long long>(result.unsupported_invocations),
        static_cast<unsigned long long>(result.case_count),
        result.complete ? "true" : "false");
  return 0;
}
