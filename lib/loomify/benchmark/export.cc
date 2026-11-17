// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/export.h"
#include "lib/loomify/benchmark/inventory.h"
#include "lib/loomify/benchmark/stateful_inputs.h"
#include "loom/util/digest.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace hrx::loomify_gguf::benchmark {
namespace {
namespace fs = std::filesystem;

bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
std::string Quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  constexpr char digits[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 0x20)
      out << "\\u00" << digits[c >> 4] << digits[c & 15];
    else
      out << c;
  }
  out << '"';
  return out.str();
}
std::string DigestString(const std::string &value) {
  loom_sha256_t digest;
  loom_sha256_compute(iree_make_const_byte_span(value.data(), value.size()),
                      &digest);
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest.bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}
void WriteWorkload(std::ostream &out, const Workload &w) {
  out << "{\"phase\":" << Quote(w.phase)
      << ",\"token_capacity\":" << w.token_capacity
      << ",\"active_tokens\":" << w.active_tokens
      << ",\"context_capacity\":" << w.context_capacity
      << ",\"history_tokens\":" << w.history_tokens
      << ",\"platform\":" << Quote(w.platform) << ",\"seed\":" << w.seed
      << ",\"weight_policy\":" << Quote(w.weight_policy) << '}';
}
void WriteBuffer(std::ostream &out, const loomc_cmd_buffer_ref_t &buffer) {
  out << "{\"role\":"
      << Quote(buffer.role == LOOMC_CMD_BUFFER_ROLE_FIXED ? "fixed"
                                                          : "rebindable")
      << ",\"root_index\":" << buffer.root_index
      << ",\"byte_offset\":" << buffer.byte_offset << ",\"byte_length\":";
  if (buffer.byte_length == UINT64_MAX)
    out << "\"remaining\"";
  else
    out << buffer.byte_length;
  out << '}';
}
void WriteArgument(std::ostream &out, const loomc_cmd_argument_t &argument) {
  out << "{\"abi_kind\":" << argument.kind;
  if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
    out << ",\"buffer\":";
    WriteBuffer(out, argument.value.buffer);
  } else {
    std::ostringstream bits;
    bits << "0x" << std::hex << std::setfill('0') << std::setw(16)
         << argument.value.scalar_bits;
    out << ",\"bits\":" << Quote(bits.str());
  }
  out << '}';
}
void WriteCommand(std::ostream &out, const loomc_cmd_command_t &command) {
  out << "{\"kind\":" << command.kind << ",\"begins_barrier_wave\":"
      << (command.begins_barrier_wave ? "true" : "false");
  switch (command.kind) {
  case LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT:
  case LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
  case LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC: {
    const auto &d = command.payload.dispatch;
    out << ",\"entry_ordinal\":" << d.entry_ordinal
        << ",\"requirement_ordinal\":" << d.requirement_ordinal;
    if (command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
      out << ",\"grid\":[" << d.workgroup_count[0] << ','
          << d.workgroup_count[1] << ',' << d.workgroup_count[2] << ']';
    else {
      out << ",\"count_buffer\":";
      WriteBuffer(out, d.workgroup_count_buffer);
    }
    break;
  }
  case LOOMC_CMD_COMMAND_KIND_FILL:
    out << ",\"target\":";
    WriteBuffer(out, command.payload.fill.target);
    out << ",\"pattern\":" << command.payload.fill.pattern
        << ",\"pattern_length\":" << command.payload.fill.pattern_length;
    break;
  case LOOMC_CMD_COMMAND_KIND_COPY:
    out << ",\"source\":";
    WriteBuffer(out, command.payload.copy.source);
    out << ",\"target\":";
    WriteBuffer(out, command.payload.copy.target);
    break;
  case LOOMC_CMD_COMMAND_KIND_TIMESTAMP:
    out << ",\"target\":";
    WriteBuffer(out, command.payload.timestamp.target);
    break;
  default:
    break;
  }
  out << '}';
}
void WritePrograms(std::ostream &out, const Plan &plan) {
  out << '[';
  bool first = true;
  for (const auto &program : plan.programs) {
    if (!std::exchange(first, false))
      out << ',';
    out << "{\"root\":" << Quote(program.root_symbol) << ",\"parameters\":[";
    bool parameter_first = true;
    for (const auto &parameter : program.parameters) {
      if (!std::exchange(parameter_first, false))
        out << ',';
      out << "{\"key\":" << Quote(parameter.key)
          << ",\"fixed_buffer_index\":" << parameter.fixed_buffer_index
          << ",\"byte_offset\":" << parameter.byte_offset
          << ",\"byte_length\":" << parameter.byte_length
          << ",\"minimum_alignment\":" << parameter.minimum_alignment << '}';
    }
    out << "],\"commands\":[";
    for (size_t c = 0; c < program.commands.size(); ++c) {
      if (c)
        out << ',';
      WriteCommand(out, program.commands[c]);
    }
    out << "]}";
  }
  out << ']';
}
void WriteInvocation(std::ostream &out, const Invocation &invocation) {
  out << "\"invocation_id\":" << Quote(invocation.invocation_id)
      << ",\"program_ordinal\":" << invocation.program_ordinal
      << ",\"command_ordinal\":" << invocation.command_ordinal
      << ",\"kernel\":" << Quote(invocation.kernel_symbol)
      << ",\"implementation_index\":" << invocation.implementation_index
      << ",\"request_root_ordinal\":" << invocation.request_root_ordinal
      << ",\"arguments\":[";
  for (size_t a = 0; a < invocation.arguments.size(); ++a) {
    if (a)
      out << ',';
    WriteArgument(out, invocation.arguments[a]);
  }
  out << ']';
}
void WritePlanFacts(std::ostream &out, const Plan &plan) {
  out << "\"selection_id\":" << Quote(plan.selection_id)
      << ",\"workload_id\":" << Quote(plan.workload_id)
      << ",\"phase\":" << Quote(plan.workload.phase) << ",\"workload\":";
  WriteWorkload(out, plan.workload);
  out << ",\"model\":{\"family\":" << Quote(plan.model.family)
      << ",\"name\":" << Quote(plan.model.name)
      << ",\"embedding_width\":" << plan.model.embedding_width
      << ",\"vocabulary_size\":" << plan.model.vocabulary_size
      << ",\"layers\":" << plan.model.layers.size()
      << "},\"implementations\":[";
  for (size_t i = 0; i < plan.implementations.size(); ++i) {
    if (i)
      out << ',';
    out << "{\"implementation_id\":"
        << Quote(plan.implementations[i].implementation_id) << ",\"roots\":[";
    for (size_t r = 0; r < plan.implementations[i].roots.size(); ++r) {
      if (r)
        out << ',';
      const auto &root = plan.implementations[i].roots[r];
      out << "{\"module_ordinal\":" << root.module_ordinal
          << ",\"symbol_ordinal\":" << root.symbol_ordinal << '}';
    }
    out << "]}";
  }
  out << "],\"programs\":";
  WritePrograms(out, plan);
}

struct Artifact {
  std::string path;
  std::string sha256;
  uint64_t size = 0;
};
bool FileDigest(const fs::path &path, Artifact *artifact, std::string *error) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return Fail(error, "cannot read emitted artifact for content identity");
  loom_sha256_state_t state;
  loom_sha256_initialize(&state);
  std::array<char, 65536> buffer;
  uint64_t size = 0;
  while (file.read(buffer.data(), buffer.size()) || file.gcount()) {
    const auto count = static_cast<size_t>(file.gcount());
    loom_sha256_update(&state, iree_make_const_byte_span(buffer.data(), count));
    size += count;
  }
  if (file.bad())
    return Fail(error, "cannot finish reading emitted artifact");
  loom_sha256_t digest;
  loom_sha256_finalize(&state, &digest);
  constexpr char digits[] = "0123456789abcdef";
  artifact->sha256.clear();
  for (uint8_t byte : digest.bytes) {
    artifact->sha256 += digits[byte >> 4];
    artifact->sha256 += digits[byte & 15];
  }
  artifact->size = size;
  return true;
}
bool CollectArtifacts(const fs::path &directory,
                      std::vector<Artifact> *artifacts, std::string *error) {
  std::error_code ec;
  fs::recursive_directory_iterator current(directory, ec), end;
  if (ec)
    return Fail(error, "cannot enumerate emitted artifacts: " + ec.message());
  while (current != end) {
    const auto status = current->symlink_status(ec);
    if (ec)
      return Fail(error, "cannot inspect emitted artifact: " + ec.message());
    if (fs::is_symlink(status) ||
        (!fs::is_regular_file(status) && !fs::is_directory(status)))
      return Fail(error, "emitter produced a nonportable filesystem entry");
    if (fs::is_regular_file(status)) {
      Artifact artifact;
      artifact.path =
          current->path().lexically_relative(directory).generic_string();
      if (artifact.path == "manifest.json")
        return Fail(error, "emitter must leave manifest publication to facade");
      if (!FileDigest(current->path(), &artifact, error))
        return false;
      artifacts->push_back(std::move(artifact));
    }
    current.increment(ec);
    if (ec)
      return Fail(error, "cannot enumerate emitted artifacts: " + ec.message());
  }
  std::sort(
      artifacts->begin(), artifacts->end(),
      [](const Artifact &a, const Artifact &b) { return a.path < b.path; });
  if (std::none_of(artifacts->begin(), artifacts->end(), [](const Artifact &a) {
        return a.path == "benchmark.loombc" && a.size;
      }))
    return Fail(error, "emitter did not produce benchmark.loombc");
  return true;
}
void WriteRoot(std::ostream &out, const PreparedRoot &root) {
  out << "{\"role\":" << root.role << ",\"root_index\":" << root.root_index
      << ",\"original_byte_offset\":" << root.original_byte_offset
      << ",\"logical_byte_length\":" << root.bytes.size()
      << ",\"fixture_suffix_bytes\":" << root.fixture_suffix_bytes
      << ",\"fixture_payload_byte_length\":"
      << root.bytes.size() + root.fixture_suffix_bytes
      << ",\"minimum_alignment\":" << root.minimum_alignment
      << ",\"fixture\":" << Quote(root.fixture_path)
      << ",\"content_sha256\":" << Quote(root.sha256) << '}';
}
void WriteWeightSources(std::ostream &out,
                        const std::vector<WeightSource> &sources) {
  out << '[';
  for (size_t i = 0; i < sources.size(); ++i) {
    if (i)
      out << ',';
    const auto &source = sources[i];
    out << "{\"key\":" << Quote(source.key)
        << ",\"source_offset\":" << source.source_offset
        << ",\"byte_length\":" << source.byte_length << '}';
  }
  out << ']';
}
std::string ManifestJson(const Plan &plan, const CaseInventory &inventory,
                         const std::vector<EmittedCase> &emitted,
                         const std::vector<Artifact> &artifacts,
                         const ExportOptions &options, uint64_t source_size,
                         const std::string &manifest_id) {
  const auto report =
      ProjectReportInventory(plan, inventory, emitted, manifest_id);
  std::ostringstream out;
  out << "{\"format\":\"loomify.benchmark.bundle.v1\",\"manifest_id\":"
      << Quote(manifest_id) << ',';
  WritePlanFacts(out, plan);
  out << ",\"source\":{\"size\":" << source_size
      << ",\"identity_policy\":\"consumed_tensor_bytes_sha256\"}"
      << ",\"generator_identity\":" << Quote(options.generator_identity)
      << ",\"resource_policy\":{\"max_case_input_bytes\":";
  if (options.max_case_input_bytes == UINT64_MAX)
    out << "null";
  else
    out << options.max_case_input_bytes;
  out << "},\"coverage\":{\"unit\":\"selected_dispatch_origins\",\"selected_invocations\":"
      << report.invocations.size() << ",\"runnable_invocations\":";
  const auto supported = std::count_if(
      report.invocations.begin(), report.invocations.end(),
      [](const ReportInvocation &origin) { return !origin.case_id.empty(); });
  out << supported << ",\"unsupported_invocations\":"
      << report.invocations.size() - supported << ",\"complete\":"
      << (supported == report.invocations.size() ? "true" : "false")
      << "},\"input_footprint\":{\"scope\":\"logical_bytes_excluding_fixture_guards_and_headers\",\"retained_bytes\":"
      << inventory.retained_input_bytes << ",\"unique_fixture_payload_bytes\":"
      << inventory.unique_fixture_payload_bytes
      << ",\"unique_fixture_count\":" << inventory.unique_fixture_count
      << "},\"artifacts\":[";
  for (size_t i = 0; i < artifacts.size(); ++i) {
    if (i)
      out << ',';
    const auto &artifact = artifacts[i];
    out << "{\"path\":" << Quote(artifact.path) << ",\"size\":" << artifact.size
        << ",\"sha256\":" << Quote(artifact.sha256) << '}';
  }
  out << "],\"cases\":[";
  for (size_t c = 0; c < report.cases.size(); ++c) {
    if (c)
      out << ',';
    const auto &definition = report.cases[c];
    const auto &product = emitted[c];
    out << "{\"case_id\":" << Quote(definition.case_id)
        << ",\"input_policy_digest\":" << Quote(definition.input_policy_digest)
        << ",\"implementation_id\":" << Quote(definition.implementation_id)
        << ",\"measurement_subject\":" << Quote(definition.measurement_subject)
        << ",\"arithmetic_check\":" << Quote(definition.arithmetic_check)
        << ",\"logical_case_key\":" << Quote(product.logical_case_key)
        << ",\"benchmark_symbol\":" << Quote(product.benchmark_symbol)
        << ",\"setup_recipe_id\":" << Quote(product.setup_recipe_id)
        << ",\"reset_policy\":" << Quote(product.reset_policy)
        << ",\"representative_invocation_id\":" << Quote(product.invocation_id)
        << ",\"weight_sources\":";
    WriteWeightSources(out, product.weight_sources);
    out << ",\"safety_checks\":[";
    for (size_t s = 0; s < product.safety_checks.size(); ++s) {
      if (s)
        out << ',';
      out << Quote(product.safety_checks[s]);
    }
    out << "],\"roots\":[";
    for (size_t r = 0; r < product.roots.size(); ++r) {
      if (r)
        out << ',';
      WriteRoot(out, product.roots[r]);
    }
    out << "],\"resets\":[";
    for (size_t r = 0; r < product.resets.size(); ++r) {
      if (r)
        out << ',';
      out << "{\"target\":";
      WriteBuffer(out, product.resets[r].target);
      out << ",\"seed\":";
      WriteRoot(out, product.resets[r].seed);
      out << '}';
    }
    out << "],\"timed_dispatches\":[";
    for (size_t d = 0; d < product.timed_dispatches.size(); ++d) {
      if (d)
        out << ',';
      const auto &dispatch = product.timed_dispatches[d];
      out << "{\"kernel\":" << Quote(dispatch.kernel_symbol)
          << ",\"reset\":" << (dispatch.reset ? "true" : "false")
          << ",\"grid\":[" << dispatch.workgroup_count[0] << ','
          << dispatch.workgroup_count[1] << ',' << dispatch.workgroup_count[2]
          << "]}";
    }
    out << "]}";
  }
  out << "],\"invocations\":[";
  for (size_t i = 0; i < report.invocations.size(); ++i) {
    if (i)
      out << ',';
    const auto &origin = report.invocations[i];
    out << '{';
    WriteInvocation(out, plan.invocations[i]);
    const auto &coverage = inventory.origins[i];
    out << ",\"case_id\":" << Quote(origin.case_id)
        << ",\"unsupported_reason\":" << Quote(origin.unsupported_reason)
        << ",\"representative_weights\":"
        << (coverage.representative_weights ? "true" : "false")
        << ",\"weight_sources\":";
    WriteWeightSources(out, coverage.weight_sources);
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

bool Destination(const fs::path &requested, fs::path *out, std::string *error) {
  if (requested.empty())
    return Fail(error, "benchmark output path is required");
  std::error_code ec;
  auto absolute = fs::absolute(requested, ec).lexically_normal();
  if (ec || absolute.filename().empty())
    return Fail(error, "cannot resolve benchmark output destination");
  const auto parent = fs::canonical(absolute.parent_path(), ec);
  if (ec || !fs::is_directory(parent, ec) || ec)
    return Fail(error, "benchmark output parent directory must exist");
  auto destination = parent / absolute.filename();
  const auto status = fs::symlink_status(destination, ec);
  if (ec && ec != std::errc::no_such_file_or_directory)
    return Fail(error, "cannot inspect benchmark output: " + ec.message());
  if (status.type() != fs::file_type::not_found)
    return Fail(error, "benchmark output must not already exist");
  *out = std::move(destination);
  return true;
}

#if defined(__linux__)
struct SourceFile {
  int fd = -1;
  fs::path path;
  struct stat initial {};
  ~SourceFile() {
    if (fd >= 0)
      close(fd);
  }
  bool Open(const std::string &name, std::string *error) {
    std::error_code ec;
    path = fs::canonical(name, ec);
    if (ec)
      return Fail(error, "cannot resolve benchmark GGUF: " + ec.message());
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0 || fstat(fd, &initial) != 0 || !S_ISREG(initial.st_mode))
      return Fail(error, "benchmark GGUF must be a readable regular file");
    return true;
  }
  bool Unchanged(std::string *error) const {
    struct stat descriptor {
    }, pathname{};
    if (fstat(fd, &descriptor) != 0 || stat(path.c_str(), &pathname) != 0)
      return Fail(error, "GGUF source changed during benchmark export");
    const auto same = [&](const struct stat &value) {
      return value.st_dev == initial.st_dev && value.st_ino == initial.st_ino &&
             value.st_size == initial.st_size &&
             value.st_mtim.tv_sec == initial.st_mtim.tv_sec &&
             value.st_mtim.tv_nsec == initial.st_mtim.tv_nsec &&
             value.st_ctim.tv_sec == initial.st_ctim.tv_sec &&
             value.st_ctim.tv_nsec == initial.st_ctim.tv_nsec;
    };
    return (same(descriptor) && same(pathname)) ||
           Fail(error, "GGUF source changed during benchmark export");
  }
};
struct Staging {
  fs::path path;
  bool owned = false;
  ~Staging() {
    if (owned) {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  }
  bool Create(const fs::path &destination, bool directory, int *file_fd,
              std::string *error) {
    std::string name =
        (destination.parent_path() /
         ("." + destination.filename().string() + ".loomify-XXXXXX"))
            .string();
    std::vector<char> chars(name.begin(), name.end());
    chars.push_back('\0');
    if (directory) {
      if (!mkdtemp(chars.data()))
        return Fail(error, "cannot create benchmark staging directory: " +
                               std::string(std::strerror(errno)));
    } else {
      *file_fd = mkstemp(chars.data());
      if (*file_fd < 0)
        return Fail(error, "cannot create benchmark staging file: " +
                               std::string(std::strerror(errno)));
    }
    path = chars.data();
    owned = true;
    return true;
  }
  bool Abort(std::string *error) {
    if (owned) {
      std::error_code ec;
      fs::remove_all(path, ec);
      if (ec && error)
        *error += "; cannot remove owned benchmark staging " + path.string() +
                  ": " + ec.message();
      if (!ec)
        owned = false;
    }
    return false;
  }
  bool Publish(const fs::path &destination, std::string *error) {
#if defined(SYS_renameat2)
    if (syscall(SYS_renameat2, AT_FDCWD, path.c_str(), AT_FDCWD,
                destination.c_str(), RENAME_NOREPLACE) == 0) {
      owned = false;
      return true;
    }
    Fail(error, "cannot publish new benchmark output without replacement: " +
                    std::string(std::strerror(errno)));
#else
    Fail(error, "atomic no-replace benchmark publication is unavailable");
#endif
    return Abort(error);
  }
};
bool WriteFileDescriptor(int fd, const std::string &text, std::string *error) {
  size_t offset = 0;
  bool success = true;
  while (offset < text.size()) {
    const auto written = write(fd, text.data() + offset, text.size() - offset);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      success = Fail(error, "cannot write benchmark JSON output");
      break;
    }
    offset += static_cast<size_t>(written);
  }
  if (close(fd) != 0)
    success = Fail(error, "cannot close benchmark JSON output");
  return success;
}
#endif

PreparationStatus PrepareAvailable(const Plan &plan,
                                   const Invocation &invocation,
                                   uint64_t max_bytes, PreparedCase *out,
                                   std::string *error) {
  std::string unsupported;
  std::set<std::string> seen;
  for (auto prepare : {PrepareStatelessCase, PrepareStatefulCase,
                       PrepareGdnCase, PrepareAttentionCase, PrepareQkCase}) {
    std::string message;
    const auto status = prepare(plan, invocation, max_bytes, out, &message);
    if (status != PreparationStatus::kUnsupported) {
      if (error)
        *error = std::move(message);
      return status;
    }
    // Keep each provider's diagnostic without interpreting its text. In
    // particular a recognized input budget exclusion must not be replaced by
    // a later provider's generic "unsupported operation" explanation.
    if (!message.empty() && seen.insert(message).second) {
      if (!unsupported.empty())
        unsupported += "; ";
      unsupported += message;
    }
  }
  if (error)
    *error = std::move(unsupported);
  return PreparationStatus::kUnsupported;
}
} // namespace

std::string BenchmarkPlanJson(const Plan &plan) {
  std::ostringstream out;
  out << "{\"format\":\"loomify.benchmark.plan.v1\",";
  WritePlanFacts(out, plan);
  out << ",\"invocations\":[";
  for (size_t i = 0; i < plan.invocations.size(); ++i) {
    if (i)
      out << ',';
    out << '{';
    WriteInvocation(out, plan.invocations[i]);
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

bool ExportGgufBenchmarks(const std::string &gguf_path,
                          const Workload &workload,
                          const std::string &library_root,
                          const ExportOptions &options,
                          loomc_context_t *context,
                          loomc_workspace_t *workspace, ExportResult *out,
                          std::string *error) {
  if (!out || !context || !workspace || !options.max_case_input_bytes)
    return Fail(
        error,
        "benchmark export requires context, workspace, output and a positive input limit");
  fs::path destination;
  if (!Destination(options.output, &destination, error))
    return false;
#if !defined(__linux__)
  return Fail(
      error,
      "atomic no-replace benchmark export requires a supported Linux host");
#else
  SourceFile source;
  if (!source.Open(gguf_path, error))
    return false;
  Plan plan;
  if (!BuildBenchmarkPlan(source.path.string(), workload, library_root, context,
                          workspace, &plan, error))
    return false;
  ExportResult result;
  result.selection_id = plan.selection_id;
  result.selected_invocations = plan.invocations.size();
  if (options.plan_only) {
    Staging stage;
    int fd = -1;
    if (!stage.Create(destination, false, &fd, error))
      return false;
    if (!WriteFileDescriptor(fd, BenchmarkPlanJson(plan), error) ||
        !source.Unchanged(error))
      return stage.Abort(error);
    if (!stage.Publish(destination, error))
      return false;
    *out = std::move(result);
    if (error)
      error->clear();
    return true;
  }
  CaseInventory inventory;
  if (!BuildCaseInventory(plan, options.max_case_input_bytes, PrepareAvailable,
                          &inventory, error))
    return false;
  result.case_count = inventory.groups.size();
  for (const auto &origin : inventory.origins)
    origin.group_index ? ++result.exported_invocations
                       : ++result.unsupported_invocations;
  result.complete = result.unsupported_invocations == 0;
  if (!options.allow_partial && !result.complete)
    return Fail(
        error,
        "benchmark export has " +
            std::to_string(result.unsupported_invocations) +
            " unsupported selected invocations; use --plan-only to inspect or --allow-partial to export the runnable subset");
  if (inventory.prepared_cases.empty())
    return Fail(error, "no selected invocations have runnable input recipes");
  Staging stage;
  int unused_fd = -1;
  if (!stage.Create(destination, true, &unused_fd, error))
    return false;
  std::vector<EmittedCase> emitted;
  if (!EmitPreparedBenchmarkBundle(plan, std::move(inventory.prepared_cases),
                                   stage.path, context, workspace, &emitted,
                                   error))
    return stage.Abort(error);
  if (emitted.size() != inventory.groups.size()) {
    Fail(error, "emitted case count does not match selected case inventory");
    return stage.Abort(error);
  }
  for (size_t c = 0; c < emitted.size(); ++c) {
    const auto &group = inventory.groups[c];
    const auto &expected = group.identity;
    if (emitted[c].case_id != expected.case_id ||
        emitted[c].input_policy_digest != expected.input_policy_digest ||
        emitted[c].logical_case_key != expected.logical_case_key ||
        emitted[c].invocation_id !=
            plan.invocations[group.representative_invocation_index]
                .invocation_id ||
        emitted[c].measurement_subject != group.measurement_subject ||
        emitted[c].arithmetic_check != group.arithmetic_check ||
        emitted[c].benchmark_symbol.empty()) {
      Fail(error, "emitted case identity does not match selected inventory");
      return stage.Abort(error);
    }
  }
  std::error_code ec;
  if (!options.emit_source && !fs::remove(stage.path / "benchmark.loom", ec)) {
    Fail(error,
         "cannot remove optional benchmark source output: " + ec.message());
    return stage.Abort(error);
  }
  std::vector<Artifact> artifacts;
  if (!CollectArtifacts(stage.path, &artifacts, error))
    return stage.Abort(error);
  auto render = [&](const std::string &id) {
    return ManifestJson(plan, inventory, emitted, artifacts, options,
                        static_cast<uint64_t>(source.initial.st_size), id);
  };
  result.manifest_id = DigestString(render(""));
  std::ofstream manifest(stage.path / "manifest.json", std::ios::binary);
  const auto json = render(result.manifest_id);
  manifest.write(json.data(), json.size());
  manifest.close();
  if (!manifest) {
    Fail(error, "cannot write or close benchmark manifest");
    return stage.Abort(error);
  }
  if (!source.Unchanged(error))
    return stage.Abort(error);
  if (!stage.Publish(destination, error))
    return false;
  *out = std::move(result);
  if (error)
    error->clear();
  return true;
#endif
}

bool WriteBenchmarkReportFile(const BenchmarkReport &report,
                              const fs::path &new_file, std::string *error) {
  fs::path destination;
  if (!Destination(new_file, &destination, error))
    return false;
#if !defined(__linux__)
  return Fail(
      error,
      "atomic no-replace benchmark output requires a supported Linux host");
#else
  Staging stage;
  int fd = -1;
  if (!stage.Create(destination, false, &fd, error))
    return false;
  if (!WriteFileDescriptor(fd, BenchmarkReportJson(report), error))
    return stage.Abort(error);
  if (!stage.Publish(destination, error))
    return false;
  if (error)
    error->clear();
  return true;
#endif
}

} // namespace hrx::loomify_gguf::benchmark
