// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_PLAN_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_PLAN_H_

#include "lib/loomify/gguf_model_builder.h"
#include "loomc/product.h"
#include "loomc/target/cmd/program.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::benchmark {

struct Workload {
  std::string phase = "prefill";
  uint64_t token_capacity = 128;
  uint64_t active_tokens = 128;
  uint64_t context_capacity = 512;
  uint64_t history_tokens = 0;
  std::string platform = "amdgpu:gfx11-generic";
  uint64_t seed = 0;
  std::string weight_policy = "representative";
};

// Independent kernel request retained beyond the command product/compiler.
// Native request root ordinals refer only to this request's exact bytecode.
struct Implementation {
  std::string implementation_id;
  std::shared_ptr<loomc_request_t> request;
  std::vector<loomc_request_root_t> roots;
};

struct Parameter {
  std::string key;
  uint32_t fixed_buffer_index = 0;
  uint64_t byte_offset = 0;
  uint64_t byte_length = 0;
  uint64_t minimum_alignment = 0;
};

struct Program {
  std::string root_symbol;
  loomc_cmd_program_layout_t layout{};
  std::vector<loomc_cmd_launch_binding_t> launch_bindings;
  std::vector<loomc_cmd_parameter_root_t> parameter_roots;
  std::vector<Parameter> parameters;
  // Complete final ordered schedule, including barriers/fills/copies/helpers.
  std::vector<loomc_cmd_command_t> commands;
};

// A dispatch occurrence in the complete selected schedule. Source provenance
// is optional; program/command ordinals are exact within this owned plan.
struct Invocation {
  std::string invocation_id;
  uint32_t program_ordinal = 0;
  uint32_t command_ordinal = 0;
  std::string kernel_symbol;
  uint32_t implementation_index = 0;
  uint32_t request_root_ordinal = 0;
  loomc_cmd_command_t command{};
  std::vector<loomc_cmd_argument_t> arguments;
};

struct Plan {
  // Selection identity only; final export manifest also binds actual fixtures.
  std::string selection_id;
  std::string workload_id;
  Workload workload;
  model::ModelDescriptor model;
  std::optional<PublicationTokenFacts> publication_tokens;
  // Generation-time immutable input; never copied into consumer fixture paths.
  std::string source_asset;
  std::vector<Implementation> implementations;
  std::vector<Program> programs;
  std::vector<Invocation> invocations;
};

struct PreparedCase;
struct CaseIdentity {
  std::string case_id;
  std::string logical_case_key;
  std::string input_policy_digest;
  // Internal grouping identity with only declared immutable weight bytes
  // masked. Empty when no weight substitution was declared; never an emitted
  // case ID.
  std::string representative_group_key;
};
// Identity binds exact emitted root bytes/alias topology, not asset filenames.
// The logical comparison key omits implementation bytes and physical grid;
// equivalence still requires the operation-owned recipe/replay proof.
bool BuildCaseIdentity(const Plan &plan, const Invocation &invocation,
                       const PreparedCase &prepared, CaseIdentity *out,
                       std::string *error);

// Synchronous CPU selection/planning. Borrows native context/workspace only for
// this call; returned requests/facts own their data. Failure preserves *out.
// Source weights are not read/materialized; no tokenizer.json/Serve/GPU needed.
bool BuildBenchmarkPlan(const std::string &gguf_path, const Workload &workload,
                        const std::string &library_root,
                        loomc_context_t *context, loomc_workspace_t *workspace,
                        Plan *out, std::string *error);

} // namespace hrx::loomify_gguf::benchmark
#endif
