// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_INPUTS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_INPUTS_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "loomc/target/cmd/program.h"

namespace hrx::loomify_gguf::benchmark {

inline constexpr uint64_t kBenchmarkTailGuardBytes = 16;
inline constexpr uint8_t kBenchmarkTailGuardValue = 0x5A;

// A bounded realization of one final native root. Root-relative offsets are
// translated by original_byte_offset; relative offsets and aliases are
// retained. Recipes may trim unused prefix/suffix bytes only when their
// operation contract proves those bytes cannot be read through scalar offsets
// or hidden views.
struct PreparedRoot {
  loomc_cmd_buffer_role_t role = LOOMC_CMD_BUFFER_ROLE_REBINDABLE;
  uint32_t root_index = 0;
  uint64_t original_byte_offset = 0;
  uint64_t minimum_alignment = 1;
  std::vector<uint8_t> bytes;
  // Internal root-relative initialized spans; conflicting alias writes reject.
  std::vector<std::pair<uint64_t, uint64_t>> initialized_ranges;
  std::string fixture_path;
  std::string sha256;
  // Fixture-only allocation guard, excluded from the kernel's logical view.
  uint64_t fixture_suffix_bytes = 0;
};

struct BufferExtent {
  loomc_cmd_buffer_ref_t binding{};
  uint64_t byte_length = 0;
  uint64_t minimum_alignment = 1;
};

// Immutable independent seed storage for one exact live-root span. The seed
// index is local to this reset list; it is not a model root or parameter.
struct StateReset {
  loomc_cmd_buffer_ref_t target{};
  PreparedRoot seed;
};

struct WeightSource {
  std::string key;
  uint64_t source_offset = 0;
  uint64_t byte_length = 0;
};

// Owned operation preparation shared by case identity and emission. It contains
// data and the declared source ABI, never model reconstruction or a compiler.
struct PreparedCase {
  std::string setup_recipe_id;
  std::string measurement_subject = "kernel";
  std::string reset_policy = "none_overwrite_only";
  std::string arithmetic_check = "unchecked";
  std::vector<PreparedRoot> roots;
  std::vector<StateReset> resets;
  // Source-backed exact extents for native buffer arguments, in ABI order.
  std::vector<BufferExtent> buffer_extents;
  // Only explicit immutable weight spans may use representative substitution.
  std::vector<BufferExtent> representative_weight_spans;
  std::vector<uint64_t> source_workloads;
  std::vector<std::string> source_argument_types;
  std::vector<WeightSource> weight_sources;
};

// Snapshot only initialized mutable spans, merging overlap/adjacency in each
// native root. Gaps/guards are untouched; seed storage is independent.
// Optional max_input_bytes is the ORIGINAL total logical root+seed cap. The
// merged seed footprint is checked before allocating seed payloads; valid
// exclusions set budget_exceeded, malformed spans do not. Output stays
// unchanged.
bool CaptureStateResets(const std::vector<BufferExtent> &mutable_spans,
                        const std::vector<PreparedRoot> &roots,
                        std::vector<StateReset> *out, std::string *error,
                        uint64_t max_input_bytes = UINT64_MAX,
                        bool *budget_exceeded = nullptr);

// Allocate the union of declared spans for each root. Equal native roots always
// share storage; distinct roots remain distinct even when fixture bytes dedup.
// max_bytes bounds the sum before allocations. Output is unchanged on failure.
// budget_exceeded is true only for a valid complete footprint above max_bytes;
// malformed ranges/host overflows remain ordinary failures.
bool AllocateInputRoots(const std::vector<BufferExtent> &extents,
                        uint64_t max_bytes, std::vector<PreparedRoot> *out,
                        std::string *error, bool *budget_exceeded = nullptr);

// Exact, checked little-endian payload writes. No inferred dtype conversions.
bool SetInputBytes(const loomc_cmd_buffer_ref_t &binding,
                   const std::vector<uint8_t> &bytes,
                   std::vector<PreparedRoot> *roots, std::string *error);
bool SetFiniteF32Input(const loomc_cmd_buffer_ref_t &binding,
                       uint64_t element_count, uint64_t seed,
                       std::vector<PreparedRoot> *roots, std::string *error);
bool SetTensorInput(const loomc_cmd_buffer_ref_t &binding,
                    const std::filesystem::path &gguf_path,
                    uint64_t tensor_offset, uint64_t byte_length,
                    std::vector<PreparedRoot> *roots, std::string *error);

// Write exact byte tensors as NPY v1 |i1 fixtures, addressed by SHA-256.
// Existing files are compared exactly before reuse. Paths are relative to
// bundle_root.
bool WriteInputFixtures(const std::filesystem::path &bundle_root,
                        std::vector<PreparedRoot> *roots, std::string *error,
                        uint64_t suffix_guard_bytes = 0);

const PreparedRoot *FindInputRoot(const std::vector<PreparedRoot> &roots,
                                  const loomc_cmd_buffer_ref_t &binding);

} // namespace hrx::loomify_gguf::benchmark
#endif
