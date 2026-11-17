// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/inputs.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

#include "loom/util/digest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool Add(uint64_t a, uint64_t b, uint64_t *out) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    return false;
  *out = a + b;
  return true;
}
bool IsAlignment(uint64_t value) { return value && !(value & (value - 1)); }
bool Initialized(const PreparedRoot &root, uint64_t begin, uint64_t end) {
  auto spans = root.initialized_ranges;
  std::sort(spans.begin(), spans.end());
  uint64_t cursor = begin;
  for (const auto &[offset, length] : spans) {
    if (offset > cursor)
      return false;
    uint64_t span_end;
    if (!Add(offset, length, &span_end))
      return false;
    cursor = std::max(cursor, span_end);
    if (cursor >= end)
      return true;
  }
  return false;
}
bool Fits(const loomc_cmd_buffer_ref_t &binding, uint64_t byte_length,
          const std::vector<PreparedRoot> *roots) {
  if (!roots)
    return false;
  const auto *root = FindInputRoot(*roots, binding);
  if (!root || binding.byte_offset < root->original_byte_offset ||
      (binding.byte_length != UINT64_MAX && byte_length > binding.byte_length))
    return false;
  const uint64_t offset = binding.byte_offset - root->original_byte_offset;
  return offset <= root->bytes.size() &&
         byte_length <= root->bytes.size() - offset;
}
std::string Digest(const std::vector<uint8_t> &bytes, uint64_t guard_bytes) {
  loom_sha256_state_t state;
  loom_sha256_initialize(&state);
  loom_sha256_update(&state,
                     iree_make_const_byte_span(bytes.data(), bytes.size()));
  std::array<uint8_t, kBenchmarkTailGuardBytes> guard;
  guard.fill(kBenchmarkTailGuardValue);
  loom_sha256_update(&state,
                     iree_make_const_byte_span(guard.data(), guard_bytes));
  loom_sha256_t digest;
  loom_sha256_finalize(&state, &digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string text;
  for (uint8_t byte : digest.bytes) {
    text += hex[byte >> 4];
    text += hex[byte & 15];
  }
  return text;
}
std::vector<uint8_t> NpyHeader(uint64_t byte_length) {
  std::string header = "{'descr': '|i1', 'fortran_order': False, 'shape': (" +
                       std::to_string(byte_length) + ",), }";
  // NPY v1 header and data boundary is a multiple of 64 bytes.
  header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
  header += '\n';
  std::vector<uint8_t> data = {0x93,
                               'N',
                               'U',
                               'M',
                               'P',
                               'Y',
                               1,
                               0,
                               static_cast<uint8_t>(header.size()),
                               static_cast<uint8_t>(header.size() >> 8)};
  data.insert(data.end(), header.begin(), header.end());
  return data;
}
} // namespace

const PreparedRoot *FindInputRoot(const std::vector<PreparedRoot> &roots,
                                  const loomc_cmd_buffer_ref_t &binding) {
  for (const auto &root : roots)
    if (root.role == binding.role && root.root_index == binding.root_index)
      return &root;
  return nullptr;
}

bool CaptureStateResets(const std::vector<BufferExtent> &mutable_spans,
                        const std::vector<PreparedRoot> &roots,
                        std::vector<StateReset> *out, std::string *error,
                        uint64_t max_input_bytes, bool *budget_exceeded) {
  if (budget_exceeded)
    *budget_exceeded = false;
  if (!out)
    return Fail(error, "state reset output is required");
  struct Span {
    uint64_t begin, end;
  };
  using RootKey = std::pair<loomc_cmd_buffer_role_t, uint32_t>;
  std::map<RootKey, std::vector<Span>> groups;
  for (const auto &extent : mutable_spans) {
    const auto &ref = extent.binding;
    uint64_t end;
    const auto *root = FindInputRoot(roots, ref);
    if (ref.role != LOOMC_CMD_BUFFER_ROLE_REBINDABLE || !extent.byte_length ||
        !root || !Add(ref.byte_offset, extent.byte_length, &end) ||
        (ref.byte_length != UINT64_MAX &&
         extent.byte_length > ref.byte_length) ||
        ref.byte_offset < root->original_byte_offset)
      return Fail(error, "state reset span is outside its initialized root");
    const uint64_t begin = ref.byte_offset - root->original_byte_offset;
    if (begin > root->bytes.size() ||
        extent.byte_length > root->bytes.size() - begin ||
        !Initialized(*root, begin, begin + extent.byte_length))
      return Fail(error, "state reset span is outside its initialized root");
    groups[{ref.role, ref.root_index}].push_back({ref.byte_offset, end});
  }
  for (auto &[key, spans] : groups) {
    std::sort(spans.begin(), spans.end(),
              [](Span a, Span b) { return a.begin < b.begin; });
    std::vector<Span> merged;
    for (const auto span : spans) {
      if (!merged.empty() && span.begin <= merged.back().end)
        merged.back().end = std::max(merged.back().end, span.end);
      else
        merged.push_back(span);
    }
    spans = std::move(merged);
  }
  // Check the complete logical footprint before allocating any seed payload.
  // max_input_bytes is the original caller cap, never a root-only reservation.
  uint64_t total = 0;
  for (const auto &root : roots)
    if (!Add(total, root.bytes.size(), &total))
      return Fail(error, "state reset footprint overflows");
  for (const auto &[key, spans] : groups)
    for (const auto span : spans)
      if (!Add(total, span.end - span.begin, &total))
        return Fail(error, "state reset footprint overflows");
  if (total > max_input_bytes) {
    if (budget_exceeded)
      *budget_exceeded = true;
    return Fail(error, "benchmark roots plus reset seeds exceed byte budget");
  }
  std::vector<StateReset> resets;
  for (const auto &[key, spans] : groups) {
    const loomc_cmd_buffer_ref_t root_ref{key.first, key.second, 0, UINT64_MAX};
    const auto *root = FindInputRoot(roots, root_ref);
    for (const auto span : spans) {
      const auto begin = span.begin - root->original_byte_offset;
      const auto end = span.end - root->original_byte_offset;
      StateReset reset;
      reset.target = {key.first, key.second, span.begin, span.end - span.begin};
      reset.seed.role = LOOMC_CMD_BUFFER_ROLE_FIXED;
      reset.seed.root_index = static_cast<uint32_t>(resets.size());
      reset.seed.minimum_alignment = 1;
      reset.seed.bytes.assign(root->bytes.begin() + begin,
                              root->bytes.begin() + end);
      reset.seed.initialized_ranges.emplace_back(0, reset.seed.bytes.size());
      resets.push_back(std::move(reset));
    }
  }
  *out = std::move(resets);
  if (error)
    error->clear();
  return true;
}

bool AllocateInputRoots(const std::vector<BufferExtent> &extents,
                        uint64_t max_bytes, std::vector<PreparedRoot> *out,
                        std::string *error, bool *budget_exceeded) {
  if (budget_exceeded)
    *budget_exceeded = false;
  if (!out)
    return Fail(error, "input root output is required");
  struct Range {
    uint64_t begin = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    uint64_t alignment = 1;
  };
  std::map<std::pair<loomc_cmd_buffer_role_t, uint32_t>, Range> ranges;
  for (const auto &extent : extents) {
    if ((extent.binding.role != LOOMC_CMD_BUFFER_ROLE_FIXED &&
         extent.binding.role != LOOMC_CMD_BUFFER_ROLE_REBINDABLE) ||
        !extent.byte_length || !IsAlignment(extent.minimum_alignment) ||
        (extent.binding.byte_length != UINT64_MAX &&
         extent.byte_length > extent.binding.byte_length))
      return Fail(error, "invalid input extent, role or alignment");
    uint64_t end;
    if (!Add(extent.binding.byte_offset, extent.byte_length, &end))
      return Fail(error, "input range overflows");
    auto &range = ranges[{extent.binding.role, extent.binding.root_index}];
    range.begin = std::min(range.begin, extent.binding.byte_offset);
    range.end = std::max(range.end, end);
    range.alignment = std::max(range.alignment, extent.minimum_alignment);
  }
  uint64_t total = 0;
  for (auto &[key, range] : ranges) {
    range.begin &= ~(range.alignment - 1);
    const uint64_t size = range.end - range.begin;
    if (size > std::numeric_limits<size_t>::max() || !Add(total, size, &total))
      return Fail(error, "benchmark input footprint exceeds host limits");
  }
  if (total > max_bytes) {
    if (budget_exceeded)
      *budget_exceeded = true;
    return Fail(error, "benchmark input footprint exceeds byte budget");
  }
  std::vector<PreparedRoot> roots;
  for (const auto &[key, range] : ranges) {
    PreparedRoot root;
    root.role = key.first;
    root.root_index = key.second;
    root.original_byte_offset = range.begin;
    root.minimum_alignment = range.alignment;
    root.bytes.resize(static_cast<size_t>(range.end - range.begin), 0);
    roots.push_back(std::move(root));
  }
  *out = std::move(roots);
  if (error)
    error->clear();
  return true;
}

bool SetInputBytes(const loomc_cmd_buffer_ref_t &binding,
                   const std::vector<uint8_t> &bytes,
                   std::vector<PreparedRoot> *roots, std::string *error) {
  if (!roots)
    return Fail(error, "input roots are required");
  for (auto &root : *roots) {
    if (root.role != binding.role || root.root_index != binding.root_index)
      continue;
    if (binding.byte_offset < root.original_byte_offset ||
        (binding.byte_length != UINT64_MAX &&
         bytes.size() > binding.byte_length))
      return Fail(error, "input write exceeds binding extent");
    const uint64_t offset = binding.byte_offset - root.original_byte_offset;
    if (offset > root.bytes.size() || bytes.size() > root.bytes.size() - offset)
      return Fail(error, "input write exceeds allocated root");
    for (const auto &[begin, length] : root.initialized_ranges) {
      const uint64_t overlap_begin = std::max(begin, offset);
      const uint64_t overlap_end =
          std::min(begin + length, offset + bytes.size());
      if (overlap_begin < overlap_end &&
          !std::equal(root.bytes.begin() + overlap_begin,
                      root.bytes.begin() + overlap_end,
                      bytes.begin() + (overlap_begin - offset)))
        return Fail(error,
                    "conflicting initialization through aliased input views");
    }
    std::copy(bytes.begin(), bytes.end(), root.bytes.begin() + offset);
    root.initialized_ranges.emplace_back(offset, bytes.size());
    root.fixture_path.clear();
    root.sha256.clear();
    root.fixture_suffix_bytes = 0;
    if (error)
      error->clear();
    return true;
  }
  return Fail(error, "input binding root was not allocated");
}

bool SetFiniteF32Input(const loomc_cmd_buffer_ref_t &binding,
                       uint64_t element_count, uint64_t seed,
                       std::vector<PreparedRoot> *roots, std::string *error) {
  if (element_count > std::numeric_limits<size_t>::max() / 4 ||
      !Fits(binding, element_count * 4, roots))
    return Fail(error, "F32 elements exceed allocated input range");
  std::vector<uint8_t> bytes(static_cast<size_t>(element_count) * 4);
  for (size_t i = 0; i < element_count; ++i) {
    // Bounded dyadic values are exactly reproducible and include mixed signs.
    const float value =
        static_cast<float>(static_cast<int>((i % 61 + seed % 61) % 61) - 30) /
        64.0f;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned lane = 0; lane < 4; ++lane)
      bytes[i * 4 + lane] = static_cast<uint8_t>(bits >> (8 * lane));
  }
  return SetInputBytes(binding, bytes, roots, error);
}

bool SetTensorInput(const loomc_cmd_buffer_ref_t &binding,
                    const std::filesystem::path &gguf_path,
                    uint64_t tensor_offset, uint64_t byte_length,
                    std::vector<PreparedRoot> *roots, std::string *error) {
  if (!Fits(binding, byte_length, roots) ||
      byte_length > std::numeric_limits<size_t>::max() ||
      tensor_offset >
          static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
      byte_length >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
    return Fail(error, "GGUF fixture range exceeds host file limits");
  std::ifstream file(gguf_path, std::ios::binary);
  if (!file)
    return Fail(error, "cannot open GGUF fixture source");
  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  if (size < 0 || tensor_offset > static_cast<uint64_t>(size) ||
      byte_length > static_cast<uint64_t>(size) - tensor_offset)
    return Fail(error, "GGUF fixture source is truncated");
  file.seekg(static_cast<std::streamoff>(tensor_offset));
  std::vector<uint8_t> bytes(static_cast<size_t>(byte_length));
  if (!file.read(reinterpret_cast<char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size())))
    return Fail(error, "cannot read GGUF fixture source span");
  return SetInputBytes(binding, bytes, roots, error);
}

bool WriteInputFixtures(const std::filesystem::path &bundle_root,
                        std::vector<PreparedRoot> *roots, std::string *error,
                        uint64_t suffix_guard_bytes) {
  if (!roots || (suffix_guard_bytes != 0 &&
                 suffix_guard_bytes != kBenchmarkTailGuardBytes))
    return Fail(error,
                "input roots and supported fixture guard size are required");
  std::error_code ec;
  std::filesystem::create_directories(bundle_root / "fixtures", ec);
  if (ec)
    return Fail(error,
                "cannot create benchmark fixture directory: " + ec.message());
  std::array<uint8_t, kBenchmarkTailGuardBytes> guard;
  guard.fill(kBenchmarkTailGuardValue);
  for (auto &root : *roots) {
    uint64_t payload_bytes;
    if (!Add(root.bytes.size(), suffix_guard_bytes, &payload_bytes))
      return Fail(error, "guarded fixture extent overflows");
    const auto digest = Digest(root.bytes, suffix_guard_bytes);
    const auto relative = "fixtures/" + digest + ".npy";
    const auto path = bundle_root / relative;
    const auto header = NpyHeader(payload_bytes);
    const bool exists = std::filesystem::exists(path, ec);
    if (ec)
      return Fail(error, "cannot inspect benchmark fixture: " + ec.message());
    if (exists) {
      std::ifstream input(path, std::ios::binary);
      // Compare in bounded chunks: real weight roots may be very large.
      auto matches = [&](const uint8_t *bytes, size_t size) {
        std::array<char, 16384> chunk;
        for (size_t offset = 0; offset < size;) {
          const size_t length = std::min(chunk.size(), size - offset);
          if (!input.read(chunk.data(), length) ||
              std::memcmp(chunk.data(), bytes + offset, length))
            return false;
          offset += length;
        }
        return true;
      };
      if (!input || !matches(header.data(), header.size()) ||
          !matches(root.bytes.data(), root.bytes.size()) ||
          !matches(guard.data(), suffix_guard_bytes) ||
          input.peek() != std::char_traits<char>::eof() || input.bad())
        return Fail(error, "existing benchmark fixture content differs");
    } else {
      std::ofstream output(path, std::ios::binary);
      if (!output ||
          !output.write(reinterpret_cast<const char *>(header.data()),
                        header.size()) ||
          !output.write(reinterpret_cast<const char *>(root.bytes.data()),
                        root.bytes.size()) ||
          !output.write(reinterpret_cast<const char *>(guard.data()),
                        suffix_guard_bytes))
        return Fail(error, "cannot write benchmark fixture");
      output.close();
      if (!output)
        return Fail(error, "cannot close benchmark fixture");
    }
    root.fixture_path = relative;
    root.sha256 = digest;
    root.fixture_suffix_bytes = suffix_guard_bytes;
  }
  if (error)
    error->clear();
  return true;
}

} // namespace hrx::loomify_gguf::benchmark
