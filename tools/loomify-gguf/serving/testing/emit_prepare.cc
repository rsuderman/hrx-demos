// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/testing/fixture.h"
#include "tools/loomify-gguf/serving/package_internal.h"
#include <cstdio>
#include <fstream>
#include <sstream>
namespace {
template <class Values>
void ExpectedWords(std::ostream &out, const char *name, const Values &values) {
  out << "  %" << name << " = buffer.view %" << name
      << "_out[%origin] : buffer -> view<" << values.size() << "xi64>\n";
  for (size_t i = 0; i < values.size(); ++i) {
    out << "  %" << name << "_" << i << " = scalar.constant " << values[i]
        << " : i64\n  view.store %" << name << "_" << i << ", %" << name << "["
        << i << "] : i64, view<" << values.size() << "xi64>\n";
  }
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 4)
    return 1;
  using namespace hrx::loomify_gguf;
  ModelPackageOptions options;
  options.source_root = argv[1];
  LinkedConfigurationSettings settings;
  settings.token_capacity = options.prefill_capacity;
  settings.context_capacity = options.context_capacity;
  LinkedConfigurationFacts facts;
  std::string error;
  if (!DeriveLinkedConfiguration(testing::Fixture(), "asset", settings, &facts,
                                 &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  detail::CompanionSources sources;
  if (!detail::RenderLlamaCompanions(options, *facts.llama_layout, &sources,
                                     &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  // Preserve production bodies and only bind server imports to valid-input
  // callbacks in the VM component test, as in the Qwen preparation tests.
  for (const char *name : {"input", "prepare", "json"}) {
    const std::string token = std::string("import(\"") + name + "\") ";
    size_t pos = 0;
    while ((pos = sources.prepare.find(token, pos)) != std::string::npos)
      sources.prepare.erase(pos, token.size());
  }
  std::ofstream actual(argv[2]);
  actual << sources.prepare;
  actual.close();
  std::ofstream expected(argv[3]);
  expected
      << R"(func.def public @expected_prepare(%allocations_out: buffer, %views_out: buffer, %origins_out: buffer, %geometry_out: buffer, %state_out: buffer, %summary_out: buffer) {
  %origin = index.constant 0 : offset
)";
  const auto &layout = *facts.llama_layout;
  std::array<uint64_t, 33> allocations;
  for (size_t i = 0; i < layout.cold_allocations.size(); ++i) {
    allocations[3 * i] = layout.cold_allocations[i].bytes;
    allocations[3 * i + 1] = layout.cold_allocations[i].alignment;
    allocations[3 * i + 2] = layout.cold_allocations[i].clear_bytes;
  }
  ExpectedWords(expected, "allocations", allocations);
  ExpectedWords(expected, "views", layout.row_views);
  ExpectedWords(expected, "origins", layout.origins);
  ExpectedWords(expected, "geometry", layout.pooled_geometry);
  ExpectedWords(expected, "summary", std::array<uint64_t, 3>{128, 8, 0});
  expected
      << R"(  %state = buffer.view %state_out[%origin] : buffer -> view<2xi32>
  %prefill = scalar.constant 128 : i32
  %decode = scalar.constant 1 : i32
  view.store %prefill, %state[0] : i32, view<2xi32>
  view.store %decode, %state[1] : i32, view<2xi32>
  func.return
}
)";
  expected.close();
  return actual && expected ? 0 : 1;
}
