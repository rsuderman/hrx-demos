// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loom-model-inspect/architecture_report.h"

#include <limits>
#include <locale>
#include <sstream>

#include "gtest/gtest.h"

namespace hrx::model_inspect {
namespace {
std::string RowValue(const std::string &report, const std::string &label) {
  std::istringstream lines(report);
  std::string line;
  const auto prefix = "  " + label + " ";
  while (std::getline(lines, line)) {
    if (line.rfind(prefix, 0) == 0) {
      const auto begin = line.find_first_not_of(' ', prefix.size());
      return begin == std::string::npos ? "" : line.substr(begin);
    }
  }
  return "";
}
TEST(ArchitectureReport, PrintsOwnedModelRolesAndCompleteEscapedCatalog) {
  model::GgufCatalog catalog;
  catalog.name = "model\n\"name\"";
  catalog.tensors = {
      {"token\"embedding.weight", {8, 16, 1, 1}, 14, "q6_K", 32, 256},
      {"blk.0.ssm_a", {2, 1, 1, 1}, 0, "f32", 288, 8},
      {"extra\nweight", {1, 1, 1, 1}, 1, "f16", 296, 2},
  };
  model::ModelDescriptor descriptor;
  descriptor.family = "qwen35";
  descriptor.name = catalog.name;
  descriptor.embedding_width = 8;
  descriptor.vocabulary_size = 16;
  descriptor.output_tied_to_embedding = true;
  descriptor.globals = {{"token_embedding", catalog.tensors[0]},
                        {"output", catalog.tensors[0]}};
  model::LayerDescriptor layer;
  layer.kind = model::LayerKind::kGatedDeltaNet;
  layer.recurrent_ordinal = 0;
  layer.tensors = {{"decay", catalog.tensors[1]}};
  descriptor.layers.push_back(layer);
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Architecture"), "qwen35");
  EXPECT_EQ(RowValue(report, "Name"), "\"model\\x0a\\\"name\\\"\"");
  EXPECT_NE(RowValue(report, "Output").find("tied to embedding"),
            std::string::npos);
  EXPECT_NE(report.find("Layer 0 gated_delta_net"), std::string::npos);
  EXPECT_EQ(RowValue(report, "Recurrent ordinal"), "0 (logical)");
  EXPECT_NE(report.find("decay -> \"blk.0.ssm_a\""), std::string::npos);
  EXPECT_NE(report.find("token_embedding -> \"token\\\"embedding.weight\""),
            std::string::npos);
  EXPECT_NE(
      report.find("GGML extents [8,16,1,1]; storage q6_K (tag 14); encoded "
                  "bytes 256; file range [32,288) bytes, absolute"),
      std::string::npos);
  EXPECT_NE(report.find("output -> \"token\\\"embedding.weight\" (alias of "
                        "global.token_embedding; tied to embedding)"),
            std::string::npos);
  EXPECT_NE(report.find("tensor \"extra\\x0aweight\""), std::string::npos);
  EXPECT_NE(report.find("GGML extents [1,1,1,1]; storage f16 (tag 1); encoded "
                        "bytes 2; file range [296,298) bytes, absolute"),
            std::string::npos);
}

void SummaryFixture(model::GgufCatalog *catalog,
                    model::ModelDescriptor *descriptor, bool tied = false) {
  catalog->version = 3;
  catalog->name = "tiny Llama";
  catalog->architecture = "llama";
  catalog->data_offset = 32;
  catalog->bytes_read = 24;
  catalog->metadata.resize(2);
  catalog->tensors = {
      {"token_embd.weight", {8, 16, 1, 1}, 0, "f32", 32, 512},
      {"output_norm.weight", {8, 1, 1, 1}, 1, "f16", 544, 16},
  };
  if (!tied)
    catalog->tensors.push_back(
        {"output.weight", {8, 16, 1, 1}, 0, "f32", 560, 512});
  catalog->file_size = tied ? 560 : 1072;
  descriptor->family = "llama";
  descriptor->name = catalog->name;
  descriptor->embedding_width = 8;
  descriptor->vocabulary_size = 16;
  descriptor->training_context = 128;
  descriptor->key_head_width = 4;
  descriptor->value_head_width = 4;
  descriptor->rms_epsilon = double(float(1e-5));
  descriptor->rope.width = 4;
  descriptor->rope.original_context = 128;
  descriptor->output_tied_to_embedding = tied;
  descriptor->globals = {
      {"token_embedding", catalog->tensors[0]},
      {"output_norm", catalog->tensors[1]},
      {"output", catalog->tensors[tied ? 0 : 2]},
  };
  for (uint32_t i = 0; i < 2; ++i) {
    model::LayerDescriptor layer;
    layer.index = i;
    layer.attention_ordinal = i;
    layer.feed_forward_width = 16;
    layer.query_heads = 2;
    layer.kv_heads = 1;
    descriptor->layers.push_back(layer);
  }
}

TEST(ArchitectureSummary, UniformGeometryAndUniqueStorageFacts) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Layers"), "2 total / 2 attention / 0 recurrent");
  EXPECT_EQ(RowValue(report, "Embedding width"), "8");
  EXPECT_EQ(RowValue(report, "Feed-forward width"), "16 (uniform)");
  EXPECT_EQ(RowValue(report, "Attention heads"), "2 query / 1 KV (uniform)");
  EXPECT_EQ(RowValue(report, "Head widths"), "4 key / 4 value");
  EXPECT_EQ(RowValue(report, "Vocabulary"), "16 tokens");
  EXPECT_EQ(RowValue(report, "Training context"), "128 tokens");
  EXPECT_EQ(RowValue(report, "RMS epsilon"), "1e-05");
  EXPECT_EQ(RowValue(report, "Output"), "\"output.weight\" (separate)");
  EXPECT_EQ(RowValue(report, "GGUF version"), "3");
  EXPECT_EQ(RowValue(report, "Stored tensors"), "3");
  EXPECT_EQ(RowValue(report, "Metadata entries"), "2");
  EXPECT_EQ(RowValue(report, "File bytes"), "1072");
  EXPECT_EQ(RowValue(report, "Data offset"), "32 bytes, absolute");
  EXPECT_EQ(RowValue(report, "Encoded tensor bytes"),
            "1040 (sum of catalog entries)");
  EXPECT_EQ(RowValue(report, "f32 (tag 0)"), "2 tensors / 1024 bytes");
  EXPECT_EQ(RowValue(report, "f16 (tag 1)"), "1 tensors / 16 bytes");
  EXPECT_EQ(report.find("Recurrent geometry"), std::string::npos);
  EXPECT_TRUE(RowValue(report, "Sections").empty());
}

TEST(ArchitectureSummary, PerLayerGeometryDoesNotPretendToBeUniform) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  descriptor.layers[1].feed_forward_width = 24;
  descriptor.layers[1].query_heads = 4;
  descriptor.layers[1].kv_heads = 2;
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Feed-forward width"),
            "varies (min 16, max 24; see layers)");
  EXPECT_EQ(RowValue(report, "Attention heads"),
            "varies (2 query/KV configurations; see layers)");
}

TEST(ArchitectureSummary, RecurrentGeometryAndTiesAreReportedSeparately) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor, true);
  descriptor.family = "qwen35";
  descriptor.layers[0].kind = model::LayerKind::kGatedDeltaNet;
  descriptor.layers[0].attention_ordinal.reset();
  descriptor.layers[0].recurrent_ordinal = 0;
  descriptor.layers[0].query_heads = 999;
  descriptor.layers[0].kv_heads = 999;
  descriptor.rope.sections = {1, 1, 0, 0};
  descriptor.ssm = model::SsmDescriptor{4, 8, 4, 1, 2, 48, 32};
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Layers"), "2 total / 1 attention / 1 recurrent");
  EXPECT_EQ(RowValue(report, "Attention heads"), "2 query / 1 KV (uniform)");
  EXPECT_EQ(RowValue(report, "Sections"), "[1,1,0,0]");
  EXPECT_EQ(RowValue(report, "Heads"), "1 key / 2 value");
  EXPECT_EQ(RowValue(report, "Convolution kernel"), "4 taps");
  EXPECT_EQ(RowValue(report, "History"), "48 elements / layer / sequence");
  EXPECT_EQ(RowValue(report, "Matrix state"), "32 elements / layer / sequence");
  EXPECT_EQ(RowValue(report, "Output"),
            "\"token_embd.weight\" (tied to embedding)");
  EXPECT_EQ(RowValue(report, "Stored tensors"), "2");
  EXPECT_EQ(RowValue(report, "Encoded tensor bytes"),
            "528 (sum of catalog entries)");
  EXPECT_EQ(RowValue(report, "f32 (tag 0)"), "1 tensors / 512 bytes");
}

TEST(ArchitectureSummary, OmitsInapplicableAttentionOrSsmSections) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  descriptor.ssm = model::SsmDescriptor{};
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(report.find("Recurrent geometry"), std::string::npos);
  for (auto &layer : descriptor.layers) {
    layer.kind = model::LayerKind::kGatedDeltaNet;
    layer.attention_ordinal.reset();
    layer.recurrent_ordinal = layer.index;
  }
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Attention heads"),
            "not applicable (no attention layers)");
  EXPECT_TRUE(RowValue(report, "Head widths").empty());
  EXPECT_EQ(report.find("Position encoding"), std::string::npos);
}

TEST(ArchitectureSummary, MissingNameAndExactEscapedSourceLabel) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  descriptor.name.clear();
  const auto source = std::string(200, 'p') + "\n";
  std::string report, error;
  ASSERT_TRUE(
      FormatArchitecture(catalog, descriptor, &report, &error, {source, {}}))
      << error;
  EXPECT_EQ(RowValue(report, "Name"), "not provided");
  EXPECT_EQ(RowValue(report, "Source"),
            "\"" + std::string(200, 'p') + "\\x0a\"");
}

class CommaPunct : public std::numpunct<char> {
protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\3"; }
};

TEST(ArchitectureSummary, HumanFloatPrecisionAndIntegersAreLocaleIndependent) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  descriptor.rms_epsilon = 1.2345678e-27;
  descriptor.rope.base = 12345.5;
  catalog.file_size = 9007199254740993ULL;
  const auto old_locale = std::locale();
  std::locale::global(std::locale(std::locale::classic(), new CommaPunct));
  std::string report, error;
  const bool ok = FormatArchitecture(catalog, descriptor, &report, &error);
  std::locale::global(old_locale);
  ASSERT_TRUE(ok) << error;
  EXPECT_EQ(RowValue(report, "RMS epsilon"), "1.23457e-27");
  EXPECT_EQ(RowValue(report, "Base"), "12345.5");
  EXPECT_EQ(RowValue(report, "File bytes"), "9007199254740993");
}

TEST(ArchitectureSummary, RejectsTotalOverflowWithoutPublishingPartialReport) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  catalog.tensors[0].byte_size = std::numeric_limits<uint64_t>::max();
  catalog.tensors[0].absolute_offset = 0;
  std::string report = "unchanged", error;
  EXPECT_FALSE(FormatArchitecture(catalog, descriptor, &report, &error));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("overflows"), std::string::npos);
}

TEST(ArchitectureLayers, SemanticOrderKeepsUnknownRolesAndUsesCatalogFacts) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  const auto exact_name = std::string(220, 'x') + "\"\\\n" + char(0xff);
  catalog.tensors.push_back({exact_name, {8, 4, 1, 1}, 8, "q8_0", 1072, 34});
  auto copied = catalog.tensors.back();
  copied.storage_name = "stale";
  copied.normalized_extents = {99, 1, 1, 1};
  descriptor.layers[0].tensors = {
      {"ffn_down", copied}, {"new_role", copied},   {"key_norm", copied},
      {"query", copied},    {"input_norm", copied}, {"post_norm", copied}};
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  const auto begin = report.find("Layer 0 attention");
  const auto layer =
      report.substr(begin, report.find("Layer 1 attention") - begin);
  size_t previous = 0;
  for (const auto *role : {"input_norm", "query", "key_norm", "post_norm",
                           "ffn_down", "new_role"}) {
    const auto position = layer.find(std::string(role) + " -> ");
    ASSERT_NE(position, std::string::npos) << role;
    EXPECT_GE(position, previous) << role;
    EXPECT_EQ(layer.find(std::string(role) + " -> ", position + 1),
              std::string::npos);
    previous = position;
  }
  EXPECT_NE(layer.find("  Attention\n"), std::string::npos);
  EXPECT_NE(layer.find("  Feed-forward\n"), std::string::npos);
  EXPECT_NE(layer.find("  Other\n"), std::string::npos);
  EXPECT_NE(layer.find("GGML extents [8,4,1,1]; storage q8_0 (tag 8)"),
            std::string::npos);
  EXPECT_EQ(layer.find("stale"), std::string::npos);
  EXPECT_EQ(layer.find("[99,1,1,1]"), std::string::npos);
  const auto escaped = "\"" + std::string(220, 'x') + "\\\"\\\\\\x0a\\xff\"";
  EXPECT_NE(layer.find(escaped), std::string::npos);
  EXPECT_NE(layer.find("alias of layer 0.input_norm"), std::string::npos);
  const auto full = layer.find("GGML extents");
  EXPECT_EQ(layer.find("GGML extents", full + 1), std::string::npos);
}

TEST(ArchitectureLayers, SelectionUsesModelIndexAndFirstOccurrenceInShownView) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor, true);
  catalog.tensors.push_back({"shared", {8, 4, 1, 1}, 0, "f32", 560, 128});
  descriptor.layers[0].index = 7;
  descriptor.layers[0].tensors = {{"query", catalog.tensors.back()}};
  descriptor.layers[1].index = 42;
  descriptor.layers[1].attention_ordinal = 1;
  descriptor.layers[1].feed_forward_width = 24;
  descriptor.layers[1].query_heads = 4;
  descriptor.layers[1].kv_heads = 2;
  descriptor.layers[1].tensors = {{"key", catalog.tensors.back()}};
  std::string report, error;
  ASSERT_TRUE(
      FormatArchitecture(catalog, descriptor, &report, &error, {{}, 42}))
      << error;
  EXPECT_EQ(RowValue(report, "Layers"), "2 total / 2 attention / 0 recurrent");
  EXPECT_NE(report.find("Global tensors"), std::string::npos);
  EXPECT_NE(report.find("tied to embedding"), std::string::npos);
  EXPECT_NE(report.find("Layer details: 1 of 2"), std::string::npos);
  EXPECT_EQ(report.find("Layer 7 attention"), std::string::npos);
  const auto layer = report.substr(report.find("Layer 42 attention"));
  EXPECT_EQ(RowValue(layer, "Model index"), "42");
  EXPECT_EQ(RowValue(layer, "Attention ordinal"), "1 (logical)");
  EXPECT_EQ(RowValue(layer, "Feed-forward width"), "24");
  EXPECT_EQ(RowValue(layer, "Attention heads"), "4 query / 2 KV");
  EXPECT_NE(layer.find("key -> \"shared\"\n      GGML extents [8,4,1,1]"),
            std::string::npos);
  EXPECT_EQ(layer.find("alias of layer 7"), std::string::npos);
  EXPECT_EQ(layer.find("Unassigned tensors"), std::string::npos);
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_NE(report.find("key -> \"shared\" (alias of layer 7.query)"),
            std::string::npos);
  report = "unchanged";
  EXPECT_FALSE(
      FormatArchitecture(catalog, descriptor, &report, &error, {{}, 1}));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("unknown model layer index: 1"), std::string::npos);
}

TEST(ArchitectureLayers, RecurrentGeometryAndMissingOrdinalAreHonest) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  auto &layer = descriptor.layers[0];
  layer.kind = model::LayerKind::kGatedDeltaNet;
  layer.attention_ordinal.reset();
  layer.recurrent_ordinal = 0;
  layer.query_heads = 999;
  layer.kv_heads = 999;
  descriptor.ssm = model::SsmDescriptor{4, 8, 4, 1, 2, 48, 32};
  // Loader order deliberately differs from the semantic group order.
  for (const auto *role :
       {"ffn_up", "decay", "recurrent_output", "beta", "alpha", "recurrent_qkv",
        "input_norm", "recurrent_norm", "dt_bias", "convolution",
        "recurrent_gate"})
    layer.tensors.push_back({role, catalog.tensors[0]});
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error, {{}, 0}))
      << error;
  const auto detail = report.substr(report.find("Layer 0 gated_delta_net"));
  EXPECT_EQ(RowValue(detail, "Recurrent ordinal"), "0 (logical)");
  EXPECT_EQ(RowValue(detail, "Recurrent heads"), "1 key / 2 value");
  EXPECT_EQ(detail.find("999"), std::string::npos);
  EXPECT_EQ(detail.find("Attention heads"), std::string::npos);
  size_t previous = 0;
  for (const auto *role : {"input_norm", "recurrent_qkv", "recurrent_gate",
                           "alpha", "beta", "convolution", "dt_bias", "decay",
                           "recurrent_norm", "recurrent_output", "ffn_up"}) {
    const auto position = detail.find(std::string(role) + " -> ");
    ASSERT_NE(position, std::string::npos) << role;
    EXPECT_GE(position, previous) << role;
    previous = position;
  }
  layer.recurrent_ordinal.reset();
  descriptor.ssm.reset();
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error, {{}, 0}))
      << error;
  EXPECT_EQ(RowValue(report, "Recurrent ordinal"), "not provided");
  EXPECT_EQ(RowValue(report, "Recurrent geometry"), "not provided");
}

TEST(ArchitectureLayers, DanglingDisplayedBindingDoesNotPublishPartialReport) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  descriptor.layers[0].tensors.push_back({"query", {}});
  std::string report = "unchanged", error;
  EXPECT_FALSE(FormatArchitecture(catalog, descriptor, &report, &error));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("absent from catalog"), std::string::npos);
}

TEST(ArchitectureInventory, UniqueEntriesAllRolesAndUnassignedExtras) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor, true);
  catalog.tensors.push_back(
      {"shared", {8, 4, 1, 1}, 0, "f32", 4294967296ULL, 128});
  // Equal shapes and even equal spans do not make different names aliases.
  catalog.tensors.push_back(
      {"extra_a", {1, 1, 1, 1}, 1, "f16", 4294967424ULL, 2});
  catalog.tensors.push_back(
      {"extra_b", {1, 1, 1, 1}, 1, "f16", 4294967424ULL, 2});
  descriptor.layers[0].index = 7;
  descriptor.layers[0].tensors = {{"query", catalog.tensors[2]}};
  descriptor.layers[1].index = 42;
  descriptor.layers[1].tensors = {{"ffn_up", catalog.tensors[2]},
                                  {"key", catalog.tensors[2]}};
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_EQ(RowValue(report, "Stored tensors"), "5");
  EXPECT_EQ(RowValue(report, "Model-role references"), "6");
  EXPECT_EQ(RowValue(report, "Shown unique tensors"), "5");
  EXPECT_EQ(RowValue(report, "Shown encoded bytes"), "660");
  EXPECT_EQ(RowValue(report, "Shown model-role refs"), "6");
  EXPECT_NE(
      report.find(
          "References (model-wide): global.token_embedding; global.output"),
      std::string::npos);
  EXPECT_NE(report.find("References (model-wide): layer 7.query; layer 42.key; "
                        "layer 42.ffn_up"),
            std::string::npos);
  EXPECT_NE(report.find("file range [4294967296,4294967424) bytes, absolute"),
            std::string::npos);
  EXPECT_NE(report.find("tensor \"extra_a\""), std::string::npos);
  EXPECT_NE(report.find("tensor \"extra_b\""), std::string::npos);
  EXPECT_NE(report.find("References (model-wide): unassigned (no model role)"),
            std::string::npos);
  size_t position = 0, full_records = 0;
  while ((position = report.find("GGML extents [", position)) !=
         std::string::npos) {
    ++full_records;
    ++position;
  }
  EXPECT_EQ(full_records, 5);
  ReportOptions options;
  options.tensor_name = "shared";
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error, options))
      << error;
  EXPECT_EQ(RowValue(report, "Stored tensors"), "5");
  EXPECT_EQ(RowValue(report, "Model-role references"), "6");
  EXPECT_EQ(RowValue(report, "Shown unique tensors"), "1");
  EXPECT_EQ(RowValue(report, "Shown encoded bytes"), "128");
  EXPECT_EQ(RowValue(report, "Shown model-role refs"), "3");
  EXPECT_NE(report.find("tensor \"shared\""), std::string::npos);
  EXPECT_NE(report.find("References (model-wide): layer 7.query; layer 42.key; "
                        "layer 42.ffn_up"),
            std::string::npos);
  EXPECT_EQ(report.find("Global tensors"), std::string::npos);
  EXPECT_EQ(report.find("Layer details"), std::string::npos);
  options.tensor_name = "extra_a";
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error, options))
      << error;
  EXPECT_EQ(RowValue(report, "Shown model-role refs"), "0");
  EXPECT_NE(report.find("unassigned (no model role)"), std::string::npos);
  options.tensor_name.reset();
  options.layer_index = 42;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error, options))
      << error;
  EXPECT_EQ(RowValue(report, "Shown unique tensors"), "3");
  EXPECT_EQ(RowValue(report, "Shown encoded bytes"), "656");
  EXPECT_EQ(RowValue(report, "Shown model-role refs"), "6");
  EXPECT_NE(report.find("References (model-wide): layer 7.query; layer 42.key; "
                        "layer 42.ffn_up"),
            std::string::npos);
  EXPECT_EQ(report.find("extra_a"), std::string::npos);
}

TEST(ArchitectureInventory, TensorSelectionMatchesRawBytesIncludingEmptyName) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  const std::vector<std::string> names = {
      "", "x", "X", "x.more", " x", "x ", "\xc3\xa9", "x\n", "x\\x0a"};
  for (const auto &name : names)
    catalog.tensors.push_back({name, {1, 1, 1, 1}, 0, "f32", 0, 4});
  std::string report, error;
  ReportOptions options;
  for (const auto &name : names) {
    options.tensor_name = name;
    ASSERT_TRUE(
        FormatArchitecture(catalog, descriptor, &report, &error, options))
        << error;
    EXPECT_EQ(RowValue(report, "Stored tensors"), "9");
    EXPECT_EQ(RowValue(report, "Shown unique tensors"), "1");
    EXPECT_EQ(RowValue(report, "Shown encoded bytes"), "4");
    EXPECT_EQ(RowValue(report, "Shown model-role refs"), "0");
  }
  options.tensor_name = "x.*";
  report = "unchanged";
  EXPECT_FALSE(
      FormatArchitecture(catalog, descriptor, &report, &error, options));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("unknown tensor name"), std::string::npos);
  options.tensor_name = "x";
  options.layer_index = 0;
  EXPECT_FALSE(
      FormatArchitecture(catalog, descriptor, &report, &error, options));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("mutually exclusive"), std::string::npos);
}

TEST(ArchitectureInventory, DuplicateOrMissingCatalogIdentityRejectsAllViews) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  catalog.tensors.push_back(catalog.tensors[0]);
  std::string report = "unchanged", error;
  EXPECT_FALSE(FormatArchitecture(catalog, descriptor, &report, &error));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("duplicate catalog tensor name"), std::string::npos);
  catalog.tensors.pop_back();
  descriptor.layers[1].tensors.push_back({"hidden", {}});
  EXPECT_FALSE(
      FormatArchitecture(catalog, descriptor, &report, &error, {{}, 0, {}}));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("absent from catalog"), std::string::npos);
}

TEST(ArchitectureInventory, CheckedRangeEndsAndExactUint64Offsets) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  catalog.tensors = {{"at_end",
                      {1, 1, 1, 1},
                      0,
                      "f32",
                      std::numeric_limits<uint64_t>::max() - 4,
                      4}};
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_NE(report.find(
                "[18446744073709551611,18446744073709551615) bytes, absolute"),
            std::string::npos);
  catalog.tensors[0].byte_size = 5;
  report = "unchanged";
  EXPECT_FALSE(FormatArchitecture(catalog, descriptor, &report, &error));
  EXPECT_EQ(report, "unchanged");
  EXPECT_NE(error.find("range end overflows"), std::string::npos);
}
} // namespace
} // namespace hrx::model_inspect

namespace hrx::model_inspect {
namespace {
TEST(ArchitectureLayers, ShortConvolutionHasOwnKindAndCompactOrdinal) {
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  SummaryFixture(&catalog, &descriptor);
  auto &layer = descriptor.layers[0];
  layer.kind = model::LayerKind::kShortConvolution;
  layer.attention_ordinal.reset();
  layer.convolution_ordinal = 0;
  descriptor.short_convolution = model::ShortConvolutionDescriptor{2048, 7, 6};
  layer.tensors.push_back({"convolution_input", catalog.tensors[0]});
  std::string report, error;
  ASSERT_TRUE(FormatArchitecture(catalog, descriptor, &report, &error))
      << error;
  EXPECT_NE(report.find("1 short convolution"), std::string::npos);
  EXPECT_NE(report.find("short_convolution"), std::string::npos);
  EXPECT_EQ(RowValue(report, "Convolution ordinal"), "0 (logical)");
  EXPECT_EQ(RowValue(report, "Convolution geometry"),
            "2048 channels / 7 taps / 6 history rows");
  EXPECT_EQ(report.find("gated_delta_net"), std::string::npos);
  EXPECT_NE(report.find("Short convolution"), std::string::npos);
}
} // namespace
} // namespace hrx::model_inspect
