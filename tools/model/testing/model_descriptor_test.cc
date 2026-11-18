// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor.h"
#include "tools/model/tensor_names.h"
#include "tools/model/testing/f32_gguf_fixture.h"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
namespace hrx::model {
namespace {
void Scalar(GgufCatalog &c, const std::string &key, MetadataScalar value,
            uint32_t type = 4) {
  c.metadata.push_back({key, type, std::move(value)});
}
void Array(GgufCatalog &c, const std::string &key,
           std::vector<MetadataScalar> values, uint32_t type = 4) {
  MetadataRecord f;
  f.key = key;
  f.source_type = 9;
  f.element_type = type;
  f.element_count = values.size();
  f.elements = std::move(values);
  c.metadata.push_back(std::move(f));
}
void Tensor(GgufCatalog &c, const std::string &name,
            std::initializer_list<int64_t> shape) {
  TensorRecord t;
  t.name = name;
  std::copy(shape.begin(), shape.end(), t.normalized_extents.begin());
  t.storage_name = "f32";
  t.absolute_offset = 32;
  t.byte_size = 4;
  for (auto n : t.normalized_extents)
    t.byte_size *= n;
  c.tensors.push_back(t);
}
GgufCatalog Model(bool qwen = false, bool tie = false) {
  GgufCatalog c;
  c.architecture = qwen ? "qwen35" : "llama";
  c.name = "tiny structural fixture";
  const auto p = c.architecture + ".";
  for (const auto &item : std::vector<std::pair<std::string, uint64_t>>{
           {"block_count", 2},
           {"embedding_length", 8},
           {"context_length", 128},
           {"feed_forward_length", 16},
           {"attention.head_count", 2},
           {"attention.head_count_kv", 1},
           {"attention.key_length", 4},
           {"attention.value_length", 4},
           {"vocab_size", 16}})
    Scalar(c, p + item.first, item.second);
  Scalar(c, p + "attention.layer_norm_rms_epsilon", 1e-5, 6);
  Tensor(c, "token_embd.weight", {8, 16});
  Tensor(c, "output_norm.weight", {8});
  if (!tie)
    Tensor(c, "output.weight", {8, 16});
  if (qwen) {
    for (const auto &item : std::vector<std::pair<std::string, uint64_t>>{
             {"ssm.conv_kernel", 4},
             {"ssm.inner_size", 8},
             {"ssm.state_size", 4},
             {"ssm.group_count", 1},
             {"ssm.time_step_rank", 2}})
      Scalar(c, p + item.first, item.second);
    Array(c, p + "rope.dimension_sections",
          {uint64_t(1), uint64_t(1), uint64_t(0), uint64_t(0)});
    Array(c, p + "attention.recurrent_layers", {true, false}, 7);
  }
  for (int i = 0; i < 2; ++i) {
    auto b = "blk." + std::to_string(i) + ".";
    Tensor(c, b + "attn_norm.weight", {8});
    Tensor(c, b + (qwen ? "post_attention_norm.weight" : "ffn_norm.weight"),
           {8});
    Tensor(c, b + "ffn_gate.weight", {8, 16});
    Tensor(c, b + "ffn_up.weight", {8, 16});
    Tensor(c, b + "ffn_down.weight", {16, 8});
    if (qwen && i == 0) {
      Tensor(c, b + "attn_qkv.weight", {8, 16});
      Tensor(c, b + "attn_gate.weight", {8, 8});
      Tensor(c, b + "ssm_conv1d.weight", {4, 16});
      Tensor(c, b + "ssm_dt.bias", {2});
      Tensor(c, b + "ssm_a", {2});
      Tensor(c, b + "ssm_beta.weight", {8, 2});
      Tensor(c, b + "ssm_alpha.weight", {8, 2});
      Tensor(c, b + "ssm_norm.weight", {4});
      Tensor(c, b + "ssm_out.weight", {8, 8});
    } else {
      Tensor(c, b + "attn_q.weight", {8, qwen ? 16 : 8});
      Tensor(c, b + "attn_k.weight", {8, 4});
      Tensor(c, b + "attn_v.weight", {8, 4});
      Tensor(c, b + "attn_output.weight", {8, 8});
      if (qwen) {
        Tensor(c, b + "attn_q_norm.weight", {4});
        Tensor(c, b + "attn_k_norm.weight", {4});
      }
    }
  }
  return c;
}
MetadataRecord &Field(GgufCatalog &c, const std::string &suffix) {
  return *std::find_if(c.metadata.begin(), c.metadata.end(), [&](auto &f) {
    return f.key == c.architecture + "." + suffix;
  });
}
void Reject(const GgufCatalog &c, const std::string &message) {
  ModelDescriptor m;
  m.name = "unchanged";
  std::string error;
  EXPECT_FALSE(LoadModelDescriptor(c, &m, &error));
  EXPECT_EQ(m.name, "unchanged");
  EXPECT_NE(error.find(message), std::string::npos) << error;
}
TEST(ModelDescriptor, DenseLlamaAndOwnedDtypeBindings) {
  auto c = Model();
  c.tensors[0].storage_type = 14;
  c.tensors[0].storage_name = "q6_k";
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_EQ(m.layers.size(), 2);
  EXPECT_EQ(m.layers[1].attention_ordinal, 1);
  EXPECT_FALSE(m.output_tied_to_embedding);
  EXPECT_EQ(m.rope.width, 4);
  EXPECT_EQ(m.rope.base, 10000);
  EXPECT_EQ(m.rope.original_context, 128);
  EXPECT_EQ(m.globals[0].tensor.storage_type, 14);
  c.tensors.clear();
  EXPECT_EQ(m.globals[0].tensor.name, "token_embd.weight");
}
TEST(ModelDescriptor, QwenHybridStateAndExplicitMapPrecedence) {
  auto c = Model(true, true);
  Scalar(c, "qwen35.full_attention_interval", uint64_t(0));
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_TRUE(m.output_tied_to_embedding);
  EXPECT_EQ(m.globals[2].tensor.name, "token_embd.weight");
  EXPECT_EQ(m.layers[0].kind, LayerKind::kGatedDeltaNet);
  EXPECT_EQ(m.layers[0].recurrent_ordinal, 0);
  EXPECT_EQ(m.layers[1].attention_ordinal, 0);
  EXPECT_EQ(m.ssm->convolution_history_elements, 48);
  EXPECT_EQ(m.ssm->recurrent_state_elements, 32);
  EXPECT_EQ(m.stored_layer_count, 2);
}
GgufCatalog QwenWithAuxiliaryTail(uint64_t stored = 3, uint64_t auxiliary = 1) {
  auto c = Model(true);
  const auto prototype = c.tensors;
  c.tensors.erase(std::remove_if(c.tensors.begin(), c.tensors.end(),
                                 [](const auto &t) {
                                   return t.name.rfind("blk.", 0) == 0;
                                 }),
                  c.tensors.end());
  Field(c, "block_count").scalar = stored;
  Scalar(c, "qwen35.nextn_predict_layers", auxiliary);
  auto &map = Field(c, "attention.recurrent_layers");
  map.element_count = stored;
  map.elements.clear();
  for (uint64_t i = 0; i < stored; ++i) {
    const bool recurrent = i % 4 != 3;
    map.elements.push_back(recurrent);
    if (i >= stored - auxiliary)
      continue;
    const std::string prefix = recurrent ? "blk.0." : "blk.1.";
    for (auto tensor : prototype)
      if (tensor.name.rfind(prefix, 0) == 0) {
        tensor.name = "blk." + std::to_string(i) + "." +
                      tensor.name.substr(prefix.size());
        tensor.absolute_offset = 4096 + i * 8192;
        c.tensors.push_back(std::move(tensor));
      }
  }
  // The excluded tail need not expose decoder roles. Keep opaque graph data
  // in the catalog instead of fabricating a stripped replacement checkpoint.
  Tensor(c, "blk." + std::to_string(stored - 1) + ".nextn.eh_proj.weight",
         {16, 8});
  return c;
}
TEST(ModelDescriptor, QwenSelectsOrdinaryPrefixAndRetainsStoredIdentities) {
  auto c = QwenWithAuxiliaryTail(65);
  const auto catalog_size = c.tensors.size();
  const auto untouched_tail = c.tensors.back();
  Tensor(c, "blk.64.nextn.optional.bias", {8});
  Tensor(c, "blk.64.nextn.optional.scale", {8});
  ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &error)) << error;
  ASSERT_EQ(m.layers.size(), 64);
  EXPECT_EQ(m.stored_layer_count, 65);
  size_t recurrent = 0, attention = 0;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &layer = m.layers[i];
    EXPECT_EQ(layer.index, i);
    if (layer.kind == LayerKind::kGatedDeltaNet)
      EXPECT_EQ(layer.recurrent_ordinal, recurrent++);
    else
      EXPECT_EQ(layer.attention_ordinal, attention++);
    for (const auto &binding : layer.tensors) {
      EXPECT_EQ(binding.tensor.name.rfind("blk." + std::to_string(i) + ".", 0),
                0);
      EXPECT_EQ(binding.tensor.absolute_offset, 4096 + i * 8192);
    }
  }
  EXPECT_EQ(recurrent, 48);
  EXPECT_EQ(attention, 16);
  EXPECT_EQ(c.tensors.size(), catalog_size + 2);
  EXPECT_EQ(c.tensors[catalog_size - 1].name, untouched_tail.name);
  EXPECT_EQ(c.tensors[catalog_size - 1].absolute_offset,
            untouched_tail.absolute_offset);
}
TEST(ModelDescriptor, QwenTailValidatesStoredDimensionAndRecurrentArrays) {
  auto c = QwenWithAuxiliaryTail();
  auto &ff = Field(c, "feed_forward_length");
  ff.source_type = 9;
  ff.element_count = 3;
  ff.elements = {uint64_t(16), uint64_t(16), uint64_t(48)};
  ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &error)) << error;
  EXPECT_EQ(m.layers.size(), 2);
  EXPECT_EQ(m.layers.back().feed_forward_width, 16);
  ff.element_count = 2;
  ff.elements.pop_back();
  Reject(c, "array length");
  c = QwenWithAuxiliaryTail();
  Field(c, "attention.recurrent_layers").element_count = 2;
  Field(c, "attention.recurrent_layers").elements.pop_back();
  Reject(c, "recurrent array length");
  c = QwenWithAuxiliaryTail();
  Field(c, "attention.recurrent_layers").elements.back() = uint64_t(2);
  Reject(c, "boolean recurrent map");
}
TEST(ModelDescriptor, QwenTailFailuresDoNotPublishPartialDescriptor) {
  auto c = QwenWithAuxiliaryTail();
  Field(c, "nextn_predict_layers").scalar = uint64_t(3);
  Reject(c, "ordinary layer");
  Field(c, "nextn_predict_layers").scalar = uint64_t(4);
  Reject(c, "ordinary layer");
  Field(c, "nextn_predict_layers").scalar = int64_t(-1);
  Reject(c, "integer");
  c = QwenWithAuxiliaryTail();
  c.tensors.erase(std::remove_if(c.tensors.begin(), c.tensors.end(),
                                 [](const auto &t) {
                                   return t.name == "blk.1.ffn_down.weight";
                                 }),
                  c.tensors.end());
  Reject(c, "blk.1.ffn_down.weight");
  c = QwenWithAuxiliaryTail();
  Tensor(c, "blk.1.unknown.bias", {8});
  Reject(c, "optional bias");
  c = QwenWithAuxiliaryTail();
  Tensor(c, "blk.3.unknown.scale", {8});
  Reject(c, "optional scale");
  c = Model();
  Scalar(c, "llama.nextn_predict_layers", uint64_t(1));
  Reject(c, "MoE/MTP");
}
TEST(ModelDescriptor, FallbackAndMalformedRecurrentMap) {
  auto c = Model(true);
  c.metadata.erase(std::remove_if(c.metadata.begin(), c.metadata.end(),
                                  [](auto &f) {
                                    return f.key ==
                                           "qwen35.attention.recurrent_layers";
                                  }),
                   c.metadata.end());
  Scalar(c, "qwen35.full_attention_interval", uint64_t(2));
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  Field(c, "full_attention_interval").scalar = uint64_t(0);
  Reject(c, "geometry");
  c = Model(true);
  Field(c, "attention.recurrent_layers").elements.pop_back();
  Reject(c, "length");
  c = Model(true);
  Field(c, "attention.recurrent_layers").elements[0] = double(1);
  Reject(c, "integer");
}
TEST(ModelDescriptor, PerLayerDimensionsAndDefaults) {
  auto c = Model();
  Field(c, "feed_forward_length").source_type = 9;
  Field(c, "feed_forward_length").element_count = 2;
  Field(c, "feed_forward_length").elements = {uint64_t(16), uint64_t(24)};
  for (auto &t : c.tensors)
    if (t.name == "blk.1.ffn_gate.weight" || t.name == "blk.1.ffn_up.weight")
      t.normalized_extents[1] = 24;
    else if (t.name == "blk.1.ffn_down.weight")
      t.normalized_extents[0] = 24;
  c.metadata.erase(std::remove_if(c.metadata.begin(), c.metadata.end(),
                                  [](auto &f) {
                                    return f.key ==
                                           "llama.attention.head_count_kv";
                                  }),
                   c.metadata.end());
  for (auto &t : c.tensors)
    if (t.name.find("attn_k.weight") != std::string::npos ||
        t.name.find("attn_v.weight") != std::string::npos)
      t.normalized_extents[1] = 8;
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_EQ(m.layers[1].feed_forward_width, 24);
  EXPECT_EQ(m.layers[0].kv_heads, 2);
}
TEST(ModelDescriptor, BadMetadataGeometryAndShapeRollback) {
  auto c = Model();
  c.metadata.erase(c.metadata.begin());
  Reject(c, "missing required");
  c = Model();
  Field(c, "embedding_length").scalar = std::string("8");
  Reject(c, "integer");
  c = Model();
  Field(c, "attention.layer_norm_rms_epsilon").scalar =
      std::numeric_limits<double>::infinity();
  Reject(c, "finite");
  c = Model();
  Field(c, "attention.value_length").scalar = uint64_t(2);
  Reject(c, "unequal");
  c = Model();
  c.tensors[0].normalized_extents[3] = 2;
  Reject(c, "shape mismatch");
  c = Model();
  Field(c, "attention.head_count_kv").scalar = uint64_t(3);
  Reject(c, "divisible");
  c = Model();
  Field(c, "feed_forward_length").source_type = 9;
  Reject(c, "array length");
}
TEST(ModelDescriptor, UnsupportedVariantsRemainCatalogFacts) {
  auto c = Model();
  c.architecture = "qwen3moe";
  Reject(c, "unsupported architecture");
  c = Model();
  Scalar(c, "llama.expert_count", uint64_t(8));
  Reject(c, "MoE/MTP");
  c = Model();
  Scalar(c, "llama.rope.scaling.type", std::string("yarn"), 8);
  Reject(c, "RoPE scaling");
  c = Model();
  Tensor(c, "blk.0.attn_qkv.weight", {8, 16});
  Reject(c, "fused attention");
  c = Model();
  Tensor(c, "blk.0.attn_q.bias", {8});
  Reject(c, "optional bias");
}
TEST(ModelDescriptor, ExplicitHeadWidthsAndPreferredRopeFactor) {
  auto c = Model();
  Field(c, "embedding_length").scalar = uint64_t(7);
  // Explicit widths define independent projected dimensions, even if H%Q!=0.
  for (auto &t : c.tensors)
    for (auto &n : t.normalized_extents)
      if (n == 8)
        n = 7;
  // Restore query/output projected widths (4*2), and down FFN input16.
  for (auto &t : c.tensors) {
    if (t.name.find("attn_q.weight") != std::string::npos)
      t.normalized_extents[1] = 8;
    if (t.name.find("attn_output.weight") != std::string::npos)
      t.normalized_extents[0] = 8;
  }
  Scalar(c, "llama.rope.scaling.factor", double(2), 6);
  Scalar(c, "llama.rope.scale_linear", std::string("ignored legacy"), 8);
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_EQ(m.rope.scaling_factor, 2);
  EXPECT_EQ(m.key_head_width, 4);
  Tensor(c, "blk.0.attn_q.scale", {1});
  Reject(c, "optional scale");
}
TEST(ModelDescriptor, PerLayerHeadArraysAndScalarRecurrence) {
  auto c = Model();
  auto &heads = Field(c, "attention.head_count");
  heads.source_type = 9;
  heads.element_count = 2;
  heads.elements = {uint64_t(2), uint64_t(4)};
  auto &kv = Field(c, "attention.head_count_kv");
  kv.source_type = 9;
  kv.element_count = 2;
  kv.elements = {uint64_t(1), uint64_t(2)};
  for (auto &t : c.tensors) {
    if (t.name == "blk.1.attn_q.weight")
      t.normalized_extents[1] = 16;
    if (t.name == "blk.1.attn_k.weight" || t.name == "blk.1.attn_v.weight")
      t.normalized_extents[1] = 8;
    if (t.name == "blk.1.attn_output.weight")
      t.normalized_extents[0] = 16;
  }
  ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_EQ(m.layers[1].query_heads, 4);
  EXPECT_EQ(m.layers[1].kv_heads, 2);
  c = Model(true);
  auto &map = Field(c, "attention.recurrent_layers");
  map.source_type = 7;
  map.scalar = true;
  map.elements.clear();
  c.tensors.erase(
      std::remove_if(c.tensors.begin(), c.tensors.end(),
                     [](auto &t) { return t.name.rfind("blk.1.", 0) == 0; }),
      c.tensors.end());
  auto originals = c.tensors;
  for (auto t : originals)
    if (t.name.rfind("blk.0.", 0) == 0) {
      t.name.replace(0, 6, "blk.1.");
      c.tensors.push_back(std::move(t));
    }
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &e)) << e;
  EXPECT_EQ(m.layers[1].recurrent_ordinal, 1);
}
void Number(std::string &s, uint64_t n, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    s.push_back(char(n >> (8 * i)));
}
void String(std::string &s, const std::string &v) {
  Number(s, v.size(), 8);
  s += v;
}
TEST(GgufCatalog, CompleteTypedSparseHeaderAndNoPayloadRead) {
  std::string s = "GGUF";
  Number(s, 3, 4);
  Number(s, 40, 8);
  Number(s, 74, 8);
  String(s, "general.architecture");
  Number(s, 8, 4);
  String(s, "unimplemented-inspectable");
  for (int i = 0; i < 70; ++i) {
    String(s,
           (i == 69 ? std::string(190, 'm') : "metadata.") + std::to_string(i));
    Number(s, i == 0 ? 6 : i == 1 ? 5 : 4, 4);
    Number(s, i == 0 ? 0x3fc00000 : i == 1 ? uint32_t(-5) : i, 4);
  }
  String(s, "test.boolean.array");
  Number(s, 9, 4);
  Number(s, 7, 4);
  Number(s, 2, 8);
  Number(s, 1, 1);
  Number(s, 0, 1);
  String(s, "tokenizer.ggml.tokens");
  Number(s, 9, 4);
  Number(s, 8, 4);
  Number(s, 2, 8);
  String(s, "secret");
  String(s, "token");
  String(s, "architecture.integer.array");
  Number(s, 9, 4);
  Number(s, 5, 4);
  Number(s, 2, 8);
  Number(s, uint32_t(-1), 4);
  Number(s, 7, 4);
  uint64_t offset = 0;
  for (int i = 0; i < 40; ++i) {
    String(s, std::string(55, 'a') + std::to_string(i));
    Number(s, 2, 4);
    Number(s, 256, 8);
    Number(s, 2, 8);
    const uint32_t types[] = {0, 1, 12, 14};
    Number(s, types[i % 4], 4);
    Number(s, offset, 8);
    const uint64_t sizes[] = {2048, 1024, 288, 420};
    offset += (sizes[i % 4] + 31) / 32 * 32;
  }
  while (s.size() % 32)
    s.push_back(0);
  auto data_offset = s.size();
  std::unique_ptr<FILE, decltype(&fclose)> f(tmpfile(), fclose);
  ASSERT_TRUE(f);
  ASSERT_EQ(setvbuf(f.get(), nullptr, _IONBF, 0), 0);
  ASSERT_EQ(fwrite(s.data(), 1, s.size(), f.get()), s.size());
  ASSERT_EQ(fseeko(f.get(), data_offset + offset - 1, SEEK_SET), 0);
  ASSERT_NE(fputc(0, f.get()), EOF);
  GgufCatalog c;
  std::string e;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &c, &e)) << e;
  EXPECT_EQ(c.metadata.size(), 74);
  EXPECT_EQ(c.tensors.size(), 40);
  EXPECT_EQ(c.tensors[39].name, std::string(55, 'a') + "39");
  EXPECT_EQ(c.tensors[3].storage_type, 14);
  EXPECT_EQ(c.tensors[3].storage_name, "q6_K");
  EXPECT_EQ(c.tensors[2].byte_size, 288);
  EXPECT_EQ(c.tensors[1].normalized_extents,
            (std::array<int64_t, 4>{256, 2, 1, 1}));
  EXPECT_EQ(std::get<double>(FindMetadata(c, "metadata.0")->scalar), 1.5);
  EXPECT_EQ(std::get<int64_t>(FindMetadata(c, "metadata.1")->scalar), -5);
  const auto *integers = FindMetadata(c, "architecture.integer.array");
  ASSERT_TRUE(integers);
  EXPECT_FALSE(integers->elements_omitted);
  EXPECT_EQ(integers->element_type, 5);
  EXPECT_EQ(std::get<int64_t>(integers->elements[0]), -1);
  auto *array = FindMetadata(c, "test.boolean.array");
  ASSERT_TRUE(array);
  EXPECT_EQ(array->element_type, 7);
  EXPECT_TRUE(std::get<bool>(array->elements[0]));
  auto *tokens = FindMetadata(c, "tokenizer.ggml.tokens");
  EXPECT_TRUE(tokens->elements_omitted);
  EXPECT_EQ(tokens->element_count, 2);
  EXPECT_EQ(FindMetadata(c, std::string(190, 'm') + "69")->source_type, 4);
  EXPECT_EQ(std::get<uint64_t>(FindMetadata(c, "metadata.68")->scalar), 68);
  EXPECT_LE(c.bytes_read, data_offset + 24);
  // A second failed read cannot publish a partial catalog.
  ASSERT_EQ(fseeko(f.get(), 0, SEEK_SET), 0);
  ASSERT_NE(fputc('X', f.get()), EOF);
  c.name = "unchanged";
  EXPECT_FALSE(ReadGgufCatalog(f.get(), &c, &e));
  EXPECT_EQ(c.name, "unchanged");
  ModelDescriptor model;
  EXPECT_FALSE(LoadModelDescriptor(c, &model, &e));
}
} // namespace

TEST(LlamaConfiguration, RejectsExactUnsupportedNormsWithoutChangingOutput) {
  for (const auto &name :
       {"blk.0.attn_q_norm.weight", "blk.123.attn_k_norm.weight"}) {
    auto catalog = Model();
    Tensor(catalog, name, {4});
    ModelDescriptor output;
    output.name = "sentinel";
    std::string error;
    EXPECT_FALSE(LoadModelDescriptor(catalog, &output, &error));
    EXPECT_EQ(output.name, "sentinel");
    EXPECT_NE(error.find("Q/K normalization"), std::string::npos);
  }
  auto catalog = Model();
  Tensor(catalog, "blk.x.attn_q_norm.weight", {4});
  ModelDescriptor output;
  std::string error;
  EXPECT_TRUE(LoadModelDescriptor(catalog, &output, &error)) << error;
}
TEST(LlamaConfiguration,
     ExtractedFactsOutliveBorrowedFileAndPreserveTiedWeights) {
  auto catalog = Model(false, true);
  std::string bytes = "GGUF";
  auto number = [&](uint64_t n, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
      bytes.push_back(char(n >> (i * 8)));
  };
  auto text = [&](const std::string &s) {
    number(s.size(), 8);
    bytes += s;
  };
  number(3, 4);
  number(catalog.tensors.size(), 8);
  number(catalog.metadata.size() + 1, 8);
  text("general.architecture");
  number(8, 4);
  text("llama");
  for (const auto &field : catalog.metadata) {
    text(field.key);
    number(field.source_type, 4);
    if (const auto *v = std::get_if<uint64_t>(&field.scalar))
      number(*v, 4);
    else if (const auto *v = std::get_if<double>(&field.scalar)) {
      float value = *v;
      uint32_t bits;
      std::memcpy(&bits, &value, 4);
      number(bits, 4);
    } else
      FAIL() << "unexpected fixture metadata";
  }
  uint64_t relative = 0;
  for (const auto &tensor : catalog.tensors) {
    text(tensor.name);
    auto rank = tensor.normalized_extents[1] == 1 ? 1u : 2u;
    number(rank, 4);
    for (unsigned i = 0; i < rank; ++i)
      number(tensor.normalized_extents[i], 8);
    number(0, 4);
    number(relative, 8);
    relative += (tensor.byte_size + 31) / 32 * 32;
  }
  while (bytes.size() % 32)
    bytes.push_back(0);
  bytes.append(relative, 0);
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  ASSERT_EQ(std::fwrite(bytes.data(), 1, bytes.size(), file.get()),
            bytes.size());
  ModelDescriptor extracted;
  std::string error;
  GgufCatalog parsed;
  ASSERT_TRUE(ReadGgufCatalog(file.get(), &parsed, &error)) << error;
  ASSERT_TRUE(LoadModelDescriptor(parsed, &extracted, &error)) << error;
  EXPECT_EQ(parsed.architecture, "llama");
  EXPECT_EQ(std::fseek(file.get(), 0, SEEK_SET),
            0); // Extractor does not close it.
  file.reset();
  catalog = {};
  EXPECT_EQ(extracted.family, "llama");
  EXPECT_EQ(extracted.layers.size(), 2u);
  EXPECT_TRUE(extracted.output_tied_to_embedding);
  auto saved = extracted;
  std::unique_ptr<FILE, decltype(&std::fclose)> bad(std::tmpfile(),
                                                    &std::fclose);
  ASSERT_TRUE(bad);
  ASSERT_EQ(std::setvbuf(bad.get(), nullptr, _IONBF, 0), 0);
  EXPECT_FALSE(ReadGgufCatalog(bad.get(), &parsed, &error));
  EXPECT_EQ(extracted.globals.size(), saved.globals.size());
  EXPECT_EQ(extracted.globals[0].tensor.name, saved.globals[0].tensor.name);
}
TEST(ModelDescriptor, F32PayloadRangesAgreeWithNativeNamedIndex) {
  for (const uint32_t width : {4, 8}) {
    GgufCatalog catalog;
    ModelDescriptor descriptor;
    std::string error;
    const auto bytes = testing::EmbeddingGguf(width, 3);
    ASSERT_TRUE(testing::ReadFixture(bytes, &catalog, &descriptor, &error))
        << error;
    ASSERT_EQ(descriptor.family, "llama");
    EXPECT_EQ(descriptor.embedding_width, width);
    EXPECT_EQ(descriptor.vocabulary_size, 3u);
    const auto *tensor = FindTensor(catalog, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    EXPECT_GT(tensor->absolute_offset, catalog.data_offset);
    EXPECT_EQ(tensor->byte_size, width * 3 * sizeof(float));
    std::vector<float> payload;
    ASSERT_TRUE(
        testing::ReadNamedF32(bytes, "token_embd.weight", &payload, &error))
        << error;
    ASSERT_EQ(payload.size(), width * 3);
    EXPECT_EQ(payload[width], 110);
    EXPECT_EQ(payload[2 * width + width - 1], 210 + width - 1);
  }
}
TEST(ModelDescriptorQwen3, OrdinaryLayerHasIndependentQueryWidthAndHeadNorms) {
  auto c = Model(false, true);
  c.architecture = "qwen3";
  for (auto &metadata : c.metadata)
    if (metadata.key.rfind("llama.", 0) == 0)
      metadata.key.replace(0, 6, "qwen3.");
  Field(c, "attention.head_count").scalar = uint64_t{4};
  Field(c, "attention.head_count_kv").scalar = uint64_t{2};
  for (auto &tensor : c.tensors) {
    if (tensor.name.find("attn_q.weight") != std::string::npos)
      tensor.normalized_extents = {8, 16, 1, 1};
    if (tensor.name.find("attn_k.weight") != std::string::npos ||
        tensor.name.find("attn_v.weight") != std::string::npos)
      tensor.normalized_extents = {8, 8, 1, 1};
    if (tensor.name.find("attn_output.weight") != std::string::npos)
      tensor.normalized_extents = {16, 8, 1, 1};
  }
  for (int layer = 0; layer < 2; ++layer) {
    const auto p = "blk." + std::to_string(layer) + ".";
    Tensor(c, p + "attn_q_norm.weight", {4});
    Tensor(c, p + "attn_k_norm.weight", {4});
  }
  ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(LoadModelDescriptor(c, &m, &error)) << error;
  EXPECT_EQ(m.family, "qwen3");
  EXPECT_EQ(m.embedding_width, 8u);
  EXPECT_EQ(m.layers[0].query_heads * m.key_head_width, 16u);
  EXPECT_EQ(m.layers[0].kv_heads * m.value_head_width, 8u);
  EXPECT_TRUE(m.output_tied_to_embedding);
  EXPECT_FALSE(m.ssm);
  EXPECT_EQ(m.layers[0].tensors.size(), 11u);
  const auto norm = std::find_if(
      m.layers[0].tensors.begin(), m.layers[0].tensors.end(),
      [](const auto &binding) { return binding.role == "post_norm"; });
  ASSERT_NE(norm, m.layers[0].tensors.end());
  EXPECT_EQ(norm->tensor.name, "blk.0.ffn_norm.weight");
  auto missing = c;
  missing.tensors.erase(
      std::remove_if(
          missing.tensors.begin(), missing.tensors.end(),
          [](const auto &t) { return t.name == "blk.1.attn_k_norm.weight"; }),
      missing.tensors.end());
  EXPECT_FALSE(LoadModelDescriptor(missing, &m, &error));
  EXPECT_NE(error.find("attn_k_norm.weight"), std::string::npos);
  EXPECT_EQ(m.layers.size(), 2u);
  Scalar(c, "qwen3.rope.dimension_count", uint64_t{2});
  EXPECT_FALSE(LoadModelDescriptor(c, &m, &error));
  EXPECT_NE(error.find("RoPE width"), std::string::npos);
}
} // namespace hrx::model

namespace hrx::model {
TEST(ModelTensorNames, SharedSpellingFamilyExceptionsAndRoleScope) {
  for (const std::string family :
       {"llama", "qwen2", "qwen3", "gemma3", "qwen35"}) {
    ModelDescriptor m;
    m.family = family;
    std::string name;
    ASSERT_TRUE(ExpectedTensorName(m, "query", 7, &name));
    EXPECT_EQ(name, "blk.7.attn_q.weight");
    ASSERT_TRUE(ExpectedTensorName(m, "ffn_down", 0, &name));
    EXPECT_EQ(name, "blk.0.ffn_down.weight");
    ASSERT_TRUE(ExpectedTensorName(m, "output", std::nullopt, &name));
    EXPECT_EQ(name, "output.weight");
    m.output_tied_to_embedding = true;
    ASSERT_TRUE(ExpectedTensorName(m, "output", std::nullopt, &name));
    EXPECT_EQ(name, "token_embd.weight");
    for (const auto &invalid :
         std::vector<std::pair<std::string, std::optional<uint32_t>>>{
             {"query", std::nullopt},
             {"output", 0},
             {"unknown", 0},
             {"unknown", std::nullopt},
             {"rope_frequencies", 0},
             {"post_norm", std::nullopt},
             {"query_bias", std::nullopt}}) {
      name = "unchanged";
      EXPECT_FALSE(ExpectedTensorName(m, invalid.first, invalid.second, &name));
      EXPECT_EQ(name, "unchanged");
    }
    EXPECT_EQ(ExpectedTensorName(m, "post_norm", 2, &name), family != "gemma3");
    if (family != "gemma3")
      EXPECT_EQ(name, family == "qwen35" ? "blk.2.post_attention_norm.weight"
                                         : "blk.2.ffn_norm.weight");
    EXPECT_EQ(ExpectedTensorName(m, "query_norm", 2, &name),
              family != "llama" && family != "qwen2");
    EXPECT_EQ(ExpectedTensorName(m, "convolution", 2, &name),
              family == "qwen35");
    EXPECT_EQ(ExpectedTensorName(m, "rope_frequencies", std::nullopt, &name),
              family == "llama");
    if (family == "llama")
      EXPECT_EQ(name, "rope_freqs.weight");
    EXPECT_EQ(ExpectedTensorName(m, "query_bias", 12, &name),
              family == "qwen2");
    if (family == "qwen2")
      EXPECT_EQ(name, "blk.12.attn_q.bias");
    EXPECT_EQ(ExpectedTensorName(m, "ffn_post_norm", 2, &name),
              family == "gemma3");
  }
  ModelDescriptor unknown;
  unknown.family = "unimplemented";
  std::string name = "unchanged";
  EXPECT_FALSE(ExpectedTensorName(unknown, "query", 0, &name));
  EXPECT_EQ(name, "unchanged");
}
} // namespace hrx::model
