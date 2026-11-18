// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::internal {
namespace {
model::TensorRecord Encoded(uint32_t tag, uint64_t input, uint64_t output,
                            const std::string &name = "output.weight") {
  model::TensorRecord tensor;
  tensor.name = name;
  tensor.storage_type = tag;
  tensor.rank = 2;
  tensor.normalized_extents = {int64_t(input), int64_t(output), 1, 1};
  EXPECT_TRUE(
      model::EncodedTensorByteSize(tag, input, output, &tensor.byte_size));
  return tensor;
}

TEST(ComponentConfigurationTest,
     CanonicalFormatBoundsFollowTheConsumedOperation) {
  for (const auto tag : {2u, 3u, 6u, 8u, 11u, 12u, 13u, 14u, 20u, 21u, 23u}) {
    SCOPED_TRACE(tag);
    const uint64_t minimum =
        tag == 2 || tag == 3 || tag == 6 || tag == 8 || tag == 20 ? 128 : 256;
    const uint64_t maximum = tag == 14 ? 17408 : 32768;
    const uint64_t output_limit = tag == 8 || tag == 14 ? 262208 : 262144;
    for (const auto width : {minimum, maximum}) {
      const auto tensor = Encoded(tag, width, output_limit);
      CanonicalProjectionFacts result{};
      std::string error;
      ASSERT_TRUE(CheckCanonicalProjection(tensor, width, output_limit, &result,
                                           &error))
          << error;
      EXPECT_EQ(result.storage_tag, tag);
      EXPECT_EQ(result.weight_bytes, tensor.byte_size);
      EXPECT_EQ(result.input_width, width);
      EXPECT_EQ(result.output_width, output_limit);
    }
    for (const auto width : {minimum - 1, maximum + minimum}) {
      auto tensor = Encoded(tag, minimum, 1);
      CanonicalProjectionFacts result{99, 77, 66, 55};
      EXPECT_FALSE(
          CheckCanonicalProjection(tensor, width, 1, &result, nullptr));
      EXPECT_EQ(result.storage_tag, 99u);
      EXPECT_EQ(result.weight_bytes, 77u);
    }
    auto tensor = Encoded(tag, minimum, 1);
    CanonicalProjectionFacts result{};
    EXPECT_FALSE(
        CheckCanonicalProjection(tensor, minimum, 0, &result, nullptr));
    const auto oversized = Encoded(tag, minimum, output_limit + 1);
    EXPECT_FALSE(CheckCanonicalProjection(oversized, minimum, output_limit + 1,
                                          &result, nullptr));
    tensor.byte_size++;
    EXPECT_FALSE(
        CheckCanonicalProjection(tensor, minimum, 1, &result, nullptr));
  }
  auto unsupported = Encoded(0, 256, 1);
  CanonicalProjectionFacts result{};
  EXPECT_FALSE(CheckCanonicalProjection(unsupported, 256, 1, &result, nullptr));
}

TEST(ComponentConfigurationTest, NewProjectionFormatsDoNotAdmitUnusedGathers) {
  for (const auto tag : {11u, 20u, 21u}) {
    const auto tensor = Encoded(tag, 256, 3);
    OrdinaryEmbeddingFacts result{};
    EXPECT_FALSE(CheckOrdinaryEmbedding(tensor, 256, 3, &result, nullptr));
  }
  // Equal wire extents do not erase the identity of the two 3-bit codecs.
  CanonicalProjectionFacts iq{}, q{};
  ASSERT_TRUE(
      CheckCanonicalProjection(Encoded(21, 256, 3), 256, 3, &iq, nullptr));
  ASSERT_TRUE(
      CheckCanonicalProjection(Encoded(11, 256, 3), 256, 3, &q, nullptr));
  EXPECT_EQ(iq.weight_bytes, q.weight_bytes);
  EXPECT_NE(iq.storage_tag, q.storage_tag);
}

TEST(ComponentConfigurationTest,
     OrdinaryGatherHasItsOwnFiniteEncodingContract) {
  for (const auto tag : {8u, 12u, 13u, 14u}) {
    SCOPED_TRACE(tag);
    const uint64_t vocabulary_limit = tag == 8 || tag == 14 ? 262208 : 262144;
    for (const auto hidden : {256u, 32768u}) {
      const auto tensor = Encoded(tag, hidden, vocabulary_limit);
      OrdinaryEmbeddingFacts result{};
      ASSERT_TRUE(CheckOrdinaryEmbedding(tensor, hidden, vocabulary_limit,
                                         &result, nullptr));
      EXPECT_EQ(result.storage_tag, tag);
      EXPECT_EQ(result.hidden_size, hidden);
      EXPECT_EQ(result.vocabulary_size, vocabulary_limit);
      EXPECT_EQ(result.weight_bytes, tensor.byte_size);
    }
    auto tensor = Encoded(tag, 256, 3);
    OrdinaryEmbeddingFacts result{99, 77, 66, 55};
    for (const auto hidden : {0u, 255u, 257u, 33024u}) {
      EXPECT_FALSE(CheckOrdinaryEmbedding(tensor, hidden, 3, &result, nullptr));
      EXPECT_EQ(result.storage_tag, 99u);
    }
    EXPECT_FALSE(CheckOrdinaryEmbedding(tensor, 256, 0, &result, nullptr));
    const auto oversized = Encoded(tag, 256, vocabulary_limit + 1);
    EXPECT_FALSE(CheckOrdinaryEmbedding(oversized, 256, vocabulary_limit + 1,
                                        &result, nullptr));
    tensor.byte_size--;
    EXPECT_FALSE(CheckOrdinaryEmbedding(tensor, 256, 3, &result, nullptr));
  }
  auto unsupported = Encoded(1, 256, 3);
  OrdinaryEmbeddingFacts result{};
  EXPECT_FALSE(CheckOrdinaryEmbedding(unsupported, 256, 3, &result, nullptr));
}

TEST(ComponentConfigurationTest, FullVocabularyIsBoundedToReachedHeadFormats) {
  for (const auto tag : {8u, 14u}) {
    SCOPED_TRACE(tag);
    const auto tensor = Encoded(tag, 3840, 262208);
    CanonicalProjectionFacts projection{};
    OrdinaryEmbeddingFacts embedding{};
    ASSERT_TRUE(
        CheckCanonicalProjection(tensor, 3840, 262208, &projection, nullptr));
    ASSERT_TRUE(
        CheckOrdinaryEmbedding(tensor, 3840, 262208, &embedding, nullptr));
    const uint64_t row_bytes = tag == 8 ? 4080 : 3150;
    EXPECT_EQ(projection.weight_bytes, row_bytes * 262208);
    EXPECT_EQ(embedding.weight_bytes, projection.weight_bytes);
    EXPECT_EQ((262207 * row_bytes) + row_bytes, projection.weight_bytes);
    auto oversized = Encoded(tag, 3840, 262209);
    EXPECT_FALSE(CheckCanonicalProjection(oversized, 3840, 262209, &projection,
                                          nullptr));
    EXPECT_FALSE(
        CheckOrdinaryEmbedding(oversized, 3840, 262209, &embedding, nullptr));
    auto truncated = tensor;
    --truncated.byte_size;
    EXPECT_FALSE(CheckCanonicalProjection(truncated, 3840, 262208, &projection,
                                          nullptr));
    EXPECT_FALSE(
        CheckOrdinaryEmbedding(truncated, 3840, 262208, &embedding, nullptr));
  }
  for (const auto tag : {12u, 13u}) {
    const auto tensor = Encoded(tag, 3840, 262208);
    CanonicalProjectionFacts projection{};
    OrdinaryEmbeddingFacts embedding{};
    EXPECT_FALSE(
        CheckCanonicalProjection(tensor, 3840, 262208, &projection, nullptr));
    EXPECT_FALSE(
        CheckOrdinaryEmbedding(tensor, 3840, 262208, &embedding, nullptr));
  }
}

TEST(ComponentConfigurationTest, NarrowQ8GatherDoesNotBroadenKBlockFormats) {
  const auto q8 = Encoded(8, 896, 151936);
  OrdinaryEmbeddingFacts result{};
  ASSERT_TRUE(CheckOrdinaryEmbedding(q8, 896, 151936, &result, nullptr));
  EXPECT_EQ(result.hidden_size, 896u);
  for (const auto tag : {12u, 13u, 14u}) {
    auto tensor = Encoded(tag, 1024, 3);
    tensor.normalized_extents[0] = 896;
    EXPECT_FALSE(CheckOrdinaryEmbedding(tensor, 896, 3, &result, nullptr));
  }
}

TEST(ComponentConfigurationTest, BindingFailuresDoNotPublishPartialFacts) {
  model::ModelDescriptor descriptor;
  descriptor.family = "spark2_5";
  std::string error;
  Producer producer;
  producer.model = &descriptor;
  producer.asset = "asset";
  producer.error = &error;
  const auto good = Encoded(13, 1024, 3, "token_embd.weight");
  OrdinaryEmbeddingFacts result{99, 77, 66, 55};
  for (int mutation = 0; mutation != 6; ++mutation) {
    auto tensor = good;
    std::vector<model::TensorBinding> bindings;
    switch (mutation) {
    case 0:
      tensor.name = "wrong.weight";
      break;
    case 1:
      tensor.rank = 1;
      break;
    case 2:
      tensor.normalized_extents[0] = 2048;
      break;
    case 3:
      tensor.byte_size++;
      break;
    case 4:
      tensor.absolute_offset = uint64_t(INT64_MAX);
      break;
    default:
      break;
    }
    bindings.push_back({"token_embedding", tensor});
    if (mutation == 5)
      bindings.push_back({"token_embedding", tensor});
    EXPECT_FALSE(producer.OrdinaryEmbedding(bindings, "token_embedding", 1024,
                                            3, &result))
        << mutation;
    EXPECT_TRUE(producer.facts.parameters.empty());
    EXPECT_TRUE(producer.identities.empty());
    EXPECT_TRUE(producer.projection_widths.empty());
    EXPECT_EQ(result.storage_tag, 99u);
  }
  ASSERT_TRUE(producer.OrdinaryEmbedding({{"token_embedding", good}},
                                         "token_embedding", 1024, 3, &result));
  EXPECT_EQ(producer.facts.parameters.size(), 1u);
  EXPECT_EQ(result.storage_tag, 13u);
  auto conflicting = good;
  conflicting.absolute_offset = 16;
  EXPECT_FALSE(producer.OrdinaryEmbedding({{"token_embedding", conflicting}},
                                          "token_embedding", 1024, 3, &result));
  EXPECT_EQ(producer.facts.parameters.size(), 1u);
  EXPECT_EQ(producer.identities.at(good.name).absolute_offset,
            good.absolute_offset);
}

TEST(ComponentConfigurationTest, CanonicalBindingRecordsOnlyAdmittedCapacity) {
  model::ModelDescriptor descriptor;
  descriptor.family = "spark2_5";
  Producer producer;
  producer.model = &descriptor;
  producer.asset = "asset";
  std::string error;
  producer.error = &error;
  auto tensor = Encoded(6, 1024, 3);
  CanonicalProjectionFacts result{99, 77, 66, 55};
  tensor.absolute_offset = uint64_t(INT64_MAX);
  EXPECT_FALSE(producer.CanonicalProjection({{"output", tensor}}, "output",
                                            1024, 3, std::nullopt, &result));
  EXPECT_TRUE(producer.projection_widths.empty());
  EXPECT_EQ(producer.packed_input_width, 0u);
  EXPECT_TRUE(producer.facts.parameters.empty());
  tensor.absolute_offset = 0;
  ASSERT_TRUE(producer.CanonicalProjection({{"output", tensor}}, "output", 1024,
                                           3, std::nullopt, &result))
      << error;
  EXPECT_EQ(result.storage_tag, 6u);
  EXPECT_EQ(producer.projection_widths.at(6), 3u);
  EXPECT_EQ(producer.packed_input_width, 1024u);
  EXPECT_EQ(producer.facts.parameters.size(), 1u);
}
} // namespace
} // namespace hrx::loomify_gguf::internal
