// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/text_metadata.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#if defined(__GLIBC__)
#include <unistd.h>
#endif
namespace hrx::model {
namespace {
void Number(std::string &out, uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    out.push_back(char(value >> (8 * i)));
}
void String(std::string &out, const std::string &value) {
  Number(out, value.size(), 8);
  out += value;
}
struct Field {
  std::string key;
  uint32_t type;
  std::string value;
};
Field Text(std::string key, std::string text) {
  std::string s;
  String(s, text);
  return {std::move(key), 8, std::move(s)};
}
Field Int(std::string key, uint64_t value, uint32_t type = 4,
          unsigned width = 4) {
  std::string s;
  Number(s, value, width);
  return {std::move(key), type, std::move(s)};
}
Field Strings(std::string key, const std::vector<std::string> &values) {
  std::string s;
  Number(s, 8, 4);
  Number(s, values.size(), 8);
  for (const auto &v : values)
    String(s, v);
  return {std::move(key), 9, std::move(s)};
}
std::string Fixture(std::vector<Field> fields, uint32_t version = 3) {
  fields.insert(fields.begin(), Text("general.architecture", "llama"));
  std::string s = "GGUF";
  Number(s, version, 4);
  Number(s, 0, 8);
  Number(s, fields.size(), 8);
  for (const auto &f : fields) {
    String(s, f.key);
    Number(s, f.type, 4);
    s += f.value;
  }
  while (s.size() % 32)
    s.push_back(0);
  s += std::string(64, '!');
  return s;
}
using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
File Open(const std::string &bytes) {
  File f(std::tmpfile(), std::fclose);
  if (f &&
      (std::setvbuf(f.get(), nullptr, _IONBF, 0) != 0 ||
       std::fwrite(bytes.data(), 1, bytes.size(), f.get()) != bytes.size()))
    f.reset();
  return f;
}
const TextMetadataIssue *IssueAt(const std::vector<TextMetadataIssue> &issues,
                                 const std::string &key) {
  for (const auto &issue : issues)
    if (issue.source_key == key)
      return &issue;
  return nullptr;
}
const ChatTemplateSource *Template(const TextMetadata &text,
                                   const std::string &key) {
  for (const auto &t : text.templates)
    if (t.source_key == key)
      return &t;
  return nullptr;
}
TEST(TextMetadata, AbsenceZeroFalseAndOwnedFactsAreDistinct) {
  static_assert(!std::is_default_constructible<GgufStringArraySpan>::value);
  static_assert(!std::is_constructible<GgufStringArraySpan, uint64_t, uint64_t,
                                       uint64_t, uint64_t>::value);
  for (auto version : {2u, 3u}) {
    auto f = Open(Fixture({Strings("tokenizer.ggml.tokens", {"zero", "one"}),
                           Int("tokenizer.ggml.bos_token_id", 0),
                           Int("tokenizer.ggml.padding_token_id", 0),
                           Int("tokenizer.ggml.add_bos_token", 0, 7, 1),
                           Text("tokenizer.ggml.model", "gpt2"),
                           Text("tokenizer.ggml.pre", "fixture"),
                           Text("tokenizer.chat_template", " x\r\nπ\n")},
                          version));
    ASSERT_TRUE(f);
    TextMetadata text;
    std::string error = "stale";
    {
      GgufCatalog catalog;
      ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error)) << error;
      ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error)) << error;
      EXPECT_EQ(catalog.header_end, text.vocabulary->header_end());
      EXPECT_LE(text.vocabulary->end(), catalog.header_end);
      EXPECT_LE(catalog.header_end, catalog.data_offset);
    }
    f.reset();
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(text.tokenizer_model, "gpt2");
    EXPECT_EQ(text.pre_tokenizer, "fixture");
    ASSERT_TRUE(text.add_bos.has_value());
    EXPECT_FALSE(*text.add_bos);
    EXPECT_FALSE(text.add_eos.has_value());
    ASSERT_EQ(text.tokens.size(), 2);
    EXPECT_EQ(text.tokens[0].id, 0);
    ASSERT_EQ(text.token_pieces.size(), 1);
    EXPECT_EQ(text.token_pieces.at(0), "zero");
    EXPECT_TRUE(text.vocabulary_issues.empty());
    EXPECT_TRUE(text.token_issues.empty());
    EXPECT_TRUE(text.template_issues.empty());
    ASSERT_EQ(text.templates.size(), 1);
    EXPECT_FALSE(text.templates[0].name);
    EXPECT_EQ(text.templates[0].text, " x\r\nπ\n");
  }
  auto f = Open(Fixture({}));
  GgufCatalog catalog;
  TextMetadata text;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error));
  EXPECT_FALSE(text.vocabulary);
  EXPECT_FALSE(text.add_bos);
  EXPECT_FALSE(text.add_eos);
  EXPECT_FALSE(text.tokenizer_model);
  EXPECT_TRUE(text.templates.empty());
  EXPECT_TRUE(text.tokens.empty());
}
TEST(TextMetadata, EveryIntegerWireTypeAndMetadataOrderWorks) {
  for (auto order : {false, true}) {
    std::vector<Field> fields;
    unsigned n = 0;
    for (auto type : {0u, 1u, 2u, 3u, 4u, 5u, 10u, 11u}) {
      const unsigned width = type < 2 ? 1 : type < 4 ? 2 : type < 6 ? 4 : 8;
      fields.push_back(
          Int("tokenizer.ggml.role" + std::to_string(n++) + "_token_id", 1,
              type, width));
    }
    auto tokens = Strings("tokenizer.ggml.tokens", {"a", "b"});
    fields.insert(order ? fields.end() : fields.begin(), tokens);
    auto f = Open(Fixture(fields));
    GgufCatalog catalog;
    TextMetadata text;
    std::string error;
    ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
    ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error)) << error;
    EXPECT_TRUE(text.token_issues.empty());
    ASSERT_EQ(text.tokens.size(), 8);
    for (const auto &t : text.tokens)
      EXPECT_EQ(t.id, 1);
    EXPECT_EQ(text.token_pieces.size(), 1);
    EXPECT_EQ(text.token_pieces.at(1), "b");
  }
}
TEST(TextMetadata, SemanticIssuesDoNotHideUnrelatedFacts) {
  auto f = Open(Fixture(
      {Strings("tokenizer.ggml.tokens", {"good", "other"}),
       Int("tokenizer.ggml.eos_token_id", 0),
       Int("tokenizer.ggml.bos_token_id", 255, 1, 1),
       Int("tokenizer.ggml.padding_token_id", 2),
       Int("tokenizer.ggml.mask_token_id", 1, 7, 1),
       Int("tokenizer.ggml.floating_token_id", 0x3f800000, 6, 4),
       Text("tokenizer.ggml.unknown_token_id", "1"),
       Int("tokenizer.ggml.add_bos_token", 0), Int("tokenizer.ggml.model", 1),
       Int("tokenizer.chat_templates", 2),
       Text("tokenizer.chat_template", " valid bytes ")}));
  GgufCatalog catalog;
  TextMetadata text;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error)) << error;
  EXPECT_TRUE(error.empty());
  ASSERT_EQ(text.tokens.size(), 1);
  EXPECT_EQ(text.token_pieces.at(0), "good");
  for (auto key :
       {"bos_token_id", "padding_token_id", "mask_token_id", "unknown_token_id",
        "add_bos_token", "model", "floating_token_id"})
    EXPECT_NE(IssueAt(text.token_issues, std::string("tokenizer.ggml.") + key),
              nullptr)
        << key;
  EXPECT_NE(IssueAt(text.template_issues, "tokenizer.chat_templates"), nullptr);
  EXPECT_EQ(Template(text, "tokenizer.chat_template")->text, " valid bytes ");
  EXPECT_FALSE(text.add_bos);
  EXPECT_FALSE(text.tokenizer_model);
  auto wrong = Open(Fixture(
      {Int("tokenizer.ggml.tokens", 1), Int("tokenizer.ggml.eos_token_id", 0),
       Text("tokenizer.chat_template", "still present")}));
  ASSERT_TRUE(ReadGgufCatalog(wrong.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(wrong.get(), catalog, &text, &error));
  EXPECT_FALSE(text.vocabulary);
  ASSERT_FALSE(text.vocabulary_issues.empty());
  EXPECT_FALSE(text.token_issues.empty());
  EXPECT_EQ(text.templates[0].text, "still present");
}
TEST(TextMetadata, NamedTemplatesKeepExactIndexAndScalarIdentity) {
  const std::vector<std::string> names = {"tool_use", "default"};
  auto f =
      Open(Fixture({Strings("tokenizer.chat_templates", names),
                    Text("tokenizer.chat_template.tool_use", "tools\n"),
                    Text("tokenizer.chat_template.default", "named default"),
                    Text("tokenizer.chat_template", "scalar default")}));
  GgufCatalog catalog;
  TextMetadata text;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error));
  EXPECT_EQ(text.named_template_order, names);
  EXPECT_TRUE(text.template_issues.empty());
  ASSERT_EQ(text.templates.size(), 3);
  const auto *scalar = Template(text, "tokenizer.chat_template");
  ASSERT_TRUE(scalar);
  EXPECT_FALSE(scalar->name);
  EXPECT_EQ(scalar->text, "scalar default");
  const auto *named = Template(text, "tokenizer.chat_template.default");
  ASSERT_TRUE(named);
  EXPECT_EQ(named->name, "default");
  EXPECT_TRUE(named->listed);
  EXPECT_EQ(named->text, "named default");
  EXPECT_EQ(Template(text, "tokenizer.chat_template.tool_use")->text,
            "tools\n");
}
TEST(TextMetadata, MalformedTemplateIndexPreservesCandidatesAndTokenGroup) {
  auto f = Open(Fixture({Strings("tokenizer.chat_templates",
                                 {"good", "good", "", "missing", "wrong"}),
                         Text("tokenizer.chat_template.good", "known"),
                         Text("tokenizer.chat_template.orphan", "unlisted"),
                         Int("tokenizer.chat_template.wrong", 3),
                         Strings("tokenizer.ggml.tokens", {"marker"}),
                         Int("tokenizer.ggml.eos_token_id", 0)}));
  GgufCatalog catalog;
  TextMetadata text;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error));
  EXPECT_EQ(text.named_template_order.size(), 5);
  EXPECT_GE(text.template_issues.size(), 5);
  EXPECT_NE(IssueAt(text.template_issues, "tokenizer.chat_template.missing"),
            nullptr);
  EXPECT_NE(IssueAt(text.template_issues, "tokenizer.chat_template.wrong"),
            nullptr);
  ASSERT_TRUE(Template(text, "tokenizer.chat_template.orphan"));
  EXPECT_FALSE(Template(text, "tokenizer.chat_template.orphan")->listed);
  EXPECT_TRUE(text.token_issues.empty());
  EXPECT_EQ(text.token_pieces.at(0), "marker");
  auto oversized = Open(Fixture({Strings(
      "tokenizer.chat_templates", std::vector<std::string>(4097, "x"))}));
  ASSERT_TRUE(ReadGgufCatalog(oversized.get(), &catalog, &error));
  ASSERT_TRUE(ReadTextMetadata(oversized.get(), catalog, &text, &error));
  EXPECT_TRUE(text.named_template_order.empty());
  ASSERT_NE(IssueAt(text.template_issues, "tokenizer.chat_templates"), nullptr);
  EXPECT_NE(IssueAt(text.template_issues, "tokenizer.chat_templates")
                ->message.find("count"),
            std::string::npos);
}
TEST(TextMetadata, ArrayReadersPreserveOrderRawBytesAndFilePosition) {
  const std::vector<std::string> values = {"zero", "", std::string("a\0b", 3),
                                           "π"};
  auto f = Open(Fixture({Strings("tokenizer.ggml.tokens", values),
                         Strings("tokenizer.ggml.merges", {"a b", "b c"})}));
  GgufCatalog catalog;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  const auto span =
      *FindMetadata(catalog, "tokenizer.ggml.tokens")->string_array;
  ASSERT_EQ(fseeko(f.get(), 7, SEEK_SET), 0);
  std::vector<std::string> pieces;
  ASSERT_TRUE(
      ReadGgufTokenPieces(f.get(), span, {3, 0, 2, 3, 1}, &pieces, &error))
      << error;
  EXPECT_EQ(pieces, (std::vector<std::string>{values[3], values[0], values[2],
                                              values[3], values[1]}));
  EXPECT_EQ(ftello(f.get()), 7);
  std::vector<std::string> visited;
  ASSERT_TRUE(VisitGgufStringArray(
      f.get(), span,
      [&](uint64_t id, std::string_view piece, std::string *) {
        EXPECT_EQ(id, visited.size());
        visited.emplace_back(piece);
        return true;
      },
      &error));
  EXPECT_EQ(visited, values);
  EXPECT_EQ(ftello(f.get()), 7);
  EXPECT_FALSE(VisitGgufStringArray(
      f.get(), span,
      [&](uint64_t, std::string_view, std::string *e) {
        *e = "intentional stop";
        return false;
      },
      &error));
  EXPECT_NE(error.find("intentional stop"), std::string::npos);
  EXPECT_EQ(ftello(f.get()), 7);
  pieces = {"prior"};
  EXPECT_FALSE(ReadGgufTokenPieces(f.get(), span, {4}, &pieces, &error));
  EXPECT_EQ(pieces, (std::vector<std::string>{"prior"}));
  EXPECT_EQ(ftello(f.get()), 7);
  std::vector<uint64_t> excess(
      64 * 1024 * 1024 /
              (sizeof(std::string) + sizeof(std::pair<uint64_t, size_t>)) +
          1,
      0);
  EXPECT_FALSE(ReadGgufTokenPieces(f.get(), span, excess, &pieces, &error));
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(pieces[0], "prior");
}
TEST(TextMetadata,
     ChangedEncodedLengthsFailTransactionallyInsideAdmittedBounds) {
  auto f = Open(Fixture({Strings("tokenizer.ggml.tokens", {"one", "two"}),
                         Int("tokenizer.ggml.eos_token_id", 0)}));
  GgufCatalog catalog;
  TextMetadata prior;
  prior.tokenizer_model = "prior";
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  const auto span =
      *FindMetadata(catalog, "tokenizer.ggml.tokens")->string_array;
  std::string bad;
  Number(bad, UINT64_MAX, 8);
  ASSERT_EQ(fseeko(f.get(), span.begin(), SEEK_SET), 0);
  ASSERT_EQ(fwrite(bad.data(), 1, bad.size(), f.get()), bad.size());
  ASSERT_EQ(fseeko(f.get(), 3, SEEK_SET), 0);
  EXPECT_FALSE(ReadTextMetadata(f.get(), catalog, &prior, &error));
  EXPECT_EQ(prior.tokenizer_model, "prior");
  EXPECT_EQ(ftello(f.get()), 3);
  std::vector<std::string> old = {"old"};
  EXPECT_FALSE(ReadGgufTokenPieces(f.get(), span, {0}, &old, &error));
  EXPECT_EQ(old[0], "old");
  EXPECT_EQ(ftello(f.get()), 3);
}
#if defined(__GLIBC__)
struct Cookie {
  std::string bytes;
  uint64_t position = 0;
  std::vector<std::pair<uint64_t, uint64_t>> reads;
};
ssize_t CookieRead(void *opaque, char *out, size_t size) {
  auto &c = *static_cast<Cookie *>(opaque);
  size = std::min<uint64_t>(
      size, c.bytes.size() - std::min<uint64_t>(c.position, c.bytes.size()));
  if (size) {
    std::memcpy(out, c.bytes.data() + c.position, size);
    c.reads.emplace_back(c.position, c.position + size);
    c.position += size;
  }
  return size;
}
int CookieSeek(void *opaque, off64_t *offset, int origin) {
  auto &c = *static_cast<Cookie *>(opaque);
  const int64_t base = origin == SEEK_SET   ? 0
                       : origin == SEEK_END ? c.bytes.size()
                                            : c.position;
  if (*offset < -base ||
      (*offset > 0 && uint64_t(*offset) > UINT64_MAX - uint64_t(base))) {
    errno = EINVAL;
    return -1;
  }
  c.position = base + *offset;
  *offset = c.position;
  return 0;
}
TEST(TextMetadata, SelectiveReadsSkipOtherPiecesMergesAndAllPayload) {
  Cookie c{Fixture({Strings("tokenizer.ggml.tokens",
                            {"picked", std::string(100000, 'x'), "last"}),
                    Strings("tokenizer.ggml.merges", {std::string(10000, 'm')}),
                    Int("tokenizer.ggml.eos_token_id", 0)})};
  cookie_io_functions_t functions{};
  functions.read = CookieRead;
  functions.seek = CookieSeek;
  File f(fopencookie(&c, "rb", functions), std::fclose);
  ASSERT_TRUE(f);
  ASSERT_EQ(setvbuf(f.get(), nullptr, _IONBF, 0), 0);
  GgufCatalog catalog;
  TextMetadata text;
  std::string error;
  ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error));
  const auto before = catalog.bytes_read;
  c.reads.clear();
  ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error)) << error;
  EXPECT_EQ(catalog.bytes_read, before);
  EXPECT_EQ(text.token_pieces.at(0), "picked");
  ASSERT_TRUE(text.merges);
  const auto span = *text.vocabulary;
  const uint64_t unselected_begin = span.begin() + 8 + 6 + 8,
                 unselected_end = unselected_begin + 100000;
  for (auto range : c.reads) {
    EXPECT_LE(range.second, catalog.header_end);
    EXPECT_TRUE(range.second <= unselected_begin ||
                range.first >= unselected_end);
    EXPECT_TRUE(range.second <= text.merges->begin() ||
                range.first >= text.merges->end());
  }
  // A retained catalog cannot authorize reading a now-truncated file.
  c.bytes.resize(span.end() - 1);
  TextMetadata prior;
  prior.tokenizer_model = "prior";
  EXPECT_FALSE(ReadTextMetadata(f.get(), catalog, &prior, &error));
  EXPECT_EQ(prior.tokenizer_model, "prior");
}
#endif
#if defined(LOOMIFY_TEXT_REAL_ASSETS)
TEST(TextMetadataRealAssets, CapturedQwenHeaderFactsAndTemplateBytes) {
  for (bool large : {false, true}) {
    const auto *path = std::getenv(large ? "LOOMIFY_TEST_QWEN_27B_GGUF"
                                         : "LOOMIFY_TEST_QWEN_GGUF");
    ASSERT_NE(path, nullptr) << "explicit immutable GGUF fixture required";
    File f(std::fopen(path, "rb"), std::fclose);
    ASSERT_TRUE(f);
    ASSERT_EQ(setvbuf(f.get(), nullptr, _IONBF, 0), 0);
    GgufCatalog catalog;
    TextMetadata text;
    std::string error;
    ASSERT_TRUE(ReadGgufCatalog(f.get(), &catalog, &error)) << error;
    ASSERT_TRUE(ReadTextMetadata(f.get(), catalog, &text, &error)) << error;
    ASSERT_TRUE(text.vocabulary);
    EXPECT_EQ(text.vocabulary->count(), 248320);
    EXPECT_TRUE(text.token_issues.empty());
    EXPECT_TRUE(text.template_issues.empty());
    EXPECT_TRUE(text.vocabulary_issues.empty());
    EXPECT_EQ(text.tokenizer_model, "gpt2");
    EXPECT_EQ(text.pre_tokenizer, "qwen35");
    const auto *t = Template(text, "tokenizer.chat_template");
    ASSERT_TRUE(t);
    EXPECT_EQ(t->text.size(), large ? 8952 : 7816);
    bool eos = false, pad = false, bos = false;
    for (const auto &role : text.tokens) {
      if (role.source_key == "tokenizer.ggml.eos_token_id") {
        eos = true;
        EXPECT_EQ(role.id, 248046);
        EXPECT_EQ(text.token_pieces.at(role.id), "<|im_end|>");
      }
      if (role.source_key == "tokenizer.ggml.padding_token_id") {
        pad = true;
        EXPECT_EQ(role.id, large ? 248044 : 248055);
      }
      if (role.source_key == "tokenizer.ggml.bos_token_id") {
        bos = true;
        EXPECT_EQ(role.id, 248044);
      }
    }
    EXPECT_TRUE(eos);
    EXPECT_TRUE(pad);
    EXPECT_EQ(bos, large);
    EXPECT_EQ(text.add_bos.has_value(), large);
    if (large)
      EXPECT_FALSE(*text.add_bos);
    if (const auto *out = std::getenv("LOOMIFY_TEXT_EVIDENCE_DIR")) {
      const auto name = std::string(out) + (large ? "/qwen27-template.jinja"
                                                  : "/qwen08-template.jinja");
      File output(std::fopen(name.c_str(), "wb"), std::fclose);
      ASSERT_TRUE(output);
      ASSERT_EQ(fwrite(t->text.data(), 1, t->text.size(), output.get()),
                t->text.size());
    }
  }
}
#endif
} // namespace
} // namespace hrx::model
