// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/text/jinja/compiler.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::jinja::testing {
namespace {
namespace fs = std::filesystem;

class TemplateProbeRunnerTest : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    const char *runfiles = std::getenv("TEST_SRCDIR");
    const char *workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(temporary, nullptr);
    ASSERT_NE(runfiles, nullptr);
    ASSERT_NE(workspace, nullptr);
    helpers_ = (fs::path(runfiles) / workspace).string();
    std::string pattern = (fs::path(temporary) / "runner-test-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = ::mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    scratch_ = created;
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(scratch_, ignored);
  }
  bool Prepare(const GeneratedInputRenderer &generated,
               std::unique_ptr<PreparedTemplate> *out, std::string *error) {
    return PreparedTemplate::Prepare(generated, helpers_, scratch_.string(),
                                     GetParam(), out, error);
  }
  bool Compile(std::string_view source, GeneratedInputRenderer *out,
               std::string *error) {
    Diagnostic diagnostic;
    if (CompileChatTemplate(source, "runner-test.jinja", out, &diagnostic))
      return true;
    *error = diagnostic.message;
    return false;
  }
  GeneratedInputRenderer Echo() {
    return {"func.def public @runner.echo(%value: buffer) -> (buffer) {\n"
            "  func.return %value : buffer\n}\n",
            "runner.echo",
            {}};
  }

  std::string helpers_;
  fs::path scratch_;
};

TEST_P(TemplateProbeRunnerTest, RepeatedRenderRecoversAfterTemplateError) {
  GeneratedInputRenderer generated;
  std::string error;
  ASSERT_TRUE(Compile("{% if fail %}{{ raise_exception('requested failure') }}"
                      "{% else %}{{ value|default('') }}{% endif %}",
                      &generated, &error))
      << error;
  std::unique_ptr<PreparedTemplate> prepared;
  ASSERT_TRUE(Prepare(generated, &prepared, &error)) << error;
  std::string output = "unchanged";
  ASSERT_TRUE(prepared->Render(R"({"value":"first"})", &output, &error))
      << error;
  EXPECT_EQ(output, "first");
  EXPECT_FALSE(prepared->Render(R"({"fail":true})", &output, &error));
  EXPECT_NE(error.find("requested failure"), std::string::npos) << error;
  EXPECT_EQ(output, "first");
  EXPECT_FALSE(prepared->Render("[]", &output, &error));
  EXPECT_EQ(output, "first");
  ASSERT_TRUE(prepared->Render(R"({"value":"last"})", &output, &error))
      << error;
  EXPECT_EQ(output, "last");
  ASSERT_TRUE(prepared->Render("{}", &output, &error)) << error;
  EXPECT_TRUE(output.empty());
}

TEST_P(TemplateProbeRunnerTest, CopiesAliasedBufferBeforeArgumentRelease) {
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error;
  ASSERT_TRUE(Prepare(Echo(), &prepared, &error)) << error;
  std::string output("a\0b", 3);
  ASSERT_TRUE(prepared->Render(output, &output, &error)) << error;
  EXPECT_EQ(output, std::string("a\0b", 3));
  prepared.reset();
  EXPECT_EQ(output, std::string("a\0b", 3));
  EXPECT_TRUE(fs::is_empty(scratch_));
}

TEST_P(TemplateProbeRunnerTest, MissingResourcesPreserveExistingProgram) {
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error;
  ASSERT_TRUE(Prepare(Echo(), &prepared, &error)) << error;
  auto *original = prepared.get();
  auto missing = Echo();
  missing.helper_requirements = {"missing/helper.loom"};
  EXPECT_FALSE(Prepare(missing, &prepared, &error));
  EXPECT_NE(error.find("missing template helper resource"), std::string::npos);
  EXPECT_EQ(prepared.get(), original);
  std::string output;
  ASSERT_TRUE(prepared->Render("still usable", &output, &error)) << error;
  EXPECT_EQ(output, "still usable");
}

TEST_P(TemplateProbeRunnerTest, RejectsHelperTraversalBeforePreparation) {
  for (const char *path : {"../helper.loom", "lib/../helper.loom",
                           "/helper.loom", "./helper.loom", ""}) {
    SCOPED_TRACE(path);
    auto generated = Echo();
    generated.helper_requirements = {path};
    std::unique_ptr<PreparedTemplate> prepared;
    std::string error;
    EXPECT_FALSE(Prepare(generated, &prepared, &error));
    EXPECT_EQ(prepared, nullptr);
    EXPECT_NE(error.find("invalid template helper resource"),
              std::string::npos);
    EXPECT_TRUE(fs::is_empty(scratch_));
  }
}

TEST_P(TemplateProbeRunnerTest, FailedNativeLoadRemovesOwnedScratch) {
  auto generated = Echo();
  generated.entrypoint = "runner.missing";
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error;
  EXPECT_FALSE(Prepare(generated, &prepared, &error));
  EXPECT_EQ(prepared, nullptr);
  EXPECT_FALSE(error.empty());
  EXPECT_TRUE(fs::is_empty(scratch_));
}

TEST_P(TemplateProbeRunnerTest,
       InvokesTwoScalarSelectorAndPreservesOnTypeError) {
  GeneratedInputRenderer selector{
      "func.def public @runner.select(%value: buffer) -> (i64, i64) {\n"
      "  %format = scalar.constant 42 : i64\n"
      "  %mode = scalar.constant 2 : i64\n"
      "  func.return %format, %mode : i64, i64\n}\n",
      "runner.select",
      {}};
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error;
  ASSERT_TRUE(Prepare(selector, &prepared, &error)) << error;
  int64_t format = -1, mode = -2;
  ASSERT_TRUE(prepared->SelectFormat("{}", &format, &mode, &error)) << error;
  EXPECT_EQ(format, 42);
  EXPECT_EQ(mode, 2);
  GeneratedInputRenderer wrong_types{
      "func.def public @runner.wrong(%value: buffer) -> (buffer, buffer) {\n"
      "  func.return %value, %value : buffer, buffer\n}\n",
      "runner.wrong",
      {}};
  ASSERT_TRUE(Prepare(wrong_types, &prepared, &error)) << error;
  EXPECT_FALSE(prepared->SelectFormat("{}", &format, &mode, &error));
  EXPECT_EQ(format, 42);
  EXPECT_EQ(mode, 2);
}

INSTANTIATE_TEST_SUITE_P(SerializedForms, TemplateProbeRunnerTest,
                         ::testing::Bool());

} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
