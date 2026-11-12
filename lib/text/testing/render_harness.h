// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_TEXT_TESTING_RENDER_HARNESS_H_
#define HRX_SOLUTIONS_LIB_TEXT_TESTING_RENDER_HARNESS_H_

// Test-only SDK consumer support. The transport headers are SDK-private; the
// external consumer explicitly records its isolated visibility exception.
#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/program.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/tool_xml.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::loomify_gguf::testing {

iree_string_view_t View(std::string_view);
std::string Copy(iree_string_view_t);
std::string Diagnostic(iree_status_t);
bool Ok(iree_status_t);
bool Read(const std::string &, std::string *);
std::string Env(const std::string &);
bool Field(iree_string_view_t, const char *, iree_string_view_t *);
bool Decode(iree_string_view_t, std::string *);
iree_status_t Row(void *, iree_host_size_t, iree_json_value_type_t,
                  iree_string_view_t);
iree_status_t Token(void *, iree_host_size_t, iree_string_view_t);
bool TransportRequest(iree_string_view_t, std::string_view, std::string *);

class ChatVM {
public:
  ~ChatVM();
  bool Init(const std::string &product, const std::string &tokenizer_path,
            std::string_view expected_name);
  bool SupportsReasoning() const;
  std::string_view name() const;
  iree_status_t RequestOptions(const std::string &body);
  iree_status_t Render(const std::string &body, loom_serve_text_chat_t *chat);
  iree_status_t ParseTools(std::string_view schemas, std::string_view response,
                           std::string *out, bool truncated = false);
  bool Encode(std::string_view text, std::vector<int32_t> *tokens);
  bool Prepare(const loom_serve_text_chat_t &chat, size_t capacity,
               std::vector<int32_t> *tokens);

private:
  loomc_target_environment_t *target_ = nullptr;
  iree_vm_environment_t *environment_ = nullptr;
  iree_tokenizer_t *tokenizer_ = nullptr;
  iree_vm_module_t *json_ = nullptr;
  iree_vm_module_t *input_ = nullptr;
  iree_vm_module_t *tools_ = nullptr;
  loom_runtime_program_t *program_ = nullptr;
  loom_serve_text_chat_policy_t policy_{};
  bool policy_ready_ = false;
  std::string body_; // Native chat borrows request bytes through deinitialize.
};

bool RawChat(ChatVM &, std::string_view, loom_serve_text_chat_t *);
iree_status_t PrepareMode(const loom_serve_text_chat_t &,
                          loom_serve_text_chat_input_format_t,
                          loom_serve_text_chat_boundary_t, size_t, int32_t *,
                          iree_host_size_t *);

struct RenderReferences {
  std::string one_user, schema_request, history_request, history_text;
  std::vector<int32_t> history_ids;
};
struct TextRevisionCase {
  const char *id;
  const char *key;
  const char *package_name;
  const char *representative_case;
  size_t expected_rows, expected_template_errors, expected_host_rejections;
  bool supports_reasoning;
  void (*check_policy)(ChatVM &, const RenderReferences &);
};
struct InputPolicyExpectations {
  std::string raw_input, user_input, assistant_header, framed_user_turn;
  std::string open_separator, terminated_separator, fresh_user_reference;
  int32_t terminal_anchor; // Exact baseline retained-prefix fixture ID.
};
void CheckInputPolicy(ChatVM &, const InputPolicyExpectations &);
void CheckRenderRevision(const TextRevisionCase &, bool bytecode,
                         const std::string &reference_path);

} // namespace hrx::loomify_gguf::testing
#endif // HRX_SOLUTIONS_LIB_TEXT_TESTING_RENDER_HARNESS_H_
