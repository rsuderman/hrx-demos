// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_PROBE_RUNNER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_PROBE_RUNNER_H_

#include "lib/loomify/text/jinja/compiler.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

typedef struct iree_tokenizer_t iree_tokenizer_t;

namespace hrx::loomify_gguf::jinja {

// An owned CPU instance of an actual compiled template. Callers supply declared
// helper resources and an existing scratch parent; no serving/model state is
// consulted. Instances support sequential synchronous renders, not concurrent
// invocation. Native compiler/runtime objects remain private.
class PreparedTemplate {
public:
  ~PreparedTemplate();
  PreparedTemplate(const PreparedTemplate &) = delete;
  PreparedTemplate &operator=(const PreparedTemplate &) = delete;

  // Prepare once through strict native LINK and SOURCE or bytecode loading.
  // Helpers must be relative resource paths without traversal. A unique owned
  // scratch child is removed on failure/destruction. Failure preserves *out.
  // An optional native tokenizer supplies the existing input module's encoding
  // capabilities. It is borrowed and must outlive the returned instance.
  static bool Prepare(const GeneratedInputRenderer &generated,
                      const std::string &helper_root,
                      const std::string &scratch_parent, bool bytecode,
                      std::unique_ptr<PreparedTemplate> *out,
                      std::string *error,
                      const iree_tokenizer_t *tokenizer = nullptr);

  // Invokes the same prepared program. Copies result bytes before releasing VM
  // variants, including when the return aliases an argument. Failure preserves
  // *out and does not prevent a subsequent render.
  // When requested, native_status_code receives the actual native status code
  // (zero on success); diagnostics consumers need not guess its classification
  // from message text. This does not expose native handles or status ownership.
  bool Render(std::string_view environment_json, std::string *out,
              std::string *error, int32_t *native_status_code = nullptr);

  // Invokes a prepared generated selector with its two-scalar result ABI.
  // Failure preserves both outputs; no private scanner state is interpreted.
  bool SelectFormat(std::string_view environment_json, int64_t *format_id,
                    int64_t *startup_mode, std::string *error);

private:
  struct Impl;
  PreparedTemplate();
  std::unique_ptr<Impl> impl_;
};

} // namespace hrx::loomify_gguf::jinja

#endif // HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_PROBE_RUNNER_H_
