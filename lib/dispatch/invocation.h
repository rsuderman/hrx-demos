// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_INVOCATION_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_INVOCATION_H_
#include "lib/dispatch/registry.h"
#include "lib/dispatch/shapes/result.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"
#include <utility>

namespace hrx::dispatch {
// Shared native staging over one existing graph and command body. The graph
// stays immutable here; native handles retain the existing module/body scope.
// Caller retains the graph and native owners. Discard them after staging
// failure.
class NativeEmissionSession {
public:
  NativeEmissionSession(const Registry &registry,
                        const operations::OperationGraph &graph,
                        loomify_gguf::ModuleBuilder &module,
                        loomify_gguf::CommandProgramBuilder &program);
  // Explicit boundary bindings. Parameter declares a named immutable asset;
  // dependent native parameter views are staged internally at each invocation.
  loomc_status_t Bind(operations::ValueId logical, loomify_gguf::Value native);
  loomc_status_t Parameter(operations::ValueId logical,
                           loomify_gguf::Value parameters, std::string key);
  // Prefill families consume existing control state (count at byte offset
  // zero). Full-row static graph spans are supported here; active subspans need
  // their native mapping before admission. Decode needs no external control
  // buffer.
  loomc_status_t Control(loomify_gguf::Value control);
  const operations::OperationGraph &graph() const { return graph_; }
  // An explicit compile-time proof-entry assertion, not inferred from capacity:
  // these exact I64 control identities mean begin=0/count=1 for this
  // entrypoint. Configure once before emission. Prefill still needs control
  // count field=1.
  loomc_status_t SpecializeOneRow(operations::ValueId row_begin,
                                  operations::ValueId row_count);
  bool one_row_specialized() const { return one_row_; }
  // Emit the actual invocation owned by graph(), in forward append order.
  // Outputs/scratch are native-only: no logical values or operations are added.
  loomc_status_t EmitExisting(Id dispatch,
                              const operations::Invocation &invocation,
                              const std::string &variant = {});
  // Returns precisely the scoped native buffer used by the emitted call.
  loomc_status_t Native(operations::ValueId logical,
                        loomify_gguf::Value *out) const;

private:
  friend class InvocationProgram;
  loomc_status_t Emit(Id dispatch, const operations::OperationAttrs &attributes,
                      const std::vector<shapes::Operand> &inputs,
                      operations::ValueId existing_result,
                      shapes::Result *out_result,
                      loomify_gguf::Value *out_native,
                      const std::string &variant);
  struct Binding {
    operations::ValueId logical;
    loomify_gguf::Value native;
    std::string parameter;
  };
  const Binding *Find(operations::ValueId logical) const;
  loomc_status_t CheckBuffer(loomify_gguf::Value native) const;
  loomc_status_t FailMutation(loomc_status_t status);
  const Registry &registry_;
  const operations::OperationGraph &graph_;
  loomify_gguf::ModuleBuilder &module_;
  loomify_gguf::CommandProgramBuilder &program_;
  loomify_gguf::CommandValues values_;
  std::vector<Binding> bindings_;
  loomify_gguf::Value control_;
  bool has_control_ = false;
  bool failed_ = false;
  bool started_ = false;
  bool one_row_ = false;
  operations::ValueId row_begin_, row_count_;
  bool has_emitted_existing_ = false;
  size_t last_existing_ = 0;
};
// Mutable output-producing authoring. Native emission uses the same session as
// existing-operation emission, then this wrapper adds its result/operation
// once.
class InvocationProgram {
public:
  InvocationProgram(const Registry &registry, operations::OperationGraph &graph,
                    loomify_gguf::ModuleBuilder &module,
                    loomify_gguf::CommandProgramBuilder &program)
      : graph_(graph), session_(registry, graph, module, program) {}
  loomc_status_t Bind(operations::ValueId logical, loomify_gguf::Value native) {
    return session_.Bind(logical, native);
  }
  loomc_status_t Parameter(operations::ValueId logical,
                           loomify_gguf::Value parameters, std::string key) {
    return session_.Parameter(logical, parameters, std::move(key));
  }
  loomc_status_t Control(loomify_gguf::Value control) {
    return session_.Control(control);
  }
  loomc_status_t Native(operations::ValueId logical,
                        loomify_gguf::Value *out) const {
    return session_.Native(logical, out);
  }
  loomc_status_t Invoke(Id dispatch,
                        const operations::OperationAttrs &attributes,
                        const std::vector<shapes::Operand> &inputs,
                        operations::ValueId *out,
                        const std::string &variant = {});

private:
  operations::OperationGraph &graph_;
  NativeEmissionSession session_;
};
} // namespace hrx::dispatch
#endif
