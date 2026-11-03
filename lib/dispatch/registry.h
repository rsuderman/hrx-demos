// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_REGISTRY_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_REGISTRY_H_
#include "lib/dispatch/shapes/result.h"
#include "lib/operations/operation_graph.h"
#include "loomc/module.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"
#include <string>
#include <vector>
namespace hrx::dispatch {
enum class Kind { Kernel, CommandFamily };
enum class Id { RmsNorm, GateUpPrefill, GateUpDecode, DownPrefill, DownDecode };
struct OperandContract {
  operations::Port port;
  operations::Access access;
};
// A synchronous request borrows the logical owner; no native handles are
// retained by registrations. Supports is pure, and Build stages a native
// producer.
struct InvocationRequest {
  const operations::OperationGraph &graph;
  const operations::OperationAttrs &attributes;
  const std::vector<shapes::Operand> &inputs;
  const shapes::Result &result;
  const operations::TensorType *Type(operations::Port port) const;
};
struct NativeOperand {
  operations::Port port;
  operations::ValueId logical;
  loomify_gguf::Value native;
  std::string parameter;
};
struct BuildContext {
  const InvocationRequest &request;
  loomify_gguf::ModuleBuilder &module;
  loomify_gguf::CommandProgramBuilder &program;
  loomify_gguf::CommandValues &values;
  const std::vector<NativeOperand> &inputs;
  loomify_gguf::Value control;
  bool has_control;
  // Existing output or declared alias; empty for a fresh producer. The builder
  // allocates the fresh result and scratch internally through Allocate.
  loomify_gguf::Value output;
  bool has_output = false;
  bool staged = false;
  const NativeOperand *Operand(operations::Port port) const;
  loomc_status_t Allocate(uint64_t bytes, loomify_gguf::Value *out);
};
struct Registration;
struct OperationBuilder {
  operations::OpCode operation;
  Kind kind;
  std::vector<OperandContract> operands;
  bool (*supports)(const Registration &, const InvocationRequest &,
                   std::string *) = nullptr;
  loomc_status_t (*build)(const Registration &, BuildContext &,
                          loomify_gguf::Value *) = nullptr;
  std::vector<std::string> required_definitions;
};
OperationBuilder NormBuilder();
OperationBuilder FfnBuilder(operations::OpCode operation);
// Semantic facts supplement native declarations; they never redefine the ABI.
// Entries sharing an id/name are ordered named implementations of one
// operation.
struct Registration {
  Id id;
  operations::OpCode operation;
  Kind kind;
  std::string name;
  std::string symbol;
  std::string interface_file;
  std::vector<std::string> provider_files;
  std::vector<OperandContract> operands;
  std::vector<std::string> required_config;
  std::string variant = "default";
  operations::Phase phase = operations::Phase::Prefill;
  // Native family declaration used to check a selected implementation's ABI.
  std::string interface_symbol;
  OperationBuilder builder{};
};
class Registry {
public:
  static Registry Llama3B();
  const Registration *Find(Id id) const;
  const Registration *Find(const std::string &name) const;
  std::vector<Registration> Implementations(Id id) const;
  // Empty variant selects the first supporting entry in registration order.
  // An explicit variant never silently falls back.
  const Registration *Resolve(Id id, const InvocationRequest &request,
                              const std::string &variant,
                              std::string *error) const;
  // Rejects duplicate variant keys and incompatible semantic contracts.
  bool Add(Registration entry, std::string *error);

private:
  std::vector<Registration> entries_;
};
// Normal authoring loads operation IDs; registered implementations and their
// source closure are collected internally, so models do not enumerate variants.
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const Registry &registry, const std::vector<Id> &operations,
                    const std::string &source_root, loomc_module_t **out,
                    loomc_result_t **result);
// Loads the explicit interface and complete owned source closure, then uses
// native MERGE, required-config inspection and strict LINK verification to
// construct one authoring-visible module. source_root is supplied by the
// build/runfiles caller, never embedded in data. On failure *out is null. The
// caller releases both module and result; native diagnosed failures may return
// an OK API status with a failed result.
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const Registration &entry, const std::string &source_root,
                    loomc_module_t **out, loomc_result_t **result);
// Load one deduplicated declaration/provider closure before authoring bodies.
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const std::vector<Registration> &entries,
                    const std::string &source_root, loomc_module_t **out,
                    loomc_result_t **result);
} // namespace hrx::dispatch
#endif
