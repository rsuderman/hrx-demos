// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LOOMIFY_COMMAND_PROGRAM_BUILDER_H_
#define HRX_SOLUTIONS_LOOMIFY_COMMAND_PROGRAM_BUILDER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loomc/module.h"

namespace hrx::loomify_gguf {
namespace detail {
struct AuthoringState;
}
class ModuleBuilder;
class CommandProgramBuilder;
class RootRef;

// These references are identities in one mutable module, never raw cross-module
// IDs. Kernel references can be reused by any command definition in that
// module.
class KernelRef {
public:
  KernelRef() = default;

private:
  friend class ModuleBuilder;
  friend class CommandProgramBuilder;
  friend class RootRef;
  uint64_t owner_ = 0;
  loom_symbol_id_t symbol_ = LOOM_SYMBOL_ID_INVALID;
};
class KernelPipelineRef {
public:
  KernelPipelineRef() = default;

private:
  friend class ModuleBuilder;
  friend class CommandProgramBuilder;
  friend class RootRef;
  uint64_t owner_ = 0;
  loom_symbol_id_t symbol_ = LOOM_SYMBOL_ID_INVALID;
};
class ProgramRef {
public:
  ProgramRef() = default;

private:
  friend class ModuleBuilder;
  friend class CommandProgramBuilder;
  friend class RootRef;
  uint64_t owner_ = 0;
  loom_symbol_id_t symbol_ = LOOM_SYMBOL_ID_INVALID;
};
class RootRef {
public:
  RootRef() = default;
  RootRef(ProgramRef ref) : owner_(ref.owner_), symbol_(ref.symbol_) {}
  RootRef(KernelPipelineRef ref) : owner_(ref.owner_), symbol_(ref.symbol_) {}
  RootRef(KernelRef ref) : owner_(ref.owner_), symbol_(ref.symbol_) {}

private:
  friend class ModuleBuilder;
  uint64_t owner_ = 0;
  loom_symbol_id_t symbol_ = LOOM_SYMBOL_ID_INVALID;
};
class Value {
public:
  Value() = default;

private:
  friend class ModuleBuilder;
  friend class CommandProgramBuilder;
  uint64_t owner_ = 0;
  uint64_t scope_ = 0;
  loom_value_id_t id_ = LOOM_VALUE_ID_INVALID;
};
class Type {
public:
  Type() = default;

private:
  friend class ModuleBuilder;
  uint64_t owner_ = 0;
  loom_type_t type_ = {};
};

// One host-side owner for definitions, symbol identities, providers and
// linking. Not thread safe. The input module is cloned once and is never
// mutated.
class ModuleBuilder {
public:
  static loomc_status_t Create(loomc_context_t *context,
                               loomc_workspace_t *workspace,
                               const loomc_module_t *definitions,
                               std::unique_ptr<ModuleBuilder> *out_builder);
  ~ModuleBuilder();
  ModuleBuilder(const ModuleBuilder &) = delete;
  ModuleBuilder &operator=(const ModuleBuilder &) = delete;

  Type Scalar(loom_scalar_type_t type) const;
  Type Buffer() const;
  loomc_status_t Kernel(std::string_view name, KernelRef *out) const;
  loomc_status_t KernelPipeline(std::string_view name,
                                KernelPipelineRef *out) const;
  loomc_status_t Program(std::string_view name, ProgramRef *out) const;
  // Resolve other existing entry kinds (e.g. functions) for artifact roots.
  loomc_status_t Root(std::string_view name, RootRef *out) const;
  loomc_status_t DeclareKernel(std::string_view name,
                               const std::vector<Type> &workloads,
                               const std::vector<Type> &arguments,
                               KernelRef *out);
  loomc_status_t
  DefineProgram(std::string_view name,
                const std::vector<loom_scalar_type_t> &specialization_types,
                size_t binding_count,
                std::unique_ptr<CommandProgramBuilder> *out);

  // Provider modules are snapshotted; no borrowed provider may be mutated by
  // its caller during finalization. Normal Loom linking owns exact resolution.
  loomc_status_t AddProvider(const loomc_module_t *provider);

  // Extension seam for native definition builders (e.g. PipelineBuilder).
  // The callback must build complete top-level definitions synchronously and
  // must not retain raw IDs/builders. An error terminally fails the whole
  // owner.
  loomc_status_t BuildNative(
      const std::function<iree_status_t(loom_module_t *, loom_builder_t *)>
          &build);

  // Requires all bodies finished and at least one root. Explicit strict LINK,
  // not MERGE. Output is independently owned. A completed link attempt seals
  // this owner, including a failed-result attempt; preflight errors recover.
  loomc_status_t LinkAndVerify(const std::vector<RootRef> &roots,
                               loomc_module_t **out_module,
                               loomc_result_t **out_result);

private:
  friend class CommandProgramBuilder;
  explicit ModuleBuilder(std::shared_ptr<detail::AuthoringState> state);
  std::shared_ptr<detail::AuthoringState> state_;
};

// A body builder in a ModuleBuilder. Finish seals only this definition.
// Destruction without Finish leaves an unfinished definition and blocks output.
class CommandProgramBuilder {
public:
  ~CommandProgramBuilder();
  CommandProgramBuilder(const CommandProgramBuilder &) = delete;
  CommandProgramBuilder &operator=(const CommandProgramBuilder &) = delete;
  ProgramRef Ref() const;
  Value Specialization(size_t index) const;
  Value Binding(size_t index) const;
  loomc_status_t IndexConstant(int64_t value, Value *out);
  loomc_status_t Invoke(KernelRef kernel, const std::vector<Value> &workloads,
                        const std::vector<Value> &arguments);
  loomc_status_t Invoke(KernelPipelineRef pipeline,
                        const std::vector<Value> &specializations,
                        const std::vector<Value> &run_arguments);
  loomc_status_t Call(ProgramRef program,
                      const std::vector<Value> &specializations,
                      const std::vector<Value> &bindings);
  // Read-only preflight seam for native extensions. Inputs are lifecycle and
  // scope checked before the callback. Validation failures remain recoverable;
  // the callback must not retain native IDs or mutate the module.
  loomc_status_t InspectNative(
      const std::vector<Value> &operands,
      const std::function<iree_status_t(const loom_module_t *,
                                        const std::vector<loom_value_id_t> &)>
          &inspect) const;
  // Internal extension seam: input handles are scope checked, result IDs must
  // be defined in this schedule's block. Callbacks must not retain native IDs.
  // A native error or invalid result terminally fails the whole module.
  loomc_status_t BuildNative(
      const std::vector<Value> &operands,
      const std::function<iree_status_t(loom_module_t *, loom_builder_t *,
                                        const std::vector<loom_value_id_t> &,
                                        std::vector<loom_value_id_t> *)> &build,
      std::vector<Value> *results);
  using RegionBody = std::function<loomc_status_t(CommandProgramBuilder &)>;
  loomc_status_t Serial(const RegionBody &body);
  loomc_status_t Concurrent(const RegionBody &body);
  loomc_status_t If(Value condition, const RegionBody &then_body,
                    const RegionBody &else_body = {});
  loomc_status_t Finish();

private:
  friend class ModuleBuilder;
  struct Impl;
  explicit CommandProgramBuilder(std::unique_ptr<Impl> impl);
  loomc_status_t EnterRegion(loom_op_t *op, loom_region_t *region,
                             const RegionBody &body, bool scf_yield);
  loomc_status_t
  InvokeSymbol(loom_symbol_id_t symbol,
               const std::vector<loom_value_id_t> &expected_first,
               const std::vector<loom_value_id_t> &expected_second,
               const std::vector<Value> &first,
               const std::vector<Value> &second, bool remap_first,
               bool program_call);
  std::unique_ptr<Impl> impl_;
};

} // namespace hrx::loomify_gguf
#endif // HRX_SOLUTIONS_LOOMIFY_COMMAND_PROGRAM_BUILDER_H_
