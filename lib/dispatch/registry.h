// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_REGISTRY_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_REGISTRY_H_
#include "lib/operations/operation_graph.h"
#include "loomc/module.h"
#include <string>
#include <vector>
namespace hrx::dispatch {
enum class Kind { Kernel, CommandFamily };
enum class Id { RmsNorm, GateUpPrefill, GateUpDecode, DownPrefill, DownDecode };
struct OperandContract {
  operations::Port port;
  operations::Access access;
};
// Semantic facts supplement native declarations; they never redefine the ABI.
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
};
class Registry {
public:
  static Registry Llama3B();
  const Registration *Find(Id id) const;
  const Registration *Find(const std::string &name) const;
  // Rejects duplicate and incompatible known semantic contracts.
  bool Add(Registration entry, std::string *error);

private:
  std::vector<Registration> entries_;
};
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
// The curated provider supports precisely this profile. Native family operands
// must use generic=true, compat=true, Q4 gate/up and Q4/Q6 down. Other profile
// choices fail admission even when their native ABIs match.
bool CheckProfile(const Registration &entry, uint64_t hidden, uint64_t ffn,
                  uint64_t rows, operations::Encoding weight,
                  operations::PackingPolicy packing, std::string *error);
} // namespace hrx::dispatch
#endif
