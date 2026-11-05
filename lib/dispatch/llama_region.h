// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_LLAMA_REGION_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_LLAMA_REGION_H_
#include "lib/dispatch/invocation.h"
#include "lib/llama/structure.h"
#include <memory>
namespace hrx::dispatch {
// Selected existing operations, not another graph or numerical recipe. The
// retained owner keeps ValueIds and descriptor-coordinate associations alive.
// Prefill uses a configured native active-row mapping; decode has one row.
class FfnRegion {
public:
  static bool Select(std::shared_ptr<const llama::LlamaStructure> owner,
                     llama::LlamaRegion selection, const Registry &registry,
                     std::unique_ptr<FfnRegion> *out, std::string *error);
  // Caller configures LlamaRows and binds the external residual before
  // emission. Named parameters and all existing logical outputs retain their
  // owner. Discard the native owner after a staging failure.
  bool Emit(NativeEmissionSession &session, loomify_gguf::Value parameters,
            std::string *error) const;
  const llama::LlamaStructure &owner() const { return *owner_; }
  const llama::LlamaRegionView &view() const { return view_; }
  const std::vector<Id> &dispatches() const { return dispatches_; }
private:
  FfnRegion(std::shared_ptr<const llama::LlamaStructure> owner,
            llama::LlamaRegionView view, std::vector<Id> dispatches)
      : owner_(std::move(owner)), view_(std::move(view)),
        dispatches_(std::move(dispatches)) {}
  std::shared_ptr<const llama::LlamaStructure> owner_;
  llama::LlamaRegionView view_;
  std::vector<Id> dispatches_;
};
} // namespace hrx::dispatch
#endif
