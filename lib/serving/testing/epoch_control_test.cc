// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual SDK input.require plus maintained host execution products. The runner
// records submissions only: this tests host admission/progress, never GPU math.
// Built in the SDK's existing allowed external test package; the repository
// exports this source without depending on private SDK BUILD targets.
#include "gtest/gtest.h"
#include "iree/module/hal/types.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/target/vm.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
std::string Consume(iree_status_t status) {
  if (iree_status_is_ok(status)) return {};
  char message[4096] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  iree_status_ignore(status);
  return message;
}
bool Ok(iree_status_t status) {
  if (iree_status_is_ok(status)) return true;
  ADD_FAILURE() << Consume(status);
  return false;
}
using Field = iree_vm_module_signature_type_t;
constexpr Field kFeedbackIn[] = {{IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
constexpr Field kExecuteIn[] = {{IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr Field kCompletion[] = {{IREE_VM_SCALAR_TYPE_I64, 0}};
// Canonical type order is independent of alphabetically ordered exports.
const iree_vm_module_callable_type_declaration_t kRunnerTypes[] = {
    {{{kFeedbackIn, 4, 3, 1, 0}, {kCompletion, 1, 1, 0, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0},
    {{{kExecuteIn, 7, 1, 6, 0}, {kCompletion, 1, 1, 0, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0}};
const iree_vm_module_export_declaration_t kRunnerExports[] = {
    {IREE_SVL("execute_6"), 1, 0, 0}, {IREE_SVL("feedback"), 0, 1, 0}};
struct Runner {
  iree_vm_module_t module{};
  iree_vm_module_descriptor_t descriptor{};
  bool initialized = false;
  bool allow_execute = false, allow_feedback = false;
  int executes = 0, feedbacks = 0;
  int32_t stage = -1, slot = -1;
  int64_t offset = -1, length = -1;
  void Reset(bool execute = false, bool feedback = false) {
    allow_execute = execute;
    allow_feedback = feedback;
    executes = feedbacks = 0;
    stage = slot = -1;
    offset = length = -1;
  }
};
iree_status_t Start(iree_vm_module_t* module,
                    const iree_vm_module_function_start_params_t* params,
                    iree_vm_execution_outcome_t* outcome) {
  auto& runner = *reinterpret_cast<Runner*>(module);
  const auto& call = params->call;
  if (params->function_ordinal == 0) {
    ++runner.executes;
    runner.stage = static_cast<int32_t>(iree_vm_call_value_argument_load(&call, 0));
    if (!runner.allow_execute)
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "unexpected host execute_6");
    iree_vm_call_value_result_store(&call, 0, 77);
  } else if (params->function_ordinal == 1) {
    ++runner.feedbacks;
    runner.slot = static_cast<int32_t>(iree_vm_call_value_argument_load(&call, 0));
    runner.offset = static_cast<int64_t>(iree_vm_call_value_argument_load(&call, 1));
    runner.length = static_cast<int64_t>(iree_vm_call_value_argument_load(&call, 2));
    if (!runner.allow_feedback)
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "unexpected host feedback");
    iree_vm_call_value_result_store(&call, 0, 88);
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unknown test runner export");
  }
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}
void QueryExport(const iree_vm_module_t*, iree_host_size_t index,
                 iree_vm_module_export_declaration_t* out) { *out = kRunnerExports[index]; }
void QueryType(const iree_vm_module_t*, iree_host_size_t index,
               iree_vm_module_callable_type_declaration_t* out) { *out = kRunnerTypes[index]; }
void QueryGroup(const iree_vm_module_t*, iree_host_size_t, iree_vm_module_import_group_t*) {
  IREE_CHECK_UNREACHABLE("recording runner has no imports");
}
void QueryImport(const iree_vm_module_t*, iree_host_size_t, iree_vm_module_import_declaration_t*) {
  IREE_CHECK_UNREACHABLE("recording runner has no imports");
}
void Destroy(iree_vm_module_t*) {}
const iree_vm_module_vtable_t kRunnerVtable = {
    sizeof(kRunnerVtable), IREE_VM_MODULE_ABI_VERSION_0, Destroy, Start,
    iree_vm_module_function_resume_unreachable, nullptr, nullptr, nullptr,
    QueryGroup, QueryImport, QueryExport, QueryType,
    iree_vm_module_query_presentation_none, iree_vm_module_metadata_by_ordinal_none};
struct Arguments {
  std::vector<iree_vm_variant_t> values, results;
  ~Arguments() {
    iree_vm_variant_span_reset({values.data(), values.size()});
    iree_vm_variant_span_reset({results.data(), results.size()});
  }
  void I32(int32_t value) { values.push_back(iree_vm_variant_from_i32(value)); }
  void Null() { values.push_back(iree_vm_variant_null()); }
  bool Buffer(const iree_vm_ref_types_t& types, void* data, size_t bytes,
              bool write = false) {
    iree_vm_buffer_t* buffer = nullptr;
    iree_vm_buffer_access_flags_t access = IREE_VM_BUFFER_ACCESS_FLAG_READ;
    if (write) access |= IREE_VM_BUFFER_ACCESS_FLAG_WRITE;
    if (!Ok(iree_vm_buffer_wrap(access, iree_make_byte_span(data, bytes),
        iree_vm_buffer_release_callback_null(), iree_allocator_system(), &buffer))) return false;
    values.push_back(iree_vm_buffer_variant_from_ptr_move(&types, &buffer));
    return true;
  }
  template <class T> bool Buffer(const iree_vm_ref_types_t& types,
                                 std::vector<T>& data, bool write = false) {
    return Buffer(types, data.data(), data.size() * sizeof(T), write);
  }
};
enum class Family { kLlama, kQwen, kGemma };
struct Product { const char* name; const char* variable; Family family; };
class EpochControlTest : public ::testing::TestWithParam<Product> {
 protected:
  void SetUp() override {
    const char* path = std::getenv(GetParam().variable);
    ASSERT_TRUE(path && *path) << "required maintained execution product: " << GetParam().variable;
    auto status = loomc_target_environment_create_vm(loomc_allocator_system(), &target_);
    if (!loomc_status_is_ok(status)) {
      const auto m = loomc_status_message(status);
      ADD_FAILURE() << std::string(m.data, m.size);
      loomc_status_free(status);
      return;
    }
    loomc_status_free(status);
    ASSERT_TRUE(Ok(iree_vm_environment_allocate(iree_allocator_system(), &environment_)));
    ASSERT_TRUE(Ok(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")), &types_)));
    const iree_vm_ref_type_table_t* hal = nullptr;
    ASSERT_TRUE(Ok(iree_hal_module_register_types(environment_, &hal)));
    ASSERT_TRUE(Ok(iree_hal_module_types_resolve(hal, &hal_)));
    // Real SDK require: false is INVALID_ARGUMENT, never an observing success stub.
    ASSERT_TRUE(Ok(loom_serve_input_module_create(environment_, nullptr,
        iree_allocator_system(), &input_)));
    runner_.descriptor = {IREE_SVL("runner"), IREE_VM_MODULE_FLAG_LINKABLE,
        {&hal_.buffer, 1}, {2, 2, 0, 0, 2, 0, {6, 7, 0}}, 0};
    ASSERT_TRUE(Ok(iree_vm_module_initialize(&kRunnerVtable, &runner_.descriptor, &runner_.module)));
    runner_.initialized = true;
    const iree_string_view_t roots[] = {IREE_SV("encode_state"), IREE_SV("encode_epoch"),
        IREE_SV("publish_epoch"), IREE_SV("epoch"), IREE_SV("step")};
    iree_vm_module_t* libraries[] = {input_, &runner_.module};
    loom_serve_sources_t sources{};
    loom_serve_compilation_report_t report{};
    auto compiled = loom_serve_program_allocate(target_, environment_, &sources,
        iree_make_cstring_view(path), IREE_ARRAYSIZE(roots), roots,
        iree_vm_module_span_from_array(libraries), &report, iree_allocator_system(), &program_);
    if (!iree_status_is_ok(compiled)) {
      for (iree_host_size_t i = 0; i < report.diagnostic_count; ++i) {
        const auto& d = report.diagnostics[i];
        ADD_FAILURE() << std::string(d.message, std::min<uint64_t>(d.message_length, sizeof(d.message)));
      }
    }
    ASSERT_TRUE(Ok(compiled));
  }
  void TearDown() override {
    loom_serve_program_free(program_);
    if (runner_.initialized) iree_vm_module_release(&runner_.module);
    iree_vm_module_release(input_);
    iree_vm_environment_free(environment_);
    loomc_target_environment_release(target_);
  }
  iree_status_t Call(const char* name, Arguments& arguments, size_t result_count) {
    iree_vm_function_t function = iree_vm_function_null();
    auto found = iree_vm_process_lookup_function(loom_serve_program_process(program_),
        IREE_SV("model"), iree_make_cstring_view(name), &function);
    if (!iree_status_is_ok(found)) return found;
    arguments.results.resize(result_count);
    return iree_vm_invoke(loom_serve_program_invocation(program_), function,
        {arguments.values.data(), arguments.values.size()},
        {arguments.results.data(), arguments.results.size()});
  }
  void Reject(iree_status_t status) {
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
    const auto message = Consume(status);
    EXPECT_NE(message.find("invalid request or execution state"), std::string::npos) << message;
    EXPECT_EQ(runner_.executes, 0);
    EXPECT_EQ(runner_.feedbacks, 0);
  }
  iree_status_t EncodeState(int32_t count, int64_t row, std::vector<int64_t>* output) {
    std::vector<int64_t> records(16 * 4, 0);
    records[0] = row; records[1] = 256; records[2] = 768; records[3] = 11;
    const auto original = records;
    output->assign(16 * 4, -99);
    Arguments args;
    if (!args.Buffer(types_, records)) return iree_make_status(IREE_STATUS_INTERNAL);
    args.I32(count);
    if (!args.Buffer(types_, *output, true)) return iree_make_status(IREE_STATUS_INTERNAL);
    auto status = Call("encode_state", args, 0);
    EXPECT_EQ(records, original);
    return status;
  }
  iree_status_t Encode(std::array<int32_t, 8> controls, int32_t row = 0,
                       int32_t flags = 1, int32_t credit = 0) {
    std::vector<int32_t> state(2, 0), plan(32 * 9, 0), inputs{71,72,73};
    std::vector<int32_t> empty, metadata(600, -77), tokens(8, -88);
    plan[0] = 3; plan[1] = 11; plan[2] = 11; plan[3] = row;
    plan[4] = 0; plan[5] = 3; plan[6] = flags; plan[7] = credit;
    const auto original_plan = plan, original_inputs = inputs;
    Arguments args;
    if (!args.Buffer(types_, state) || !args.Buffer(types_, plan) ||
        !args.Buffer(types_, inputs) || !args.Buffer(types_, empty)) return iree_make_status(IREE_STATUS_INTERNAL);
    for (auto value : controls) args.I32(value);
    if (!args.Buffer(types_, metadata, true) || !args.Buffer(types_, tokens, true))
      return iree_make_status(IREE_STATUS_INTERNAL);
    auto status = Call("encode_epoch", args, 1);
    EXPECT_EQ(plan, original_plan); EXPECT_EQ(inputs, original_inputs);
    if (iree_status_is_ok(status)) {
      EXPECT_EQ(metadata[0], 3); EXPECT_EQ(metadata[1], 1); EXPECT_EQ(metadata[2], flags & 1);
      EXPECT_EQ(metadata[3], 3); EXPECT_EQ(metadata[4], 11); EXPECT_EQ(metadata[5], 0);
      EXPECT_EQ(metadata[6], 0); EXPECT_EQ(metadata[7], flags & 1 ? 0 : -1);
      if (flags & 1) EXPECT_EQ(metadata[323], 2);
      EXPECT_EQ(metadata[387], 0); EXPECT_EQ(metadata[388], 0);
      EXPECT_EQ(metadata[515], controls[7]);
      EXPECT_EQ(tokens, (std::vector<int32_t>{71,72,73,0,0,0,0,0}));
    }
    return status;
  }
  iree_status_t Publish(std::array<int32_t, 6> controls, int32_t flags = 1,
                        int32_t raw_id = 91, int32_t published_id = -1) {
    std::vector<int32_t> plan(32 * 9, 0), ids(16, 0), empty, progress(16 * 21, -77);
    plan[0] = 3; plan[1] = 11; plan[2] = 11; plan[3] = 0; plan[6] = flags;
    ids[0] = raw_id;
    const auto original_plan = plan, original_ids = ids;
    Arguments args;
    if (!args.Buffer(types_, plan) || !args.Buffer(types_, empty)) return iree_make_status(IREE_STATUS_INTERNAL);
    for (auto value : controls) args.I32(value);
    if (!args.Buffer(types_, ids) || !args.Buffer(types_, empty) ||
        !args.Buffer(types_, empty) || !args.Buffer(types_, progress, true))
      return iree_make_status(IREE_STATUS_INTERNAL);
    auto status = Call("publish_epoch", args, 0);
    EXPECT_EQ(plan, original_plan); EXPECT_EQ(ids, original_ids);
    if (iree_status_is_ok(status)) {
      std::vector<int32_t> expected(16 * 21, 0);
      expected[0] = expected[1] = 3;
      expected[2] = flags & 1; expected[4] = 14;
      // Default retains the existing Llama fixture's 91/92/93 stop mapping.
      // Forced Gemma rows supply independent expected IDs explicitly below.
      if (published_id < 0)
        published_id = GetParam().family == Family::kLlama ? 93 : 91;
      expected[5] = flags & 1 ? published_id : 0;
      EXPECT_EQ(progress, expected);
    }
    return status;
  }
  iree_status_t Issue(std::array<int32_t, 10> controls) {
    std::vector<int32_t> state(2, 0);
    Arguments args;
    if (!args.Buffer(types_, state)) return iree_make_status(IREE_STATUS_INTERNAL);
    for (auto value : controls) args.I32(value);
    for (int i = 0; i < 12; ++i) args.Null();
    auto status = Call("epoch", args, 1);
    if (iree_status_is_ok(status)) {
      int64_t completion = 0;
      EXPECT_TRUE(Ok(iree_vm_i64_from_variant(args.results[0], &completion)));
      EXPECT_EQ(completion, 77);
    }
    return status;
  }
  iree_status_t Step() {
    std::vector<int32_t> state(2, 0);
    Arguments args;
    if (!args.Buffer(types_, state)) return iree_make_status(IREE_STATUS_INTERNAL);
    args.I32(1); args.I32(11); args.I32(3);
    for (int i = 0; i < 6; ++i) args.Null();
    return Call("step", args, 2);
  }
  loomc_target_environment_t* target_ = nullptr;
  iree_vm_environment_t* environment_ = nullptr;
  iree_vm_ref_types_t types_{};
  iree_hal_module_types_t hal_{};
  iree_vm_module_t* input_ = nullptr;
  loom_serve_program_t* program_ = nullptr;
  Runner runner_{};
};
TEST_P(EpochControlTest, ActualRequireEnforcesOrdinaryHostContract) {
  runner_.Reset();
  std::vector<int64_t> rows;
  ASSERT_TRUE(Ok(EncodeState(1, 0, &rows)));
  std::vector<int64_t> expected_rows(16 * 4, -99);
  expected_rows[0] = 256; expected_rows[1] = 768; expected_rows[3] = 11;
  EXPECT_EQ(rows, expected_rows);
  Reject(EncodeState(2, 0, &rows));
  Reject(EncodeState(1, 1, &rows));
  const int32_t terminal = GetParam().family == Family::kGemma ? 106 : 93;
  const std::array<int32_t, 8> encode = {1,0,3,0,0,8,8,terminal};
  ASSERT_TRUE(Ok(Encode(encode)));
  ASSERT_TRUE(Ok(Encode(encode, 0, 0)));
  for (const auto& change : std::vector<std::pair<size_t,int32_t>>{{0,2},{1,1},{3,1},{4,1}}) {
    auto bad = encode; bad[change.first] = change.second;
    Reject(Encode(bad));
  }
  Reject(Encode(encode, 1));
  Reject(Encode(encode, 0, 2));
  Reject(Encode(encode, 0, 4));
  Reject(Encode(encode, 0, 1, 1));
  const std::array<int32_t, 6> publish = {1,0,0,0,0,0};
  ASSERT_TRUE(Ok(Publish(publish)));
  ASSERT_TRUE(Ok(Publish(publish, 0)));
  for (size_t i = 0; i < publish.size(); ++i) {
    auto bad = publish; bad[i] = i == 0 ? 2 : 1;
    Reject(Publish(bad));
  }
  if (GetParam().family == Family::kGemma) {
    // Actual generated Gemma facts: declared EOS1 and turn-end106 remain
    // distinct; host progress canonicalizes only those stops to terminal106.
    // PAD/BOS/UNK produce ordinary progress even when their decoded text is empty.
    for (const auto& item : std::array<std::pair<int32_t, int32_t>, 6>{{
             {1, 106}, {106, 106}, {0, 0}, {2, 2}, {3, 3}, {17, 17}}}) {
      SCOPED_TRACE("Gemma raw ID " + std::to_string(item.first));
      runner_.Reset();
      ASSERT_TRUE(Ok(Publish(publish, 1, item.first, item.second)));
      EXPECT_EQ(runner_.executes, 0);
      EXPECT_EQ(runner_.feedbacks, 0);
    }
    runner_.Reset();
    ASSERT_TRUE(Ok(Publish(publish, 0, 1, 106)));
    EXPECT_EQ(runner_.executes, 0);
    EXPECT_EQ(runner_.feedbacks, 0);
  }
  const std::array<int32_t, 10> issue = {5,0,0,1,1,1,0,0,0,0};
  runner_.Reset(true, true);
  ASSERT_TRUE(Ok(Issue(issue)));
  EXPECT_EQ(runner_.executes, 1); EXPECT_EQ(runner_.stage, 7);
  EXPECT_EQ(runner_.feedbacks, 1); EXPECT_EQ(runner_.slot, 0);
  EXPECT_EQ(runner_.offset, 0); EXPECT_EQ(runner_.length, 4);
  runner_.Reset(true, false);
  auto no_output = issue; no_output[4] = 0;
  ASSERT_TRUE(Ok(Issue(no_output)));
  EXPECT_EQ(runner_.executes, 1); EXPECT_EQ(runner_.feedbacks, 0);
  for (const auto& change : std::vector<std::pair<size_t,int32_t>>{
      {1,1},{2,1},{3,2},{4,2},{5,2},{7,1},{8,1},{9,1}}) {
    auto bad = issue; bad[change.first] = change.second;
    runner_.Reset();
    Reject(Issue(bad));
  }
  runner_.Reset(); Reject(Step());
  // Errors unwind the actual invocation; valid calls continue to work afterward.
  ASSERT_TRUE(Ok(Encode(encode)));
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, EpochControlTest,
    ::testing::Values(
      Product{"LlamaSource", "LOOMIFY_EPOCH_HOST_LLAMA_SOURCE", Family::kLlama},
      Product{"LlamaBytecode", "LOOMIFY_EPOCH_HOST_LLAMA_BC", Family::kLlama},
      Product{"QwenSource", "LOOMIFY_EPOCH_HOST_QWEN_SOURCE", Family::kQwen},
      Product{"QwenBytecode", "LOOMIFY_EPOCH_HOST_QWEN_BC", Family::kQwen},
      Product{"GemmaSource", "LOOMIFY_EPOCH_HOST_GEMMA_SOURCE", Family::kGemma},
      Product{"GemmaBytecode", "LOOMIFY_EPOCH_HOST_GEMMA_BC", Family::kGemma}),
    [](const ::testing::TestParamInfo<Product>& info) { return info.param.name; });
}  // namespace
