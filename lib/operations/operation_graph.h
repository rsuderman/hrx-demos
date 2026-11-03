// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_OPERATION_GRAPH_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_OPERATION_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace hrx::operations {

enum class OpCode {
  Embedding,
  RmsNorm,
  Quantize,
  Linear,
  GatedMatmul,
  LinearResidual,
  Rope,
  RotatePublishKV,
  CausalAttention,
  CopyRows,
  ArgMax,
  PublishToken
};
enum class Port {
  Input,
  Weight,
  GateWeight,
  UpWeight,
  Residual,
  Output,
  TokenIds,
  Gamma,
  RowBegin,
  RowCount,
  Positions,
  Factors,
  Query,
  Key,
  Value,
  KeyCache,
  ValueCache,
  Token,
  Control,
  History,
  Progress
};
enum class ValueKind { Tensor, Scalar, State };
enum class Access { Read, Write, ReadWrite };
// Element domains, independent of access mode. Whole covers the entire value.
// ActiveRows covers tensor axis zero [RowBegin, RowBegin+RowCount), or all
// rows when the optional pair is absent. Runtime requires 0 <= begin < R and
// 1 <= count <= R-begin; scalar declarations do not prove these bounds.
// CopyRowsSource covers that range for Range, or begin+count-1 for LastActive.
// CopyRowsDestination covers [0,count) for Range, or row zero for LastActive.
// This destination is Write: copying reads no previous destination contents.
// All remaining axes are included. ReadWrite preserves elements outside its
// domain; Write defines only its domain, leaving inactive elements undefined.
// A downstream read must stay inside initialized elements; graph type checking
// does not prove initialization, runtime bounds or native adapter behavior.
enum class AccessDomain {
  Whole,
  ActiveRows,
  CopyRowsSource,
  CopyRowsDestination,
  // Cache rows indexed by Positions over active query rows; all head axes.
  ActiveTokenPositions,
  // Union of initialized logical [0, Positions[row]+1) cache prefixes.
  CausalPrefix
};
enum class ElementType { F32, F16, I8, I32, I64 };
enum class ScalarType { F32, I32, I64 };
enum class Encoding { Dense, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q8_1X4 };
enum class Layout { RowMajor, GgmlWeightRows, PackedQ8_1X4Rows };
enum class StateKind { KVCache };
enum class Activation { SiLU };

struct GatedMatmulAttrs {
  uint64_t input_width = 0;
  uint64_t ffn_width = 0;
  Activation activation = Activation::SiLU;
};
struct LinearResidualAttrs {
  uint64_t input_width = 0;
  uint64_t output_width = 0;
};
enum class PackingPolicy { Canonical, QkCompat };
enum class NumericalPolicy { F32Accumulate };
struct EmbeddingAttrs {
  uint64_t vocabulary_size = 0;
  uint64_t embedding_width = 0;
};
struct RmsNormAttrs {
  uint64_t width = 0;
  double epsilon = 1e-5;
};
struct QuantizeAttrs {
  uint64_t width = 0;
  PackingPolicy policy = PackingPolicy::Canonical;
};
struct LinearAttrs {
  uint64_t input_width = 0;
  uint64_t output_width = 0;
  NumericalPolicy numerical_policy = NumericalPolicy::F32Accumulate;
};
enum class RowSelection { Range, LastActive };
enum class TieBreak { LowestIndex };
struct CopyRowsAttrs {
  uint64_t width = 0;
  RowSelection selection = RowSelection::Range;
};
struct ArgMaxAttrs {
  uint64_t width = 0;
  uint32_t axis = 1;
  TieBreak tie_break = TieBreak::LowestIndex;
};
enum class RopeScaling { None, Linear };
enum class FactorMode { None, PerPair };
struct RopeAttrs {
  uint64_t head_count = 0;
  uint64_t head_width = 0;
  uint64_t rotary_width = 0;
  double base = 10000;
  RopeScaling scaling = RopeScaling::Linear;
  double scaling_factor = 1;
  FactorMode factor_mode = FactorMode::None;
};
struct RotatePublishKVAttrs {
  uint64_t kv_heads = 0;
  uint64_t key_width = 0;
  uint64_t value_width = 0;
  uint64_t cache_capacity = 0;
  uint64_t rotary_width = 0;
  double base = 10000;
  RopeScaling scaling = RopeScaling::Linear;
  double scaling_factor = 1;
  FactorMode factor_mode = FactorMode::None;
};
struct CausalAttentionAttrs {
  uint64_t query_heads = 0;
  uint64_t kv_heads = 0;
  uint64_t key_width = 0;
  uint64_t value_width = 0;
  uint64_t cache_capacity = 0;
  double scale = 1;
  NumericalPolicy numerical_policy = NumericalPolicy::F32Accumulate;
};
enum class Phase { Prefill, Decode };
enum class TerminalPolicy { None, StopOnTokenIds };
struct PublishTokenAttrs {
  Phase phase = Phase::Prefill;
  uint64_t history_capacity = 0;
  TerminalPolicy terminal_policy = TerminalPolicy::None;
  std::vector<int32_t> terminal_ids;
};
using OperationAttrs =
    std::variant<GatedMatmulAttrs, LinearResidualAttrs, EmbeddingAttrs,
                 RmsNormAttrs, QuantizeAttrs, LinearAttrs, CopyRowsAttrs,
                 ArgMaxAttrs, RopeAttrs, RotatePublishKVAttrs,
                 CausalAttentionAttrs, PublishTokenAttrs>;

class BoundOperands;
class DefinitionBuilder;
class InvocationBuilder;
class OperationGraph;

namespace detail {
struct GraphIdentity;
template <typename T, size_t I = 0> constexpr size_t AttributeIndex() {
  if constexpr (I == std::variant_size_v<OperationAttrs>) {
    static_assert(!std::is_same_v<T, T>, "unsupported operation attributes");
    return I;
  } else if constexpr (std::is_same_v<
                           T, std::variant_alternative_t<I, OperationAttrs>>)
    return I;
  else
    return AttributeIndex<T, I + 1>();
}
template <typename T>
using Verifier = bool (*)(const BoundOperands &, const T &, std::string *);
template <typename Variant> struct VerifierTypes;
template <typename... T> struct VerifierTypes<std::variant<T...>> {
  using Type = std::variant<Verifier<T>...>;
};
using VerifierVariant = typename VerifierTypes<OperationAttrs>::Type;
} // namespace detail

// Non-default handles retain owner identity, not graph data. They remain stable
// across vector growth. valid() does not imply the graph is still alive.
class ValueId {
public:
  ValueId() = default;
  bool valid() const { return bool(identity_); }
  bool operator==(const ValueId &other) const;
  bool operator!=(const ValueId &other) const { return !(*this == other); }

private:
  friend class OperationGraph;
  ValueId(std::shared_ptr<const detail::GraphIdentity> identity, size_t slot);
  std::shared_ptr<const detail::GraphIdentity> identity_;
  size_t slot_ = 0;
};

// Positive concrete rank 1-3 metadata; no allocation or byte-size inference.
// RowMajor means rows/width, GgmlWeightRows means input/output GGML axes.
class TensorType {
public:
  TensorType();
  static TensorType vector(uint64_t d0, ElementType element);
  static TensorType matrix(uint64_t d0, uint64_t d1, ElementType element);
  TensorType &addDimension(uint64_t dimension);
  TensorType &setEncoding(Encoding encoding);
  TensorType &setLayout(Layout layout);
  size_t rank() const { return dimensions_.size(); }
  uint64_t dimension(size_t axis) const;
  ElementType elementType() const { return element_; }
  Encoding encoding() const { return encoding_; }
  Layout layout() const { return layout_; }

private:
  std::vector<uint64_t> dimensions_;
  ElementType element_ = ElementType::F32;
  Encoding encoding_ = Encoding::Dense;
  Layout layout_ = Layout::RowMajor;
};

class ValueDescription {
public:
  ValueId id() const { return id_; }
  ValueKind kind() const;
  const TensorType *tensorType() const;
  const ScalarType *scalarType() const;
  const StateKind *stateKind() const;

private:
  friend class OperationGraph;
  ValueDescription(ValueId id,
                   std::variant<TensorType, ScalarType, StateKind> type);
  ValueId id_;
  std::variant<TensorType, ScalarType, StateKind> type_;
};

class PortDefinition {
public:
  Port port() const { return port_; }
  ValueKind kind() const { return kind_; }
  Access access() const { return access_; }
  AccessDomain domain() const { return domain_; }
  bool required() const { return required_; }

private:
  friend class DefinitionBuilder;
  PortDefinition(Port port, ValueKind kind, Access access, bool required,
                 AccessDomain domain);
  Port port_;
  ValueKind kind_;
  Access access_;
  AccessDomain domain_;
  bool required_;
};

class OperationDefinition {
public:
  OpCode opcode() const { return opcode_; }
  const std::vector<PortDefinition> &ports() const { return ports_; }
  const PortDefinition *port(Port port) const;
  template <typename T> bool expectsAttributes() const {
    return attribute_index_ == detail::AttributeIndex<T>();
  }

private:
  friend class DefinitionBuilder;
  friend class OperationGraph;
  friend class InvocationBuilder;
  OperationDefinition() = default;
  OpCode opcode_ = OpCode::GatedMatmul;
  std::vector<PortDefinition> ports_;
  size_t attribute_index_ = 0;
  detail::VerifierVariant verifier_;
};

class OperationRegistry {
public:
  OperationRegistry() = default;
  OperationRegistry(const OperationRegistry &) = delete;
  OperationRegistry &operator=(const OperationRegistry &) = delete;
  OperationRegistry(OperationRegistry &&) = delete;
  OperationRegistry &operator=(OperationRegistry &&) = delete;
  DefinitionBuilder define(OpCode opcode);
  const OperationDefinition *find(OpCode opcode) const;
  const std::vector<OperationDefinition> &definitions() const {
    return definitions_;
  }

private:
  friend class DefinitionBuilder;
  std::vector<OperationDefinition> definitions_;
};

// Move-only staging; owner must outlive the builder. finish is terminal even on
// failure. Expected validation failures do not change the registry.
class DefinitionBuilder {
public:
  DefinitionBuilder(const DefinitionBuilder &) = delete;
  DefinitionBuilder &operator=(const DefinitionBuilder &) = delete;
  DefinitionBuilder(DefinitionBuilder &&other) noexcept;
  DefinitionBuilder &operator=(DefinitionBuilder &&other) noexcept;
  DefinitionBuilder &port(Port port, ValueKind kind, Access access,
                          AccessDomain domain = AccessDomain::Whole);
  DefinitionBuilder &optionalPort(Port port, ValueKind kind, Access access,
                                  AccessDomain domain = AccessDomain::Whole);
  template <typename T> DefinitionBuilder &attributes() {
    setAttributes(detail::AttributeIndex<T>());
    return *this;
  }
  template <typename T>
  DefinitionBuilder &verify(bool (*fn)(const BoundOperands &, const T &,
                                       std::string *)) {
    static_cast<void>(detail::AttributeIndex<T>());
    setVerifier(detail::VerifierVariant(fn));
    return *this;
  }
  bool finish(std::string *error);

private:
  friend class OperationRegistry;
  DefinitionBuilder(OperationRegistry &registry, OpCode opcode);
  bool ready();
  void fail(const std::string &error);
  DefinitionBuilder &addPort(Port port, ValueKind kind, Access access,
                             bool required, AccessDomain domain);
  void setAttributes(size_t index);
  void setVerifier(detail::VerifierVariant verifier);
  OperationRegistry *registry_ = nullptr;
  OperationDefinition staged_;
  bool consumed_ = false;
  bool has_attributes_ = false;
  bool has_verifier_ = false;
  std::string error_;
};

class PortBinding {
public:
  Port port() const { return port_; }
  ValueId value() const { return value_; }

private:
  friend class InvocationBuilder;
  PortBinding(Port port, ValueId value);
  Port port_;
  ValueId value_;
};

class Invocation {
public:
  OpCode opcode() const { return opcode_; }
  const std::vector<PortBinding> &bindings() const { return bindings_; }
  const PortBinding *binding(Port port) const;
  const OperationAttrs &attributes() const { return attributes_; }

private:
  friend class InvocationBuilder;
  Invocation() = default;
  OpCode opcode_ = OpCode::GatedMatmul;
  std::vector<PortBinding> bindings_;
  OperationAttrs attributes_;
};

// Only valid during the typed verifier call. No retained pointers or mutation.
class BoundOperands {
public:
  const ValueDescription *value(Port port) const;
  const TensorType *tensor(Port port) const;
  bool sameValue(Port a, Port b) const;

private:
  friend class InvocationBuilder;
  BoundOperands(const OperationGraph &graph, const Invocation &invocation);
  const OperationGraph &graph_;
  const Invocation &invocation_;
};

// Stages one direct invocation. Attributes are copied immediately. The graph
// must outlive the builder; append is terminal and publishes nothing on
// failure.
class InvocationBuilder {
public:
  InvocationBuilder(const InvocationBuilder &) = delete;
  InvocationBuilder &operator=(const InvocationBuilder &) = delete;
  InvocationBuilder(InvocationBuilder &&other) noexcept;
  InvocationBuilder &operator=(InvocationBuilder &&other) noexcept;
  InvocationBuilder &bind(Port port, ValueId value);
  template <typename T> InvocationBuilder &attributes(const T &attrs) {
    static_cast<void>(detail::AttributeIndex<T>());
    setAttributes(OperationAttrs(attrs));
    return *this;
  }
  bool append(std::string *error);

private:
  friend class OperationGraph;
  InvocationBuilder(OperationGraph &graph, OpCode opcode);
  bool ready();
  void fail(const std::string &error);
  void setAttributes(OperationAttrs attrs);
  OperationGraph *graph_ = nullptr;
  Invocation staged_;
  bool consumed_ = false;
  bool has_attributes_ = false;
  std::string error_;
};

// Owns a snapshot of completed definitions and copied typed value declarations.
// Inspection references are borrowed until owner mutation/destruction; retain
// ValueIds instead of vector element pointers. Invocations preserve append
// order; access modes describe mutable-value updates, not implicit SSA results.
// No thread-safety, initialization or physical alias proof implied.
class OperationGraph {
public:
  explicit OperationGraph(const OperationRegistry &registry);
  OperationGraph(const OperationGraph &) = delete;
  OperationGraph &operator=(const OperationGraph &) = delete;
  OperationGraph(OperationGraph &&) = delete;
  OperationGraph &operator=(OperationGraph &&) = delete;
  ValueId tensor(const TensorType &type, std::string *error);
  ValueId scalar(ScalarType type, std::string *error);
  ValueId state(StateKind kind, std::string *error);
  InvocationBuilder operation(OpCode opcode);
  const ValueDescription *value(ValueId id) const;
  const std::vector<ValueDescription> &values() const { return values_; }
  const OperationDefinition *definition(OpCode opcode) const;
  const std::vector<Invocation> &invocations() const { return invocations_; }

private:
  friend class InvocationBuilder;
  ValueId addValue(std::variant<TensorType, ScalarType, StateKind> type,
                   std::string *error);
  std::shared_ptr<const detail::GraphIdentity> identity_;
  std::vector<OperationDefinition> definitions_;
  std::vector<ValueDescription> values_;
  std::vector<Invocation> invocations_;
};

const char *OpCodeName(OpCode opcode);
const char *PortName(Port port);
const char *AccessName(Access access);

} // namespace hrx::operations

#endif // HRX_SOLUTIONS_LIB_OPERATIONS_OPERATION_GRAPH_H_
