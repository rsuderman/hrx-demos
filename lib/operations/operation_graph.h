// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_OPERATION_GRAPH_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_OPERATION_GRAPH_H_

#include <array>
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
enum class Port { Input, Weight, GateWeight, UpWeight, Residual, Output };
enum class ValueKind { Tensor, Scalar, State };
enum class Access { Read, Write, ReadWrite };
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
using OperationAttrs = std::variant<GatedMatmulAttrs, LinearResidualAttrs>;

class BoundOperands;
class DefinitionBuilder;
class InvocationBuilder;
class OperationGraph;

namespace detail {
struct GraphIdentity;
template <typename T> constexpr size_t AttributeIndex() {
  static_assert(std::is_same_v<T, GatedMatmulAttrs> ||
                    std::is_same_v<T, LinearResidualAttrs>,
                "unsupported operation attributes");
  if constexpr (std::is_same_v<T, GatedMatmulAttrs>)
    return 0;
  else
    return 1;
}
template <typename T>
using Verifier = bool (*)(const BoundOperands &, const T &, std::string *);
using VerifierVariant =
    std::variant<Verifier<GatedMatmulAttrs>, Verifier<LinearResidualAttrs>>;
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

// Concrete two-axis metadata only; no allocation or byte-size inference.
// RowMajor means rows/width, GgmlWeightRows means input/output GGML axes.
class TensorType {
public:
  TensorType();
  static TensorType matrix(uint64_t d0, uint64_t d1, ElementType element);
  TensorType &setEncoding(Encoding encoding);
  TensorType &setLayout(Layout layout);
  size_t rank() const { return rank_; }
  uint64_t dimension(size_t axis) const;
  ElementType elementType() const { return element_; }
  Encoding encoding() const { return encoding_; }
  Layout layout() const { return layout_; }

private:
  std::array<uint64_t, 2> dimensions_;
  size_t rank_ = 0;
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

private:
  friend class DefinitionBuilder;
  PortDefinition(Port port, ValueKind kind, Access access);
  Port port_;
  ValueKind kind_;
  Access access_;
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
  DefinitionBuilder &port(Port port, ValueKind kind, Access access);
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
