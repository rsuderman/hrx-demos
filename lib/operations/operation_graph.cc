// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/operations/operation_graph.h"

#include <algorithm>

namespace hrx::operations {
namespace detail {
struct GraphIdentity {};
} // namespace detail
namespace {
template <typename E> bool InRange(E value, E first, E last) {
  return value >= first && value <= last;
}
const OperationDefinition *
Find(const std::vector<OperationDefinition> &definitions, OpCode opcode) {
  const auto found =
      std::find_if(definitions.begin(), definitions.end(),
                   [&](const auto &d) { return d.opcode() == opcode; });
  return found == definitions.end() ? nullptr : &*found;
}
bool ValidTensor(const TensorType &type, std::string *error) {
  if (!type.rank() || type.rank() > 3) {
    *error = "tensor requires one to three positive dimensions";
    return false;
  }
  for (size_t axis = 0; axis < type.rank(); ++axis) {
    if (!type.dimension(axis)) {
      *error = "tensor requires positive dimensions";
      return false;
    }
  }
  if (!InRange(type.elementType(), ElementType::F32, ElementType::I64) ||
      !InRange(type.encoding(), Encoding::Dense, Encoding::Q8_1X4) ||
      !InRange(type.layout(), Layout::RowMajor, Layout::PackedQ8_1X4Rows)) {
    *error = "invalid tensor element type, encoding or layout";
    return false;
  }
  bool valid;
  if (type.encoding() == Encoding::Dense)
    valid = type.layout() == Layout::RowMajor ||
            (type.rank() == 2 && type.layout() == Layout::GgmlWeightRows);
  else if (type.encoding() == Encoding::Q8_1X4)
    valid = type.rank() == 2 && type.elementType() == ElementType::F32 &&
            type.layout() == Layout::PackedQ8_1X4Rows;
  else
    valid = type.rank() == 2 && type.elementType() == ElementType::F32 &&
            type.layout() == Layout::GgmlWeightRows;
  if (!valid)
    *error = "incompatible tensor element type, encoding and layout";
  return valid;
}
} // namespace

ValueId::ValueId(std::shared_ptr<const detail::GraphIdentity> identity,
                 size_t slot)
    : identity_(std::move(identity)), slot_(slot) {}
bool ValueId::operator==(const ValueId &other) const {
  return identity_ == other.identity_ && slot_ == other.slot_;
}
TensorType::TensorType() = default;
TensorType TensorType::vector(uint64_t d0, ElementType element) {
  TensorType type;
  type.dimensions_.push_back(d0);
  type.element_ = element;
  return type;
}
TensorType TensorType::matrix(uint64_t d0, uint64_t d1, ElementType element) {
  TensorType type = vector(d0, element);
  type.addDimension(d1);
  return type;
}
TensorType &TensorType::addDimension(uint64_t dimension) {
  dimensions_.push_back(dimension);
  return *this;
}
TensorType &TensorType::setEncoding(Encoding encoding) {
  encoding_ = encoding;
  return *this;
}
TensorType &TensorType::setLayout(Layout layout) {
  layout_ = layout;
  return *this;
}
uint64_t TensorType::dimension(size_t axis) const {
  return axis < dimensions_.size() ? dimensions_[axis] : 0;
}
ValueDescription::ValueDescription(
    ValueId id, std::variant<TensorType, ScalarType, StateKind> type)
    : id_(std::move(id)), type_(std::move(type)) {}
ValueKind ValueDescription::kind() const {
  if (std::holds_alternative<TensorType>(type_))
    return ValueKind::Tensor;
  if (std::holds_alternative<ScalarType>(type_))
    return ValueKind::Scalar;
  return ValueKind::State;
}
const TensorType *ValueDescription::tensorType() const {
  return std::get_if<TensorType>(&type_);
}
const ScalarType *ValueDescription::scalarType() const {
  return std::get_if<ScalarType>(&type_);
}
const StateKind *ValueDescription::stateKind() const {
  return std::get_if<StateKind>(&type_);
}
PortDefinition::PortDefinition(Port port, ValueKind kind, Access access,
                               bool required, AccessDomain domain)
    : port_(port), kind_(kind), access_(access), domain_(domain),
      required_(required) {}
const PortDefinition *OperationDefinition::port(Port port) const {
  const auto found =
      std::find_if(ports_.begin(), ports_.end(),
                   [&](const auto &p) { return p.port() == port; });
  return found == ports_.end() ? nullptr : &*found;
}
DefinitionBuilder OperationRegistry::define(OpCode opcode) {
  return DefinitionBuilder(*this, opcode);
}
const OperationDefinition *OperationRegistry::find(OpCode opcode) const {
  return Find(definitions_, opcode);
}
DefinitionBuilder::DefinitionBuilder(OperationRegistry &registry, OpCode opcode)
    : registry_(&registry) {
  staged_.opcode_ = opcode;
  if (!InRange(opcode, OpCode::Embedding, OpCode::PublishToken))
    fail("invalid operation code");
  else if (registry.find(opcode))
    fail("operation code is already registered");
}
DefinitionBuilder::DefinitionBuilder(DefinitionBuilder &&other) noexcept {
  *this = std::move(other);
}
DefinitionBuilder &
DefinitionBuilder::operator=(DefinitionBuilder &&other) noexcept {
  if (this != &other) {
    registry_ = std::exchange(other.registry_, nullptr);
    staged_ = std::move(other.staged_);
    consumed_ = std::exchange(other.consumed_, true);
    has_attributes_ = other.has_attributes_;
    has_verifier_ = other.has_verifier_;
    error_ = std::move(other.error_);
  }
  return *this;
}
void DefinitionBuilder::fail(const std::string &error) {
  if (error_.empty())
    error_ = error;
}
bool DefinitionBuilder::ready() {
  if (consumed_ || !registry_)
    fail("definition builder is consumed");
  return error_.empty();
}
DefinitionBuilder &DefinitionBuilder::port(Port port, ValueKind kind,
                                           Access access, AccessDomain domain) {
  return addPort(port, kind, access, true, domain);
}
DefinitionBuilder &DefinitionBuilder::optionalPort(Port port, ValueKind kind,
                                                   Access access,
                                                   AccessDomain domain) {
  return addPort(port, kind, access, false, domain);
}
DefinitionBuilder &DefinitionBuilder::addPort(Port port, ValueKind kind,
                                              Access access, bool required,
                                              AccessDomain domain) {
  if (!ready())
    return *this;
  if (!InRange(port, Port::Input, Port::Progress))
    fail("invalid port");
  else if (!InRange(kind, ValueKind::Tensor, ValueKind::State))
    fail("invalid value kind");
  else if (!InRange(access, Access::Read, Access::ReadWrite))
    fail("invalid port access");
  else if (!InRange(domain, AccessDomain::Whole, AccessDomain::CausalPrefix))
    fail("invalid access domain");
  else if (domain != AccessDomain::Whole && kind != ValueKind::Tensor)
    fail("row access domain requires a tensor port");
  else if (staged_.port(port))
    fail("port is already defined");
  else
    staged_.ports_.push_back(
        PortDefinition(port, kind, access, required, domain));
  return *this;
}
void DefinitionBuilder::setAttributes(size_t index) {
  if (!ready())
    return;
  if (has_attributes_) {
    fail("attributes are already declared");
    return;
  }
  staged_.attribute_index_ = index;
  has_attributes_ = true;
}
void DefinitionBuilder::setVerifier(detail::VerifierVariant verifier) {
  if (!ready())
    return;
  if (has_verifier_) {
    fail("verifier is already declared");
    return;
  }
  if (!std::visit([](auto fn) { return fn != nullptr; }, verifier)) {
    fail("verifier must not be null");
    return;
  }
  staged_.verifier_ = std::move(verifier);
  has_verifier_ = true;
}
bool DefinitionBuilder::finish(std::string *error) {
  if (consumed_ || !registry_) {
    if (error)
      *error = "definition builder is consumed";
    return false;
  }
  consumed_ = true;
  if (!error)
    return false;
  if (error_.empty()) {
    if (!has_attributes_)
      fail("attributes must be declared");
    else if (!has_verifier_)
      fail("verifier must be declared");
    else if (staged_.attribute_index_ != staged_.verifier_.index())
      fail("verifier attribute type does not match declared attributes");
    else if (registry_->find(staged_.opcode_))
      fail("operation code is already registered");
  }
  if (error_.empty()) {
    for (const auto &port : staged_.ports_) {
      if (port.domain() == AccessDomain::Whole)
        continue;
      const auto *begin = staged_.port(Port::RowBegin);
      const auto *count = staged_.port(Port::RowCount);
      if (!begin || !count || begin->kind() != ValueKind::Scalar ||
          count->kind() != ValueKind::Scalar ||
          begin->access() != Access::Read || count->access() != Access::Read ||
          begin->required() != count->required()) {
        fail("row access domain requires paired scalar read controls");
        break;
      }
      if (port.domain() == AccessDomain::ActiveTokenPositions ||
          port.domain() == AccessDomain::CausalPrefix) {
        const bool publish =
            port.domain() == AccessDomain::ActiveTokenPositions;
        const auto *positions = staged_.port(Port::Positions);
        if ((publish ? (staged_.opcode_ != OpCode::RotatePublishKV ||
                        !staged_.expectsAttributes<RotatePublishKVAttrs>() ||
                        port.access() != Access::ReadWrite)
                     : (staged_.opcode_ != OpCode::CausalAttention ||
                        !staged_.expectsAttributes<CausalAttentionAttrs>() ||
                        port.access() != Access::Read || !begin->required())) ||
            (port.port() != Port::KeyCache &&
             port.port() != Port::ValueCache) ||
            !positions || !positions->required() ||
            positions->kind() != ValueKind::Tensor ||
            positions->access() != Access::Read ||
            positions->domain() != AccessDomain::ActiveRows) {
          fail("cache domain requires matching operation, cache port and "
               "positions");
          break;
        }
      } else if (port.domain() != AccessDomain::ActiveRows &&
                 (staged_.opcode_ != OpCode::CopyRows ||
                  !staged_.expectsAttributes<CopyRowsAttrs>() ||
                  !begin->required() ||
                  (port.domain() == AccessDomain::CopyRowsSource &&
                   (port.port() != Port::Input ||
                    port.access() != Access::Read)) ||
                  (port.domain() == AccessDomain::CopyRowsDestination &&
                   (port.port() != Port::Output ||
                    port.access() != Access::Write)))) {
        fail("copy_rows domain requires matching operation, attributes and "
             "port");
        break;
      }
    }
  }
  if (!error_.empty()) {
    *error = error_;
    return false;
  }
  registry_->definitions_.push_back(std::move(staged_));
  error->clear();
  return true;
}

PortBinding::PortBinding(Port port, ValueId value)
    : port_(port), value_(std::move(value)) {}
const PortBinding *Invocation::binding(Port port) const {
  const auto found =
      std::find_if(bindings_.begin(), bindings_.end(),
                   [&](const auto &b) { return b.port() == port; });
  return found == bindings_.end() ? nullptr : &*found;
}
BoundOperands::BoundOperands(const OperationGraph &graph,
                             const Invocation &invocation)
    : graph_(graph), invocation_(invocation) {}
const ValueDescription *BoundOperands::value(Port port) const {
  const auto *binding = invocation_.binding(port);
  return binding ? graph_.value(binding->value()) : nullptr;
}
const TensorType *BoundOperands::tensor(Port port) const {
  const auto *description = value(port);
  return description ? description->tensorType() : nullptr;
}
bool BoundOperands::sameValue(Port a, Port b) const {
  const auto *first = invocation_.binding(a);
  const auto *second = invocation_.binding(b);
  return first && second && first->value() == second->value();
}
InvocationBuilder::InvocationBuilder(OperationGraph &graph, OpCode opcode)
    : graph_(&graph) {
  staged_.opcode_ = opcode;
  if (!InRange(opcode, OpCode::Embedding, OpCode::PublishToken))
    fail("invalid operation code");
  else if (!graph.definition(opcode))
    fail("operation code is not registered in graph snapshot");
}
InvocationBuilder::InvocationBuilder(InvocationBuilder &&other) noexcept {
  *this = std::move(other);
}
InvocationBuilder &
InvocationBuilder::operator=(InvocationBuilder &&other) noexcept {
  if (this != &other) {
    graph_ = std::exchange(other.graph_, nullptr);
    staged_ = std::move(other.staged_);
    consumed_ = std::exchange(other.consumed_, true);
    has_attributes_ = other.has_attributes_;
    error_ = std::move(other.error_);
  }
  return *this;
}
void InvocationBuilder::fail(const std::string &error) {
  if (error_.empty())
    error_ = error;
}
bool InvocationBuilder::ready() {
  if (consumed_ || !graph_)
    fail("invocation builder is consumed");
  return error_.empty();
}
InvocationBuilder &InvocationBuilder::bind(Port port, ValueId value) {
  if (!ready())
    return *this;
  const auto *definition = graph_->definition(staged_.opcode_)->port(port);
  const auto *description = graph_->value(value);
  if (!InRange(port, Port::Input, Port::Progress))
    fail("invalid port");
  else if (!definition)
    fail("port is not declared by operation");
  else if (staged_.binding(port))
    fail("port is already bound");
  else if (!description)
    fail("invalid or foreign value handle");
  else if (description->kind() != definition->kind())
    fail("value kind does not match operation port");
  else
    staged_.bindings_.push_back(PortBinding(port, std::move(value)));
  return *this;
}
void InvocationBuilder::setAttributes(OperationAttrs attrs) {
  if (!ready())
    return;
  if (has_attributes_)
    fail("attributes are already bound");
  else if (attrs.index() !=
           graph_->definition(staged_.opcode_)->attribute_index_)
    fail("attributes do not match operation definition");
  else {
    staged_.attributes_ = std::move(attrs);
    has_attributes_ = true;
  }
}
bool InvocationBuilder::append(std::string *error) {
  if (consumed_ || !graph_) {
    if (error)
      *error = "invocation builder is consumed";
    return false;
  }
  consumed_ = true;
  if (!error)
    return false;
  if (!error_.empty()) {
    *error = error_;
    return false;
  }
  if (!has_attributes_) {
    *error = "attributes must be bound";
    return false;
  }
  const auto *definition = graph_->definition(staged_.opcode_);
  std::vector<PortBinding> ordered;
  ordered.reserve(definition->ports().size());
  for (const auto &port : definition->ports()) {
    const auto *binding = staged_.binding(port.port());
    if (!binding && !port.required())
      continue;
    if (!binding) {
      *error = std::string("missing operation port: ") + PortName(port.port());
      return false;
    }
    ordered.push_back(*binding);
  }
  staged_.bindings_ = std::move(ordered);
  const bool has_row_domain = std::any_of(
      definition->ports().begin(), definition->ports().end(),
      [](const auto &port) { return port.domain() != AccessDomain::Whole; });
  if (has_row_domain) {
    const auto *begin = staged_.binding(Port::RowBegin);
    const auto *count = staged_.binding(Port::RowCount);
    if (bool(begin) != bool(count)) {
      *error = "row controls must be both present or absent";
      return false;
    }
    if (begin) {
      const auto *b = graph_->value(begin->value())->scalarType();
      const auto *c = graph_->value(count->value())->scalarType();
      if (!b || !c || *b != ScalarType::I64 || *c != ScalarType::I64) {
        *error = "row controls must be I64 scalars";
        return false;
      }
    }
  }
  BoundOperands operands(*graph_, staged_);
  std::string diagnostic;
  const bool accepted = std::visit(
      [&](const auto &attrs) {
        using T = std::decay_t<decltype(attrs)>;
        return std::get<detail::Verifier<T>>(definition->verifier_)(
            operands, attrs, &diagnostic);
      },
      staged_.attributes_);
  if (!accepted) {
    *error = diagnostic.empty() ? "operation verifier rejected invocation"
                                : diagnostic;
    return false;
  }
  graph_->invocations_.push_back(std::move(staged_));
  error->clear();
  return true;
}

OperationGraph::OperationGraph(const OperationRegistry &registry)
    : identity_(std::make_shared<detail::GraphIdentity>()),
      definitions_(registry.definitions()) {}
ValueId
OperationGraph::addValue(std::variant<TensorType, ScalarType, StateKind> type,
                         std::string *error) {
  ValueId id(identity_, values_.size());
  values_.push_back(ValueDescription(id, std::move(type)));
  error->clear();
  return id;
}
ValueId OperationGraph::tensor(const TensorType &type, std::string *error) {
  if (!error || !ValidTensor(type, error))
    return ValueId();
  return addValue(type, error);
}
ValueId OperationGraph::scalar(ScalarType type, std::string *error) {
  if (!error)
    return ValueId();
  if (!InRange(type, ScalarType::F32, ScalarType::I64)) {
    *error = "invalid scalar type";
    return ValueId();
  }
  return addValue(type, error);
}
ValueId OperationGraph::state(StateKind kind, std::string *error) {
  if (!error)
    return ValueId();
  if (kind != StateKind::KVCache) {
    *error = "invalid state kind";
    return ValueId();
  }
  return addValue(kind, error);
}
const ValueDescription *OperationGraph::value(ValueId id) const {
  if (id.identity_ != identity_ || id.slot_ >= values_.size())
    return nullptr;
  return &values_[id.slot_];
}
const OperationDefinition *OperationGraph::definition(OpCode opcode) const {
  return Find(definitions_, opcode);
}
InvocationBuilder OperationGraph::operation(OpCode opcode) {
  return InvocationBuilder(*this, opcode);
}

const char *OpCodeName(OpCode opcode) {
  switch (opcode) {
  case OpCode::Embedding:
    return "embedding";
  case OpCode::RmsNorm:
    return "rms_norm";
  case OpCode::Quantize:
    return "quantize";
  case OpCode::Linear:
    return "linear";
  case OpCode::GatedMatmul:
    return "gated_matmul";
  case OpCode::LinearResidual:
    return "linear_residual";
  case OpCode::Rope:
    return "rope";
  case OpCode::RotatePublishKV:
    return "rotate_publish_kv";
  case OpCode::CausalAttention:
    return "causal_attention";
  case OpCode::CopyRows:
    return "copy_rows";
  case OpCode::ArgMax:
    return "arg_max";
  case OpCode::OutputHead:
    return "output_head";
  case OpCode::PublishToken:
    return "publish_token";
  }
  return "invalid_opcode";
}
const char *PortName(Port port) {
  switch (port) {
  case Port::Input:
    return "input";
  case Port::Weight:
    return "weight";
  case Port::GateWeight:
    return "gate_weight";
  case Port::UpWeight:
    return "up_weight";
  case Port::Residual:
    return "residual";
  case Port::Output:
    return "output";
  case Port::TokenIds:
    return "token_ids";
  case Port::Gamma:
    return "gamma";
  case Port::RowBegin:
    return "row_begin";
  case Port::RowCount:
    return "row_count";
  case Port::Positions:
    return "positions";
  case Port::Factors:
    return "factors";
  case Port::Query:
    return "query";
  case Port::Key:
    return "key";
  case Port::Value:
    return "value";
  case Port::KeyCache:
    return "key_cache";
  case Port::ValueCache:
    return "value_cache";
  case Port::Token:
    return "token";
  case Port::Control:
    return "control";
  case Port::History:
    return "history";
  case Port::Progress:
    return "progress";
  }
  return "invalid_port";
}
const char *AccessName(Access access) {
  switch (access) {
  case Access::Read:
    return "read";
  case Access::Write:
    return "write";
  case Access::ReadWrite:
    return "read_write";
  }
  return "invalid_access";
}
} // namespace hrx::operations
