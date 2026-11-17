// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/output_emitter.h"
#include "lib/loomify/text/jinja/format_analysis.h"

#include <cstdint>
#include <utility>

namespace hrx::loomify_gguf::jinja {
namespace {
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}
void Literal(std::string_view name, std::string_view value, std::string *out) {
  *out += "global.rodata.def @" + std::string(name) + " = bytes(\"" +
          Hex(value) + "\")\n";
}
bool Fail(std::string_view name, std::string message, size_t offset,
          Diagnostic *diagnostic) {
  if (diagnostic) {
    *diagnostic = Diagnostic{};
    diagnostic->source_name = std::string(name);
    diagnostic->offset = offset;
    diagnostic->message = std::move(message);
  }
  return false;
}

bool Space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
std::string_view TrimLeft(std::string_view value) {
  while (!value.empty() && Space(value.front()))
    value.remove_prefix(1);
  return value;
}
std::string_view TrimRight(std::string_view value) {
  while (!value.empty() && Space(value.back()))
    value.remove_suffix(1);
  return value;
}
std::string_view Trim(std::string_view value) {
  return TrimRight(TrimLeft(value));
}

// Split only source whitespace outside literal markup. Whitespace inside a
// marker remains part of that exact marker; no tag/name vocabulary is guessed.
std::vector<std::string> MarkerPieces(std::string_view literal) {
  std::vector<std::string> result;
  bool tag = false;
  std::string current;
  for (char c : literal) {
    if (Space(c) && !tag) {
      if (!current.empty())
        result.push_back(std::move(current));
      current.clear();
    } else {
      current += c;
      if (c == '<')
        tag = true;
      if (c == '>')
        tag = false;
    }
  }
  if (!current.empty())
    result.push_back(std::move(current));
  return result;
}

// Compare the finalized operations actually consumed below, not the original
// expression trees. Whitespace skipped outside markers is not a wire dialect;
// payload framing and codecs remain exact.
const char *ParserDifference(const TextToolFormat &a, const TextToolFormat &b) {
  if (a.kind != b.kind)
    return "tool grammars";
  if (a.kind == TextToolKind::None)
    return nullptr;
  if (a.kind == TextToolKind::JsonEnvelope) {
    if (a.open != b.open || a.close != b.close)
      return "JSON envelope markers";
    if (a.name_field != b.name_field || a.arguments_field != b.arguments_field)
      return "JSON member names";
    if (a.json_argument_wire != b.json_argument_wire)
      return "JSON argument encodings";
    return nullptr;
  }
  const auto &x = a.tagged;
  const auto &y = b.tagged;
  if (MarkerPieces(x.before_name) != MarkerPieces(y.before_name) ||
      Trim(x.after_name) != Trim(y.after_name) ||
      MarkerPieces(x.after_arguments) != MarkerPieces(y.after_arguments) ||
      Trim(x.after_arguments) != Trim(y.after_arguments) ||
      MarkerPieces(x.before_key) != MarkerPieces(y.before_key) ||
      Trim(x.before_key) != Trim(y.before_key) ||
      Trim(x.after_key) != Trim(y.after_key) ||
      Trim(x.after_value) != Trim(y.after_value))
    return "tagged boundaries";
  if (std::string_view(x.after_key).substr(TrimRight(x.after_key).size()) !=
          std::string_view(y.after_key).substr(TrimRight(y.after_key).size()) ||
      std::string_view(x.after_value)
              .substr(0,
                      x.after_value.size() - TrimLeft(x.after_value).size()) !=
          std::string_view(y.after_value)
              .substr(0, y.after_value.size() - TrimLeft(y.after_value).size()))
    return "tagged payload framing";
  if (x.value_codec.wire != y.value_codec.wire ||
      x.value_codec.trim != y.value_codec.trim)
    return "tagged value encodings";
  return nullptr;
}

class TaggedEmitter {
public:
  TaggedEmitter(size_t id, const TextTaggedFormat &shape, bool key_value)
      : prefix_("generated_output.tagged." + std::to_string(id)), shape_(shape),
        key_value_(key_value) {}
  std::string Emit() {
    body_ = "func.def @" + prefix_ +
            "(%schemas: buffer, %response: buffer) -> (buffer) {\n";
    Line("%zero = index.constant 0 : offset");
    Line("%size = buffer.length %response");
    const auto empty = Buffer("");
    Line("%end, %xml = scf.while(%p = %zero : offset, %prior = " + empty +
         " : buffer) -> (offset, buffer) {");
    const auto at = Skip("%p");
    const auto more = Fresh();
    Line(more + " = index.cmp ult, " + at + ", %size : offset");
    Line("scf.condition " + more + ", " + at + ", %prior : i1, offset, buffer");
    Line("} do(%p: offset, %prior: buffer) {");
    auto cursor = Expect(shape_.before_name, "%p");
    std::string name;
    if (!Trim(shape_.after_name).empty()) {
      const auto captured = Capture(cursor, Trim(shape_.after_name));
      name = captured.first;
      cursor = captured.second;
    } else {
      const auto key_marker = Buffer(Trim(shape_.before_key));
      const auto end_marker = Buffer(Trim(shape_.after_arguments));
      const auto key_at = Fresh(), has_key = Fresh(), end_at = Fresh(),
                 has_end = Fresh();
      Line(key_at + ", " + has_key + " = func.call @text.find(%response, " +
           cursor + ", " + key_marker +
           ") : (buffer, offset, buffer) -> (offset, i1)");
      Line(end_at + ", " + has_end + " = func.call @text.find(%response, " +
           cursor + ", " + end_marker +
           ") : (buffer, offset, buffer) -> (offset, i1)");
      Line("func.call @text.output_check(" + has_end + ") : (i1)");
      const auto earlier = Fresh(), args = Fresh(), stop = Fresh();
      Line(earlier + " = index.cmp ult, " + key_at + ", " + end_at +
           " : offset");
      Line(args + " = scalar.andi " + has_key + ", " + earlier + " : i1");
      Line(stop + " = scf.select " + args + ", " + key_at + ", " + end_at +
           " : offset");
      name = Fresh();
      Line(name + " = func.call @text.slice(%response, " + cursor + ", " +
           stop + ") : (buffer, offset, offset) -> (buffer)");
      cursor = stop;
    }
    const auto initial = Fresh();
    Line(initial + " = func.call @output_tagged.begin(" + name +
         ") : (buffer) -> (buffer)");
    const auto args_end = Fresh(), args_xml = Fresh();
    Line(args_end + ", " + args_xml + " = scf.while(%arg_p = " + cursor +
         " : offset, %arg_xml = " + initial +
         " : buffer) -> (offset, buffer) {");
    const auto arg_at = Skip("%arg_p");
    const auto key_marker = Buffer(Trim(shape_.before_key));
    const auto has_arg = Fresh();
    Line(has_arg + " = func.call @text.at(%response, " + arg_at + ", " +
         key_marker + ") : (buffer, offset, buffer) -> (i1)");
    Line("scf.condition " + has_arg + ", " + arg_at +
         ", %arg_xml : i1, offset, buffer");
    Line("} do(%arg_p: offset, %arg_xml: buffer) {");
    const auto key_begin = Expect(shape_.before_key, "%arg_p");
    const auto key = Capture(key_begin, Trim(shape_.after_key));
    const auto raw = Capture(key.second, Trim(shape_.after_value));
    const auto after_key = std::string_view(shape_.after_key);
    const auto after_value = std::string_view(shape_.after_value);
    const auto opening = Buffer(after_key.substr(TrimRight(after_key).size()));
    const auto closing = Buffer(after_value.substr(
        0, after_value.size() - TrimLeft(after_value).size()));
    auto value = Fresh();
    Line(value + " = func.call @output_tagged.unframe(" + raw.first + ", " +
         opening + ", " + closing + ") : (buffer, buffer, buffer) -> (buffer)");
    if (shape_.value_codec.wire == TextValueWire::Json) {
      const auto decoded = Fresh();
      Line(decoded + " = func.call @output_tagged.json_value(" + value +
           ") : (buffer) -> (buffer)");
      value = decoded;
    }
    if (shape_.value_codec.trim) {
      const auto trimmed = Fresh();
      Line(trimmed + " = func.call @text.trim(" + value +
           ") : (buffer) -> (buffer)");
      value = trimmed;
    }
    if (shape_.value_codec.wire == TextValueWire::RawScalar ||
        shape_.value_codec.wire == TextValueWire::JsonContainersElseScalar) {
      const auto normalized = Fresh();
      Line(normalized + " = func.call @output_tagged.python_scalar(%schemas, " +
           name + ", " + key.first + ", " + value +
           ") : (buffer, buffer, buffer, buffer) -> (buffer)");
      value = normalized;
    }
    const auto appended = Fresh();
    Line(appended + " = func.call @output_tagged.parameter(%arg_xml, " +
         key.first + ", " + value + ") : (buffer, buffer, buffer) -> (buffer)");
    Line("scf.yield " + raw.second + ", " + appended + " : offset, buffer");
    Line("}");
    const auto end = Expect(shape_.after_arguments, args_end);
    const auto completed = Fresh(), joined = Fresh();
    Line(completed + " = func.call @output_tagged.end(" + args_xml +
         ") : (buffer) -> (buffer)");
    Line(
        joined + " = func.call @text.join8(%prior, " + completed + ", " +
        empty + ", " + empty + ", " + empty + ", " + empty + ", " + empty +
        ", " + empty +
        ") : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)");
    Line("scf.yield " + end + ", " + joined + " : offset, buffer");
    Line("}");
    const auto checks = Fresh(), calls = Fresh();
    Line(checks + " = scalar.constant " +
         std::string(key_value_ ? "true" : "false") + " : i1");
    Line(calls + " = func.call @output_tagged.finish(%schemas, %xml, " +
         checks + ") : (buffer, buffer, i1) -> (buffer)");
    Line("func.return " + calls + " : buffer");
    body_ += "}\n";
    return constants_ + body_;
  }

private:
  std::string Fresh() { return "%t" + std::to_string(next_++); }
  void Line(const std::string &line) { body_ += "  " + line + "\n"; }
  std::string Buffer(std::string_view value) {
    const auto name = prefix_ + ".constant." + std::to_string(next_++);
    Literal(name, value, &constants_);
    const auto result = Fresh();
    Line(result + " = global.load @" + name + " : buffer");
    return result;
  }
  std::string Skip(const std::string &at) {
    const auto result = Fresh();
    Line(result + " = func.call @output_schema.skip(%response, " + at +
         ") : (buffer, offset) -> (offset)");
    return result;
  }
  std::string Expect(std::string_view literal, std::string at) {
    for (const auto &piece : MarkerPieces(literal)) {
      at = Skip(at);
      const auto marker = Buffer(piece), next = Fresh();
      Line(next + " = func.call @output_tagged.expect(%response, " + at +
           ", %size, " + marker +
           ") : (buffer, offset, offset, buffer) -> (offset)");
      at = next;
    }
    return at;
  }
  std::pair<std::string, std::string> Capture(const std::string &at,
                                              std::string_view delimiter) {
    const auto marker = Buffer(delimiter), value = Fresh(), end = Fresh();
    Line(value + ", " + end +
         " = func.call @output_tagged.capture(%response, " + at + ", %size, " +
         marker + ") : (buffer, offset, offset, buffer) -> (buffer, offset)");
    return {value, end};
  }
  std::string prefix_, constants_, body_;
  const TextTaggedFormat &shape_;
  bool key_value_;
  size_t next_ = 0;
};
} // namespace

bool EmitOutputParser(std::string_view source_name,
                      const TextFormatAnalysis &analysis,
                      GeneratedInputRenderer *out, Diagnostic *diagnostic) {
  if (!out || analysis.alternatives.empty())
    return Fail(source_name,
                "output emission needs accepted formats and output", 0,
                diagnostic);
  if (analysis.alternatives.size() > (uint64_t{1} << 32))
    return Fail(source_name, "output format identity exceeds its state field",
                0, diagnostic);
  bool has_tagged = false;
  for (const auto &format : analysis.alternatives) {
    const auto &wire = format.tools.json_argument_wire;
    const size_t offset =
        format.evidence.empty() ? 0 : format.evidence[0].begin;
    switch (format.tools.kind) {
    case TextToolKind::None:
    case TextToolKind::JsonEnvelope:
    case TextToolKind::TaggedNamed:
    case TextToolKind::TaggedKeyValue:
      break;
    default:
      return Fail(source_name, "invalid finalized tool grammar", offset,
                  diagnostic);
    }
    if ((format.tools.kind == TextToolKind::JsonEnvelope) != wire.has_value())
      return Fail(source_name,
                  "output emission needs finalized JSON wire facts", offset,
                  diagnostic);
    if (wire) {
      switch (*wire) {
      case TextJsonArgumentWire::RawJson:
      case TextJsonArgumentWire::Json:
        break;
      default:
        return Fail(source_name, "invalid finalized JSON argument wire", offset,
                    diagnostic);
      }
    }
    has_tagged |= format.tools.kind == TextToolKind::TaggedNamed ||
                  format.tools.kind == TextToolKind::TaggedKeyValue;
  }
  const auto &tools = analysis.alternatives.front().tools;
  for (size_t i = 1; i < analysis.alternatives.size(); ++i) {
    const auto &format = analysis.alternatives[i];
    if (const char *difference = ParserDifference(tools, format.tools))
      return Fail(
          source_name,
          "fixed a401 parse_tools has no request or state to select different " +
              std::string(difference) + " (source alternatives 0 and " +
              std::to_string(i) + ")",
          format.evidence.empty() ? 0 : format.evidence.front().begin,
          diagnostic);
  }
  GeneratedInputRenderer generated;
  generated.entrypoint = "output_next";
  generated.helper_requirements = {
      "lib/text/common/strings.loom", "lib/text/common/json.loom",
      "lib/text/output/stream.loom", "lib/text/output/json_calls.loom",
      "lib/text/output/schema.loom"};
  if (has_tagged) {
    generated.helper_requirements.push_back(
        "lib/text/output/tagged_calls.loom");
  }
  auto &source = generated.source;
  source = R"(
func.decl @text.output_check(%condition: i1)
func.decl @text.output_scan(%response: buffer, %previous: i64, %state: i64, %phase: i32, %open: buffer, %close: buffer, %tool: buffer, %after_open: buffer, %after_close: buffer) -> (i64, i64, i64, i64)
func.decl @output_json.parse_json_calls(%schemas: buffer, %response: buffer, %open: buffer, %close: buffer, %name: buffer, %arguments: buffer, %json_arguments: i1) -> (buffer)
func.decl @output_json.has_requested_tools(%environment: buffer) -> (i1)
)";
  if (has_tagged)
    source += R"(
func.decl @text.at(%text: buffer, %at: offset, %marker: buffer) -> (i1)
func.decl @text.find(%text: buffer, %start: offset, %marker: buffer) -> (offset, i1)
func.decl @text.slice(%text: buffer, %begin: offset, %end: offset) -> (buffer)
func.decl @text.trim(%text: buffer) -> (buffer)
func.decl @output_schema.skip(%text: buffer, %start: offset) -> (offset)
func.decl @output_tagged.expect(%text: buffer, %at: offset, %limit: offset, %marker: buffer) -> (offset)
func.decl @output_tagged.capture(%text: buffer, %at: offset, %limit: offset, %marker: buffer) -> (buffer, offset)
func.decl @output_tagged.unframe(%value: buffer, %prefix: buffer, %suffix: buffer) -> (buffer)
func.decl @output_tagged.begin(%name: buffer) -> (buffer)
func.decl @output_tagged.parameter(%prior: buffer, %key: buffer, %value: buffer) -> (buffer)
func.decl @output_tagged.python_scalar(%schemas: buffer, %function: buffer, %key: buffer, %value: buffer) -> (buffer)
func.decl @output_tagged.json_value(%value: buffer) -> (buffer)
func.decl @output_tagged.end(%prior: buffer) -> (buffer)
func.decl @output_tagged.finish(%schemas: buffer, %xml: buffer, %check_calls: i1) -> (buffer)
)";
  for (size_t i = 0; i < analysis.alternatives.size(); ++i) {
    const auto &f = analysis.alternatives[i];
    const auto prefix = "generated_output." + std::to_string(i) + ".";
    Literal(prefix + "open", f.reasoning_open, &source);
    Literal(prefix + "close", f.reasoning_close, &source);
    Literal(prefix + "after_open", f.reasoning_after_open, &source);
    Literal(prefix + "after_close", f.reasoning_after_close, &source);
    Literal(prefix + "tool", f.tools.open, &source);
  }
  if (tools.kind == TextToolKind::JsonEnvelope) {
    Literal("generated_output.parser.open", tools.open, &source);
    Literal("generated_output.parser.close", tools.close, &source);
    Literal("generated_output.parser.name", tools.name_field, &source);
    Literal("generated_output.parser.arguments", tools.arguments_field,
            &source);
  } else if (has_tagged) {
    source += TaggedEmitter(0, tools.tagged,
                            tools.kind == TextToolKind::TaggedKeyValue)
                  .Emit();
  }
  source += R"(
func.def public @text.output_check_environment(%environment: buffer, %format: i64) {
  %false = scalar.constant false : i1
  %requested = func.call @output_json.has_requested_tools(%environment) : (buffer) -> (i1)
)";
  std::string ambiguous = "%false";
  for (size_t i = 0; i < analysis.alternatives.size(); ++i) {
    if (!analysis.alternatives[i].ambiguous_bare_json_tools)
      continue;
    const auto n = std::to_string(i);
    source += "  %id" + n + " = scalar.constant " + n + " : i64\n";
    source +=
        "  %same" + n + " = scalar.cmpi eq, %format, %id" + n + " : i64\n";
    source += "  %ambiguous" + n + " = scalar.ori " + ambiguous + ", %same" +
              n + " : i1\n";
    ambiguous = "%ambiguous" + n;
  }
  source += "  %unsupported = scalar.andi %requested, " + ambiguous + " : i1\n";
  source += R"(
  %allowed = scalar.cmpi eq, %unsupported, %false : i1
  func.call @text.output_check(%allowed) : (i1)
  func.return
}
func.def public @text.output_initial(%format: i64, %startup: i64) -> (i64) {
  %zero = scalar.constant 0 : i64
  %one = scalar.constant 1 : i64
  %two = scalar.constant 2 : i64
  %three = scalar.constant 3 : i64
  %shift = scalar.constant 32 : i64
  %lower = scalar.cmpi sge, %format, %zero : i64
  %limit = scalar.constant )" +
            std::to_string(analysis.alternatives.size()) + R"( : i64
  %upper = scalar.cmpi ult, %format, %limit : i64
  %valid_id = scalar.andi %lower, %upper : i1
  func.call @text.output_check(%valid_id) : (i1)
  %mode_lower = scalar.cmpi sge, %startup, %zero : i64
  %mode_upper = scalar.cmpi ule, %startup, %two : i64
  %valid_mode = scalar.andi %mode_lower, %mode_upper : i1
  func.call @text.output_check(%valid_mode) : (i1)
)";
  // The producer records the exact semantic mode at the same terminal that
  // emits the selector result. The initializer owns all private packing.
  source += "  %false = scalar.constant false : i1\n";
  std::string allowed = "%false";
  for (size_t i = 0; i < analysis.alternatives.size(); ++i) {
    const auto n = std::to_string(i);
    source += "  %id" + n + " = scalar.constant " + n + " : i64\n";
    source += "  %mode" + n + " = scalar.constant " +
              std::to_string(
                  static_cast<int>(analysis.alternatives[i].startup_mode)) +
              " : i64\n";
    source +=
        "  %same_id" + n + " = scalar.cmpi eq, %format, %id" + n + " : i64\n";
    source += "  %same_mode" + n + " = scalar.cmpi eq, %startup, %mode" + n +
              " : i64\n";
    source += "  %pair" + n + " = scalar.andi %same_id" + n + ", %same_mode" +
              n + " : i1\n";
    source += "  %allowed" + n + " = scalar.ori " + allowed + ", %pair" + n +
              " : i1\n";
    allowed = "%allowed" + n;
  }
  source += "  func.call @text.output_check(" + allowed + ") : (i1)\n";
  source += R"(
  %content = scalar.cmpi eq, %startup, %zero : i64
  %optional = scalar.cmpi eq, %startup, %two : i64
  %reason = scf.select %optional, %three, %two : i64
  %mode = scf.select %content, %one, %reason : i64
  %identity = scalar.shli %format, %shift : i64
  %state = scalar.ori %identity, %mode : i64
  func.return %state : i64
}
func.def public @output_next(%response: buffer, %previous: i64, %state: i64, %phase: i32) -> (i64, i64, i64, i64) {
  %shift = scalar.constant 32 : i64
  %format = scalar.shrui %state, %shift : i64
)";
  // Small finite dispatch. Every accepted id is tested, then asserted at the
  // leaf; no unknown format is silently interpreted as the last alternative.
  for (size_t i = 0; i < analysis.alternatives.size(); ++i) {
    const auto n = std::to_string(i);
    source += "  %id" + n + " = scalar.constant " + n + " : i64\n";
    source += "  %is" + n + " = scalar.cmpi eq, %format, %id" + n + " : i64\n";
    source += "  %k" + n + ", %b" + n + ", %e" + n + ", %s" + n +
              " = scf.if %is" + n + " -> (i64, i64, i64, i64) {\n";
    for (const auto *key :
         {"open", "close", "tool", "after_open", "after_close"})
      source += "    %" + std::string(key) + n +
                " = global.load @generated_output." + n + "." + key +
                " : buffer\n";
    source +=
        "    %k, %b, %e, %s = func.call @text.output_scan(%response, %previous, %state, %phase, %open" +
        n + ", %close" + n + ", %tool" + n + ", %after_open" + n +
        ", %after_close" + n +
        ") : (buffer, i64, i64, i32, buffer, buffer, buffer, buffer, buffer) -> (i64, i64, i64, i64)\n";
    source += "    scf.yield %k, %b, %e, %s : i64, i64, i64, i64\n  } else {\n";
  }
  source +=
      "    %invalid = scalar.constant false : i1\n    func.call @text.output_check(%invalid) : (i1)\n    %zero = scalar.constant 0 : i64\n    scf.yield %zero, %previous, %previous, %state : i64, i64, i64, i64\n";
  for (size_t i = analysis.alternatives.size(); i-- > 0;) {
    const auto n = std::to_string(i);
    source += "  }\n";
    source += i ? "  scf.yield %k" + n + ", %b" + n + ", %e" + n + ", %s" + n +
                      " : i64, i64, i64, i64\n"
                : "  func.return %k0, %b0, %e0, %s0 : i64, i64, i64, i64\n}\n";
  }
  source += R"(
func.def public @parse_tools(%schemas: buffer, %response: buffer, %truncated: i64) -> (buffer) {
)";
  // Incomplete-tail and malformed-record behavior belongs to the shared
  // source-qualified dialect. Neither hook argument permits request routing.
  if (tools.kind == TextToolKind::None) {
    source += "  %empty = buffer.constant utf8(\"[]\") : buffer\n"
              "  func.return %empty : buffer\n";
  } else if (tools.kind == TextToolKind::JsonEnvelope) {
    for (const auto *key : {"open", "close", "name", "arguments"})
      source += "  %" + std::string(key) +
                " = global.load @generated_output.parser." + key +
                " : buffer\n";
    source += "  %json_arguments = scalar.constant " +
              std::string(tools.json_argument_wire == TextJsonArgumentWire::Json
                              ? "true"
                              : "false") +
              " : i1\n";
    source += R"(
  %calls = func.call @output_json.parse_json_calls(%schemas, %response, %open, %close, %name, %arguments, %json_arguments) : (buffer, buffer, buffer, buffer, buffer, buffer, i1) -> (buffer)
  func.return %calls : buffer
)";
  } else {
    source += R"(
  %calls = func.call @generated_output.tagged.0(%schemas, %response) : (buffer, buffer) -> (buffer)
  func.return %calls : buffer
)";
  }
  source += "}\n";
  *out = std::move(generated);
  if (diagnostic)
    *diagnostic = Diagnostic{};
  return true;
}
} // namespace hrx::loomify_gguf::jinja
