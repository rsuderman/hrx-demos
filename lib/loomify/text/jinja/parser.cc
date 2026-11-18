// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/parser.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <utility>

namespace hrx::loomify_gguf::jinja {
namespace {

bool Space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
bool NameStart(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool NamePart(unsigned char c) {
  return NameStart(c) || (c >= '0' && c <= '9');
}
bool Digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool UnicodeSpace(uint32_t c) {
  return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 ||
         c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
         c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f ||
         c == 0x3000;
}
// Source positions always refer to original UTF-8 bytes, before newline
// cleanup.
size_t CodePoint(std::string_view s, size_t at, uint32_t *cp) {
  if (at >= s.size())
    return 0;
  unsigned char c = s[at];
  if (c < 128) {
    *cp = c;
    return 1;
  }
  size_t n = c >= 0xc2 && c <= 0xdf   ? 2
             : c >= 0xe0 && c <= 0xef ? 3
             : c >= 0xf0 && c <= 0xf4 ? 4
                                      : 0;
  if (!n || at + n > s.size())
    return 0;
  uint32_t value = c & ((1u << (7 - n)) - 1);
  for (size_t i = 1; i < n; ++i) {
    unsigned char next = s[at + i];
    if ((next & 0xc0) != 0x80)
      return 0;
    value = (value << 6) | (next & 0x3f);
  }
  if ((n == 2 && value < 0x80) || (n == 3 && value < 0x800) ||
      (n == 4 && value < 0x10000) || (value >= 0xd800 && value <= 0xdfff) ||
      value > 0x10ffff)
    return 0;
  *cp = value;
  return n;
}
void AppendCodePoint(uint32_t cp, std::string *out) {
  if (cp < 0x80)
    out->push_back(static_cast<char>(cp));
  else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 63)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
    out->push_back(static_cast<char>(0x80 | (cp & 63)));
  } else {
    out->push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 63)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
    out->push_back(static_cast<char>(0x80 | (cp & 63)));
  }
}
std::string Newlines(std::string_view s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\r') {
      if (i + 1 < s.size() && s[i + 1] == '\n')
        ++i;
      c = '\n';
    }
    out.push_back(c);
  }
  return out;
}
void RStrip(std::string *s) {
  while (!s->empty()) {
    size_t begin = s->size() - 1;
    while (begin && (static_cast<unsigned char>((*s)[begin]) & 0xc0) == 0x80)
      --begin;
    uint32_t cp = 0;
    if (!CodePoint(*s, begin, &cp) || !UnicodeSpace(cp))
      break;
    s->resize(begin);
  }
}

struct Token {
  enum Kind { End, Name, Integer, String, Symbol } kind = End;
  std::string text;
  size_t begin = 0;
  size_t end = 0;
};

class ExpressionParser {
public:
  ExpressionParser(std::string_view source, size_t begin, size_t end,
                   Diagnostic *diagnostic)
      : source_(source), begin_(begin), end_(end), diagnostic_(diagnostic) {}

  bool Lex() {
    size_t i = begin_;
    while (i < end_) {
      unsigned char c = source_[i];
      if (Space(c)) {
        ++i;
        continue;
      }
      Token t;
      t.begin = i;
      if (NameStart(c)) {
        t.kind = Token::Name;
        while (i < end_ && NamePart(source_[i]))
          t.text += source_[i++];
      } else if (Digit(c)) {
        t.kind = Token::Integer;
        while (i < end_ && Digit(source_[i]))
          t.text += source_[i++];
        if (i < end_ && (source_[i] == '.' || NameStart(source_[i])))
          return Error(i, "only decimal integer literals are supported");
        if (t.text.size() > 1 && t.text[0] == '0') {
          if (t.text.find_first_not_of('0') != std::string::npos)
            return Error(t.begin, "leading zeros in decimal integer literal");
          t.text = "0";
        }
      } else if (c == '\'' || c == '"') {
        t.kind = Token::String;
        if (!String(&i, &t.text))
          return false;
        // Jinja joins adjacent string tokens as Python does.
        if (!tokens_.empty() && tokens_.back().kind == Token::String) {
          tokens_.back().text += t.text;
          tokens_.back().end = i;
          continue;
        }
      } else {
        t.kind = Token::Symbol;
        t.text += source_[i++];
        if (i < end_) {
          std::string two = t.text + source_[i];
          if (two == "==" || two == "!=" || two == "<=" || two == ">=" ||
              two == "//" || two == "**") {
            t.text = two;
            ++i;
          }
        }
        if (t.text.find_first_not_of("+-*/%~()[]{}.,:|=<>!") !=
            std::string::npos)
          return Error(t.begin, "unsupported expression character");
      }
      t.end = i;
      tokens_.push_back(std::move(t));
      if (tokens_.size() > 1024)
        return Error(i, "expression token count exceeds compiler limit");
    }
    tokens_.push_back({Token::End, "", end_, end_});
    return true;
  }

  bool Parse(Expr *out) {
    if (!Lex() || !Conditional(out))
      return false;
    if (Peek().kind != Token::End)
      return Error(Peek().begin, "unexpected expression token: " + Peek().text);
    return CheckDepth(*out);
  }

  // Statement parsers consume their prefix before parsing the remaining
  // expression.
  bool ParseSet(std::string *name, Expr *out, bool *capture) {
    if (!Lex())
      return false;
    if (Peek().kind != Token::Name)
      return Error(Peek().begin, "expected assignment name");
    if (ConstantTarget(Peek().text))
      return Error(Peek().begin, "cannot assign to a constant");
    *name = Take().text;
    if (Peek().kind == Token::End) {
      *capture = true;
      return true;
    }
    if (At("|"))
      return Error(Peek().begin,
                   "capture assignment filters are not supported");
    if (Accept(".")) {
      if (Peek().kind != Token::Name)
        return Error(Peek().begin, "expected namespace field name");
      *name += "." + Take().text;
      if (Peek().kind == Token::End || At("|"))
        return Error(Peek().begin,
                     "namespace-field captures are not supported");
    }
    if (!Expect("="))
      return false;
    return Finish(out);
  }
  bool ParseFor(std::string *name, Expr *out, std::optional<Expr> *predicate) {
    if (!Lex())
      return false;
    if (Peek().kind != Token::Name)
      return Error(Peek().begin, "expected loop variable");
    if (ConstantTarget(Peek().text))
      return Error(Peek().begin, "cannot assign to a constant");
    *name = Take().text;
    if (Accept(",")) {
      if (Peek().kind != Token::Name || ConstantTarget(Peek().text))
        return Error(Peek().begin, "expected second loop variable");
      *name += "," + Take().text;
    }
    if (!Expect("in"))
      return false;
    // The top-level if starts a loop filter, not a conditional iterable.
    // Parenthesized expressions still parse their own conditionals normally.
    if (!Binary(0, out) || !CheckDepth(*out))
      return false;
    if (Accept("if")) {
      Expr filter;
      if (!Conditional(&filter) || !CheckDepth(filter))
        return false;
      *predicate = std::move(filter);
    }
    return Peek().kind == Token::End ||
           Error(Peek().begin, "unexpected loop expression token");
  }
  bool ParseMacro(std::string *name, std::vector<std::string> *parameters,
                  std::vector<Expr> *defaults) {
    if (!Lex())
      return false;
    if (Peek().kind != Token::Name || ConstantTarget(Peek().text))
      return Error(Peek().begin, "expected macro name");
    *name = Take().text;
    if (!Expect("("))
      return false;
    while (!At(")")) {
      if (Peek().kind != Token::Name || ConstantTarget(Peek().text))
        return Error(Peek().begin, "expected macro parameter");
      const auto parameter = Take();
      if (std::find(parameters->begin(), parameters->end(), parameter.text) !=
          parameters->end())
        return Error(parameter.begin, "duplicate macro parameter");
      parameters->push_back(parameter.text);
      if (Accept("=")) {
        Expr value;
        if (!Conditional(&value) || !CheckDepth(value))
          return false;
        defaults->push_back(std::move(value));
      } else if (!defaults->empty())
        return Error(parameter.begin, "required parameter follows default");
      if (!Accept(","))
        break;
    }
    return Expect(")") &&
           (Peek().kind == Token::End ||
            Error(Peek().begin, "unexpected macro signature token"));
  }

private:
  static bool ConstantTarget(std::string_view text) {
    return text == "true" || text == "false" || text == "True" ||
           text == "False" || text == "none" || text == "None";
  }
  struct Depth {
    explicit Depth(ExpressionParser *p) : p(p), valid(++p->depth_ <= 128) {}
    ~Depth() { --p->depth_; }
    ExpressionParser *p;
    bool valid;
  };
  bool Finish(Expr *out) {
    if (!Conditional(out))
      return false;
    if (Peek().kind != Token::End)
      return Error(Peek().begin, "unexpected statement token: " + Peek().text);
    return CheckDepth(*out);
  }
  bool CheckDepth(const Expr &root) {
    // Left-associative operator/postfix chains need a structural bound as well
    // as the recursive-descent guard above, since parsing them is iterative.
    std::vector<std::pair<const Expr *, size_t>> pending{{&root, 1}};
    while (!pending.empty()) {
      const auto entry = pending.back();
      pending.pop_back();
      if (entry.second > 128)
        return Error(entry.first->offset,
                     "expression AST depth exceeds compiler limit");
      for (const Expr &arg : entry.first->args)
        pending.push_back({&arg, entry.second + 1});
    }
    return true;
  }
  bool Error(size_t offset, std::string message) {
    if (!failed_ && diagnostic_) {
      diagnostic_->offset = offset;
      diagnostic_->message = std::move(message);
    }
    failed_ = true;
    return false;
  }
  const Token &Peek(size_t n = 0) const {
    return tokens_[std::min(position_ + n, tokens_.size() - 1)];
  }
  Token Take() { return tokens_[position_++]; }
  bool At(std::string_view text) const {
    return Peek().text == text && Peek().kind != Token::String;
  }
  bool Accept(std::string_view text) {
    if (!At(text))
      return false;
    ++position_;
    return true;
  }
  bool Expect(std::string_view text) {
    return Accept(text) ||
           Error(Peek().begin, "expected '" + std::string(text) + "'");
  }
  Expr Make(Expr::Kind kind, std::string text, size_t begin, size_t end) {
    Expr e;
    e.kind = kind;
    e.text = std::move(text);
    e.offset = begin;
    e.end = end;
    return e;
  }
  void Child(Expr *parent, Expr child, std::string keyword = {}) {
    parent->args.push_back(std::move(child));
    parent->keywords.push_back(std::move(keyword));
  }
  bool String(size_t *at, std::string *out) {
    char quote = source_[(*at)++];
    while (*at < end_) {
      char c = source_[(*at)++];
      if (c == quote)
        return true;
      if (c != '\\') {
        if (c == '\r') {
          if (*at < end_ && source_[*at] == '\n')
            ++*at;
          c = '\n';
        }
        out->push_back(c);
        continue;
      }
      if (*at == end_)
        return Error(*at - 1, "unterminated string escape");
      char escaped = source_[(*at)++];
      switch (escaped) {
      case '\n':
        break;
      case '\r':
        if (*at < end_ && source_[*at] == '\n')
          ++*at;
        break;
      case 'n':
        out->push_back('\n');
        break;
      case 'r':
        out->push_back('\r');
        break;
      case 't':
        out->push_back('\t');
        break;
      case 'b':
        out->push_back('\b');
        break;
      case 'f':
        out->push_back('\f');
        break;
      case 'v':
        out->push_back('\v');
        break;
      case 'a':
        out->push_back('\a');
        break;
      case '\\':
      case '\'':
      case '"':
        out->push_back(escaped);
        break;
      case 'x':
      case 'u':
      case 'U': {
        size_t n = escaped == 'x' ? 2 : escaped == 'u' ? 4 : 8;
        uint32_t cp = 0;
        for (size_t j = 0; j < n; ++j) {
          if (*at == end_)
            return Error(*at, "incomplete Unicode/string escape");
          char digit = source_[(*at)++];
          unsigned v = digit >= '0' && digit <= '9'   ? digit - '0'
                       : digit >= 'a' && digit <= 'f' ? digit - 'a' + 10
                       : digit >= 'A' && digit <= 'F' ? digit - 'A' + 10
                                                      : 16;
          if (v == 16)
            return Error(*at - 1, "invalid hexadecimal escape");
          cp = (cp << 4) | v;
        }
        if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
          return Error(*at - n, "escape is not a Unicode scalar");
        AppendCodePoint(cp, out);
        break;
      }
      default:
        if (escaped >= '0' && escaped <= '7') {
          uint32_t cp = escaped - '0';
          for (size_t j = 1; j < 3 && *at < end_ && source_[*at] >= '0' &&
                             source_[*at] <= '7';
               ++j)
            cp = (cp << 3) | (source_[(*at)++] - '0');
          AppendCodePoint(cp, out);
        } else {
          out->push_back('\\');
          out->push_back(escaped);
        }
      }
    }
    return Error(*at, "unterminated string literal");
  }

  bool Conditional(Expr *out) {
    Depth depth(this);
    if (!depth.valid)
      return Error(Peek().begin, "expression nesting exceeds compiler limit");
    if (!Binary(0, out))
      return false;
    if (!Accept("if"))
      return true;
    Expr yes = std::move(*out), condition, no;
    if (!Binary(0, &condition))
      return false;
    if (Accept("else")) {
      if (!Conditional(&no))
        return false;
    } else {
      no = Make(Expr::Kind::Undefined, "", condition.end, condition.end);
    }
    *out = Make(Expr::Kind::Conditional, "", yes.offset, no.end);
    Child(out, std::move(condition));
    Child(out, std::move(yes));
    Child(out, std::move(no));
    return true;
  }
  static int Precedence(std::string_view op) {
    if (op == "or")
      return 1;
    if (op == "and")
      return 2;
    if (op == "==" || op == "!=" || op == "<" || op == ">" || op == "<=" ||
        op == ">=" || op == "in" || op == "not in")
      return 4;
    if (op == "+" || op == "-")
      return 5;
    if (op == "~")
      return 6;
    if (op == "*" || op == "/" || op == "//" || op == "%")
      return 7;
    if (op == "**")
      return 8;
    return -1;
  }
  bool Binary(int minimum, Expr *out) {
    Depth depth(this);
    if (!depth.valid)
      return Error(Peek().begin, "expression nesting exceeds compiler limit");
    if (At("not")) {
      Token op = Take();
      Expr child;
      if (!Binary(3, &child))
        return false;
      *out = Make(Expr::Kind::Unary, op.text, op.begin, child.end);
      Child(out, std::move(child));
    } else if (!UnaryValue(out, true))
      return false;
    bool comparison = false;
    while (true) {
      std::string op = Peek().text;
      if (Peek().kind == Token::String || Peek().kind == Token::End)
        break;
      if (op == "not" && Peek(1).text == "in")
        op = "not in";
      int precedence = Precedence(op);
      if (precedence < minimum)
        break;
      if (precedence == 4 && comparison)
        return Error(Peek().begin, "chained comparisons are not supported");
      comparison = precedence == 4;
      Take();
      if (op == "not in")
        Take();
      Expr left = std::move(*out), right;
      if (!Binary(precedence + 1, &right))
        return false;
      *out = Make(Expr::Kind::Binary, op, left.offset, right.end);
      Child(out, std::move(left));
      Child(out, std::move(right));
    }
    return true;
  }

  bool Arguments(Expr *out) {
    if (Accept(")")) {
      out->end = tokens_[position_ - 1].end;
      return true;
    }
    bool named = false;
    do {
      std::string keyword;
      if (Peek().kind == Token::Name && Peek(1).text == "=") {
        keyword = Take().text;
        Take();
        named = true;
      } else if (named)
        return Error(Peek().begin,
                     "positional argument follows keyword argument");
      if (!keyword.empty() &&
          std::find(out->keywords.begin(), out->keywords.end(), keyword) !=
              out->keywords.end())
        return Error(Peek().begin, "duplicate keyword argument: " + keyword);
      Expr value;
      if (!Conditional(&value))
        return false;
      Child(out, std::move(value), std::move(keyword));
    } while (Accept(",") && !At(")"));
    if (!Expect(")"))
      return false;
    out->end = tokens_[position_ - 1].end;
    return true;
  }
  bool UnaryValue(Expr *out, bool with_filters) {
    Depth depth(this);
    if (!depth.valid)
      return Error(Peek().begin, "expression nesting exceeds compiler limit");
    if (At("+") || At("-")) {
      Token op = Take();
      Expr child;
      if (!UnaryValue(&child, false))
        return false;
      *out = Make(Expr::Kind::Unary, op.text, op.begin, child.end);
      Child(out, std::move(child));
    } else if (!Primary(out))
      return false;
    return Postfix(out, with_filters);
  }
  bool Postfix(Expr *out, bool with_filters) {
    while (true) {
      if (Accept(".")) {
        if (Peek().kind != Token::Name)
          return Error(Peek().begin, "expected attribute name");
        Token attr = Take();
        Expr base = std::move(*out);
        *out = Make(Expr::Kind::GetAttr, attr.text, base.offset, attr.end);
        Child(out, std::move(base));
      } else if (Accept("[")) {
        Expr base = std::move(*out), start, end, step;
        start.offset = end.offset = step.offset = Peek().begin;
        if (!At(":")) {
          if (!Conditional(&start))
            return false;
        }
        if (Accept(":")) {
          if (!At("]") && !At(":")) {
            if (!Conditional(&end))
              return false;
          }
          if (Accept(":")) {
            if (!At("]")) {
              if (!Conditional(&step))
                return false;
            }
          }
          if (!Expect("]"))
            return false;
          *out = Make(Expr::Kind::Slice, "", base.offset,
                      tokens_[position_ - 1].end);
          Child(out, std::move(base));
          Child(out, std::move(start));
          Child(out, std::move(end));
          Child(out, std::move(step));
        } else {
          if (!Expect("]"))
            return false;
          *out = Make(Expr::Kind::GetItem, "", base.offset,
                      tokens_[position_ - 1].end);
          Child(out, std::move(base));
          Child(out, std::move(start));
        }
      } else if (Accept("(")) {
        Expr callee = std::move(*out);
        *out = Make(Expr::Kind::Call, "", callee.offset, callee.end);
        Child(out, std::move(callee));
        if (!Arguments(out))
          return false;
      } else if (with_filters && Accept("|")) {
        if (Peek().kind != Token::Name)
          return Error(Peek().begin, "expected filter name");
        Token filter = Take();
        Expr base = std::move(*out);
        *out = Make(Expr::Kind::Filter, filter.text, base.offset, filter.end);
        Child(out, std::move(base));
        if (Accept("(") && !Arguments(out))
          return false;
      } else if (with_filters && Accept("is")) {
        bool negate = Accept("not");
        if (Peek().kind != Token::Name)
          return Error(Peek().begin, "expected test name");
        Token test = Take();
        Expr base = std::move(*out);
        *out = Make(Expr::Kind::Test, test.text, base.offset, test.end);
        Child(out, std::move(base));
        if (Accept("(") && !Arguments(out))
          return false;
        if (negate) {
          Expr test_expr = std::move(*out);
          *out =
              Make(Expr::Kind::Unary, "not", test_expr.offset, test_expr.end);
          Child(out, std::move(test_expr));
        }
      } else
        break;
    }
    return true;
  }
  bool Primary(Expr *out) {
    Token t = Take();
    if (t.kind == Token::Name) {
      Expr::Kind kind = t.text == "true" || t.text == "false" ||
                                t.text == "True" || t.text == "False"
                            ? Expr::Kind::Boolean
                        : t.text == "none" || t.text == "None"
                            ? Expr::Kind::None
                            : Expr::Kind::Name;
      *out = Make(kind, t.text, t.begin, t.end);
      if (out->text == "True")
        out->text = "true";
      if (out->text == "False")
        out->text = "false";
      return true;
    }
    if (t.kind == Token::String || t.kind == Token::Integer) {
      *out = Make(t.kind == Token::String ? Expr::Kind::String
                                          : Expr::Kind::Integer,
                  t.text, t.begin, t.end);
      return true;
    }
    if (t.text == "(" || t.text == "[") {
      const bool tuple = t.text == "(";
      const char *close = tuple ? ")" : "]";
      *out = Make(tuple ? Expr::Kind::Tuple : Expr::Kind::List, "", t.begin,
                  t.end);
      if (Accept(close)) {
        out->end = tokens_[position_ - 1].end;
        return true;
      }
      Expr first;
      if (!Conditional(&first))
        return false;
      if (tuple && !At(",")) {
        if (!Expect(close))
          return false;
        *out = std::move(first);
        return true;
      }
      Child(out, std::move(first));
      while (Accept(",") && !At(close)) {
        Expr value;
        if (!Conditional(&value))
          return false;
        Child(out, std::move(value));
      }
      if (!Expect(close))
        return false;
      out->end = tokens_[position_ - 1].end;
      return true;
    }
    if (t.text == "{") {
      *out = Make(Expr::Kind::Dict, "", t.begin, t.end);
      if (Accept("}")) {
        out->end = tokens_[position_ - 1].end;
        return true;
      }
      do {
        Expr key, value;
        if (!Conditional(&key) || !Expect(":") || !Conditional(&value))
          return false;
        Child(out, std::move(key));
        Child(out, std::move(value));
      } while (Accept(",") && !At("}"));
      if (!Expect("}"))
        return false;
      out->end = tokens_[position_ - 1].end;
      return true;
    }
    return Error(t.begin, "expected expression");
  }

  std::string_view source_;
  size_t begin_, end_;
  Diagnostic *diagnostic_;
  std::vector<Token> tokens_;
  size_t position_ = 0;
  size_t depth_ = 0;
  bool failed_ = false;
};

struct Segment {
  enum Kind { Text, Output, Statement } kind;
  std::string text;
  size_t begin = 0, end = 0;
};
class TemplateParser {
public:
  TemplateParser(std::string_view source, Diagnostic *diagnostic)
      : source_(source), diagnostic_(diagnostic) {}
  bool Parse(Template *out) {
    for (size_t i = 0; i < source_.size();) {
      uint32_t cp;
      size_t n = CodePoint(source_, i, &cp);
      if (!n)
        return Error(i, "template source is not valid UTF-8");
      i += n;
    }
    if (!Scan())
      return false;
    Template parsed;
    if (!Nodes({}, &parsed.body, 0))
      return false;
    if (position_ != segments_.size())
      return Error(segments_[position_].begin, "unexpected closing block");
    *out = std::move(parsed);
    return true;
  }

private:
  bool Error(size_t offset, std::string message) {
    if (diagnostic_) {
      diagnostic_->offset = offset;
      diagnostic_->message = std::move(message);
    }
    return false;
  }
  void Literal(std::string_view source, size_t begin, size_t end) {
    if (source.empty())
      return;
    if (!segments_.empty() && segments_.back().kind == Segment::Text) {
      segments_.back().text += Newlines(source);
      segments_.back().end = end;
    } else
      segments_.push_back({Segment::Text, Newlines(source), begin, end});
  }
  bool Scan() {
    // Jinja removes one final source newline before lexing. Doing this after
    // scanning could accidentally strip an earlier literal newline when a
    // trailing comment/block has already consumed the actual final newline.
    if (!source_.empty() && source_.back() == '\n') {
      source_.remove_suffix(1);
      if (!source_.empty() && source_.back() == '\r')
        source_.remove_suffix(1);
    } else if (!source_.empty() && source_.back() == '\r') {
      source_.remove_suffix(1);
    }
    size_t cursor = 0;
    while (cursor < source_.size()) {
      size_t open = source_.find('{', cursor);
      while (open != std::string_view::npos &&
             (open + 1 == source_.size() ||
              (source_[open + 1] != '{' && source_[open + 1] != '%' &&
               source_[open + 1] != '#')))
        open = source_.find('{', open + 1);
      if (open == std::string_view::npos) {
        Literal(source_.substr(cursor), cursor, source_.size());
        break;
      }
      Literal(source_.substr(cursor, open - cursor), cursor, open);
      char kind = source_[open + 1];
      bool trim_left = open + 2 < source_.size() && source_[open + 2] == '-';
      if (open + 2 < source_.size() && source_[open + 2] == '+')
        return Error(open + 2,
                     "explicit plus whitespace control is not supported");
      if (!segments_.empty() && segments_.back().kind == Segment::Text) {
        auto &prior = segments_.back().text;
        if (trim_left)
          RStrip(&prior);
        else if (kind != '{') {
          size_t line = prior.find_last_of('\n');
          line = line == std::string::npos ? 0 : line + 1;
          size_t source_line = open;
          while (source_line && source_[source_line - 1] != '\n' &&
                 source_[source_line - 1] != '\r')
            --source_line;
          bool only_indent = true;
          for (size_t i = source_line; i < open;) {
            uint32_t cp;
            size_t n = CodePoint(source_, i, &cp);
            if (!n || !UnicodeSpace(cp)) {
              only_indent = false;
              break;
            }
            i += n;
          }
          if (only_indent)
            prior.resize(line);
        }
        if (prior.empty())
          segments_.pop_back();
      }
      size_t content = open + 2 + (trim_left ? 1 : 0);
      const std::string close = kind == '{' ? "}}" : kind == '%' ? "%}" : "#}";
      size_t at = content;
      char quote = 0;
      std::vector<char> brackets;
      for (; at + 1 < source_.size(); ++at) {
        char c = source_[at];
        if (kind != '#' && quote) {
          if (c == '\\') {
            ++at;
            continue;
          }
          if (c == quote)
            quote = 0;
          continue;
        }
        if (kind != '#' && (c == '\'' || c == '"')) {
          quote = c;
          continue;
        }
        if (brackets.empty() && source_.substr(at, 2) == close)
          break;
        if (kind != '#') {
          if (c == '(' || c == '[' || c == '{')
            brackets.push_back(c);
          else if ((c == ')' || c == ']' || c == '}') && !brackets.empty())
            brackets.pop_back();
        }
      }
      if (at + 1 >= source_.size())
        return Error(open, "unterminated template tag");
      bool trim_right = at > content && source_[at - 1] == '-';
      if (at > content && source_[at - 1] == '+')
        return Error(at - 1,
                     "explicit plus whitespace control is not supported");
      size_t finish = at - (trim_right ? 1 : 0);
      if (kind != '#')
        segments_.push_back({kind == '{' ? Segment::Output : Segment::Statement,
                             "", content, finish});
      cursor = at + 2;
      if (trim_right) {
        while (cursor < source_.size()) {
          uint32_t cp;
          size_t n = CodePoint(source_, cursor, &cp);
          if (!n || !UnicodeSpace(cp))
            break;
          cursor += n;
        }
      } else if (kind != '{' && cursor < source_.size()) {
        if (source_[cursor] == '\r') {
          ++cursor;
          if (cursor < source_.size() && source_[cursor] == '\n')
            ++cursor;
        } else if (source_[cursor] == '\n')
          ++cursor;
      }
    }
    return true;
  }
  std::pair<std::string_view, size_t> Keyword(const Segment &s) const {
    size_t begin = s.begin;
    while (begin < s.end && Space(source_[begin]))
      ++begin;
    size_t end = begin;
    while (end < s.end && NamePart(source_[end]))
      ++end;
    return {source_.substr(begin, end - begin), end};
  }
  bool EmptyTail(const Segment &s, size_t at) {
    while (at < s.end && Space(source_[at]))
      ++at;
    return at == s.end || Error(at, "unexpected block arguments");
  }
  bool Nodes(std::initializer_list<std::string_view> stop,
             std::vector<Node> *out, size_t depth) {
    if (depth > 64)
      return Error(position_ < segments_.size() ? segments_[position_].begin
                                                : source_.size(),
                   "template block nesting exceeds compiler limit");
    while (position_ < segments_.size()) {
      const Segment &s = segments_[position_];
      auto keyword = s.kind == Segment::Statement
                         ? Keyword(s)
                         : std::pair<std::string_view, size_t>{"", 0};
      if (s.kind == Segment::Statement &&
          std::find(stop.begin(), stop.end(), keyword.first) != stop.end())
        return true;
      Node node;
      node.offset = s.begin;
      node.end = s.end;
      if (s.kind == Segment::Text) {
        node.kind = Node::Kind::Text;
        node.text = s.text;
        ++position_;
      } else if (s.kind == Segment::Output) {
        node.kind = Node::Kind::Output;
        if (!ExpressionParser(source_, s.begin, s.end, diagnostic_)
                 .Parse(&node.expression))
          return false;
        ++position_;
      } else if (keyword.first == "set") {
        node.kind = Node::Kind::Set;
        bool capture = false;
        if (!ExpressionParser(source_, keyword.second, s.end, diagnostic_)
                 .ParseSet(&node.text, &node.expression, &capture))
          return false;
        ++position_;
        if (capture) {
          node.kind = Node::Kind::Capture;
          if (!Nodes({"endset"}, &node.body, depth + 1))
            return false;
          if (position_ == segments_.size())
            return Error(s.begin, "missing endset");
          const auto closing = Keyword(segments_[position_]);
          if (!EmptyTail(segments_[position_], closing.second))
            return false;
          node.end = segments_[position_++].end;
        }
      } else if (keyword.first == "if") {
        if (!If(&node, depth))
          return false;
      } else if (keyword.first == "generation") {
        node.kind = Node::Kind::Generation;
        if (!EmptyTail(s, keyword.second))
          return false;
        ++position_;
        if (!Nodes({"endgeneration"}, &node.body, depth + 1))
          return false;
        if (position_ == segments_.size())
          return Error(s.begin, "missing endgeneration");
        const auto closing = Keyword(segments_[position_]);
        if (!EmptyTail(segments_[position_], closing.second))
          return false;
        node.end = segments_[position_++].end;
      } else if (keyword.first == "macro") {
        node.kind = Node::Kind::Macro;
        if (!ExpressionParser(source_, keyword.second, s.end, diagnostic_)
                 .ParseMacro(&node.text, &node.parameters, &node.defaults))
          return false;
        ++position_;
        if (!Nodes({"endmacro"}, &node.body, depth + 1))
          return false;
        if (position_ == segments_.size())
          return Error(s.begin, "missing endmacro");
        const auto closing = Keyword(segments_[position_]);
        if (!EmptyTail(segments_[position_], closing.second))
          return false;
        node.end = segments_[position_++].end;
      } else if (keyword.first == "for") {
        node.kind = Node::Kind::For;
        if (!ExpressionParser(source_, keyword.second, s.end, diagnostic_)
                 .ParseFor(&node.text, &node.expression, &node.predicate))
          return false;
        ++position_;
        if (!Nodes({"else", "endfor"}, &node.body, depth + 1))
          return false;
        if (position_ == segments_.size())
          return Error(s.begin, "missing endfor");
        auto closing = Keyword(segments_[position_]);
        if (!EmptyTail(segments_[position_], closing.second))
          return false;
        if (closing.first == "else") {
          ++position_;
          if (!Nodes({"endfor"}, &node.otherwise, depth + 1))
            return false;
          if (position_ == segments_.size())
            return Error(s.begin, "missing endfor");
          closing = Keyword(segments_[position_]);
          if (!EmptyTail(segments_[position_], closing.second))
            return false;
        }
        node.end = segments_[position_++].end;
      } else
        return Error(s.begin, "unsupported or misplaced template statement: " +
                                  std::string(keyword.first));
      out->push_back(std::move(node));
    }
    return true;
  }
  bool If(Node *node, size_t depth) {
    const Segment &s = segments_[position_];
    auto keyword = Keyword(s);
    node->kind = Node::Kind::If;
    node->offset = s.begin;
    if (!ExpressionParser(source_, keyword.second, s.end, diagnostic_)
             .Parse(&node->expression))
      return false;
    ++position_;
    if (!Nodes({"elif", "else", "endif"}, &node->body, depth + 1))
      return false;
    if (position_ == segments_.size())
      return Error(s.begin, "missing endif");
    auto closing = Keyword(segments_[position_]);
    if (closing.first == "elif") {
      Node nested;
      if (!If(&nested, depth + 1))
        return false;
      node->end = nested.end;
      node->otherwise.push_back(std::move(nested));
      return true;
    }
    if (!EmptyTail(segments_[position_], closing.second))
      return false;
    if (closing.first == "else") {
      ++position_;
      if (!Nodes({"endif"}, &node->otherwise, depth + 1))
        return false;
      if (position_ == segments_.size())
        return Error(s.begin, "missing endif");
      closing = Keyword(segments_[position_]);
      if (!EmptyTail(segments_[position_], closing.second))
        return false;
    }
    node->end = segments_[position_++].end;
    return true;
  }
  std::string_view source_;
  Diagnostic *diagnostic_;
  std::vector<Segment> segments_;
  size_t position_ = 0;
};
} // namespace

bool ParseTemplate(std::string_view source, Template *out,
                   Diagnostic *diagnostic) {
  if (!out) {
    if (diagnostic) {
      diagnostic->offset = 0;
      diagnostic->message = "parser needs an output slot";
    }
    return false;
  }
  return TemplateParser(source, diagnostic).Parse(out);
}
} // namespace hrx::loomify_gguf::jinja
