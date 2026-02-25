"""
tools.py — Custom tools available to all agents.

Each tool is decorated with @tool so CrewAI agents can discover and invoke it
automatically. The function's docstring IS the tool description the LLM sees,
so keep it clear and specific.

Tools provided:
  - duckduckgo_search   : free web search, no API key
  - read_file           : read any text file from disk
  - write_file          : write / overwrite a text file on disk
  - append_file         : append lines to an existing file (or create it)
  - calculate           : safe arithmetic / math expression evaluator
"""

from __future__ import annotations

import ast
import math
import operator
import os
from pathlib import Path
from typing import Any

from crewai.tools import tool
from duckduckgo_search import DDGS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Web Search
# ─────────────────────────────────────────────────────────────────────────────

@tool("DuckDuckGo Web Search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return the top results as plain text.

    Use this tool whenever you need current information, facts, comparisons,
    or anything that requires browsing the internet.

    Input : a plain-text search query (e.g. 'best local LLMs for 16GB RAM 2026')
    Output: up to 5 results, each with title, URL, and a short snippet.
    """
    max_results: int = 5

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"[Search failed] {exc}\nTip: confirm you have an internet connection."

    if not raw:
        return "No results found. Try rephrasing the query."

    lines: list[str] = []
    for i, r in enumerate(raw, 1):
        title   = r.get("title", "No title")
        href    = r.get("href", "")
        snippet = r.get("body", "No description")
        lines.append(f"[{i}] {title}\n    URL: {href}\n    {snippet}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. File I/O
# ─────────────────────────────────────────────────────────────────────────────

@tool("Read File")
def read_file(file_path: str) -> str:
    """
    Read the entire contents of a text file and return them as a string.

    Use this tool when you need to load existing notes, previous research,
    configuration, or any text-based document from disk.

    Input : absolute or relative path to the file (e.g. 'output/research.md')
    Output: the raw text content of the file, or an error message.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"[Error] File not found: {path}"
    if not path.is_file():
        return f"[Error] Path is a directory, not a file: {path}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[Error] Could not read file: {exc}"


@tool("Write File")
def write_file(file_path: str, content: str) -> str:
    """
    Write (or overwrite) a text file with the given content.

    Use this tool to save research notes, final reports, or any generated text
    to disk so other agents or the user can access them later.

    Input : file_path — path to write to (parent dirs are created automatically)
            content   — the full text to write
    Output: confirmation message or error description.
    """
    path = Path(file_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"[OK] Wrote {len(content)} characters to '{path}'."
    except Exception as exc:
        return f"[Error] Could not write file: {exc}"


@tool("Append to File")
def append_file(file_path: str, content: str) -> str:
    """
    Append text to the end of an existing file, or create it if it does not exist.

    Useful for incrementally building up a research log without overwriting
    earlier entries.

    Input : file_path — path of the target file
            content   — text to append (a newline is added automatically)
    Output: confirmation message or error description.
    """
    path = Path(file_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content + "\n")
        return f"[OK] Appended {len(content)} characters to '{path}'."
    except Exception as exc:
        return f"[Error] Could not append to file: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Safe Math / Calculator
# ─────────────────────────────────────────────────────────────────────────────

# Whitelist of safe AST node types — prevents arbitrary code execution
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,       # Python < 3.8 literal numbers
    ast.Constant,  # Python 3.8+ literals
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Call,      # needed for math.sqrt() etc.
    ast.Attribute, # needed for math.pi etc.
    ast.Name,      # needed for named constants like 'pi', 'e'
)

# Safe builtins exposed to the expression evaluator
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},   # no builtins at all
    # math module functions
    "sqrt": math.sqrt, "cbrt": math.pow,  # cbrt via pow(x,1/3)
    "log":  math.log,  "log2": math.log2, "log10": math.log10,
    "sin":  math.sin,  "cos":  math.cos,  "tan":  math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "ceil": math.ceil, "floor": math.floor, "abs": abs,
    "round": round,
    # constants
    "pi": math.pi, "e": math.e, "tau": math.tau,
}


def _safe_eval(expression: str) -> float:
    """Parse and evaluate a math expression with a restricted AST."""
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Unsafe operation in expression: {type(node).__name__}")
    # compile + eval inside our safe namespace
    code = compile(tree, "<string>", "eval")
    return eval(code, _SAFE_GLOBALS)  # noqa: S307 — guarded by AST whitelist


@tool("Calculator")
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, //, %, ** (power), sqrt(), log(), sin(), cos(),
    ceil(), floor(), abs(), round(), and the constants pi, e, tau.

    Does NOT execute arbitrary Python code — only safe math operations.

    Input : a math expression as a string  (e.g. '2 ** 10 + sqrt(144)')
    Output: the numeric result as a string, or an error message.

    Examples:
      '100 / 7'            → '14.285714285714286'
      'sqrt(2) * pi'       → '4.442882938158366'
      '(1 + 0.05) ** 10'   → '1.6288946267774418'
    """
    try:
        result = _safe_eval(expression)
        # Return a clean representation — strip trailing zeros for integers
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "[Error] Division by zero."
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"[Error] Invalid expression: {exc}"
    except Exception as exc:
        return f"[Error] {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: a list you can import in agents.py
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_TOOLS   = [duckduckgo_search]
FILE_TOOLS     = [read_file, write_file, append_file]
MATH_TOOLS     = [calculate]
ALL_TOOLS      = SEARCH_TOOLS + FILE_TOOLS + MATH_TOOLS
RESEARCHER_TOOLS = SEARCH_TOOLS + [read_file, append_file] + MATH_TOOLS
WRITER_TOOLS   = [read_file, write_file]
