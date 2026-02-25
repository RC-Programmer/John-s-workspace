"""
main.py — Entry point.  Assembles and runs the local AI crew.

Usage:
  python main.py                          # run with default topic & model
  python main.py --topic "your topic"     # custom topic
  python main.py --model ollama/llama3.2:11b  # different Ollama model
  python main.py --review                 # add the optional reviewer agent

Environment variables (override via .env or shell export):
  OLLAMA_MODEL      e.g. ollama/qwen2.5:14b  (default)
  OLLAMA_BASE_URL   e.g. http://localhost:11434
  OLLAMA_TEMP       float, default 0.7
  OLLAMA_MAX_TOKENS int,   default 4096
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

import httpx
from crewai import Crew, Process
from dotenv import load_dotenv

from agents import build_llm, create_researcher, create_writer, create_reviewer
from tasks import build_research_task, build_write_task, build_review_task


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults (overridden by .env or CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # loads .env if present, silently ignores if not

DEFAULT_MODEL      = os.getenv("OLLAMA_MODEL",      "ollama/qwen2.5:14b")
DEFAULT_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
DEFAULT_TEMP       = float(os.getenv("OLLAMA_TEMP", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))

DEFAULT_TOPIC = (
    "Research the latest free local AI models ranked by reasoning ability "
    "in early 2026, then compile findings for a laptop with 16GB RAM."
)

OUTPUT_DIR = Path("output")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama(base_url: str, timeout: float = 5.0) -> None:
    """
    Verify Ollama is reachable before starting the crew.

    Raises SystemExit with a helpful message if Ollama is not running so the
    user gets a clear error instead of a cryptic litellm stack trace.
    """
    health_url = base_url.rstrip("/") + "/"
    try:
        resp = httpx.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            print(f"[OK]   Ollama is running at {base_url}")
            return
        print(f"[WARN] Ollama responded with HTTP {resp.status_code} — proceeding anyway.")
    except httpx.ConnectError:
        _die(
            f"Cannot connect to Ollama at {base_url}\n\n"
            "  → Make sure Ollama is installed and running:\n"
            "       ollama serve\n\n"
            "  → Then pull the model you want to use:\n"
            "       ollama pull qwen2.5:14b\n\n"
            "  → Or change OLLAMA_BASE_URL in your .env file."
        )
    except httpx.TimeoutException:
        _die(f"Timed out connecting to Ollama at {base_url} (timeout={timeout}s)")


def _die(message: str) -> None:
    """Print a friendly error and exit."""
    border = "─" * 60
    print(f"\n{border}\n[ERROR] {message}\n{border}\n", file=sys.stderr)
    sys.exit(1)


def _banner(title: str, char: str = "═", width: int = 70) -> str:
    pad = max(0, width - len(title) - 2)
    left = pad // 2
    right = pad - left
    return f"{char * left} {title} {char * right}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local AI crew with Ollama + CrewAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --topic "Compare Rust vs Go for CLI tooling in 2026"
              python main.py --model ollama/deepseek-r1:14b --temp 0.3
              python main.py --review
        """),
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Research topic / question for the crew to tackle",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP,
        help=f"LLM temperature, 0.0–1.0 (default: {DEFAULT_TEMP})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per LLM response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Add the optional Reviewer agent as a third step",
    )
    parser.add_argument(
        "--word-count",
        type=int,
        default=600,
        help="Target word count for the final report (default: 600)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Print run config ──────────────────────────────────────────────────────
    print(_banner("LOCAL AI AGENT CREW"))
    print(f"  Model     : {args.model}")
    print(f"  Ollama URL: {args.base_url}")
    print(f"  Topic     : {args.topic[:80]}{'…' if len(args.topic) > 80 else ''}")
    print(f"  Reviewer  : {'enabled' if args.review else 'disabled'}")
    print()

    # ── Preflight: confirm Ollama is up ───────────────────────────────────────
    check_ollama(args.base_url)
    print()

    # ── Create output directory ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build the shared LLM ──────────────────────────────────────────────────
    llm = build_llm(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temp,
        max_tokens=args.max_tokens,
    )

    # ── Instantiate agents ────────────────────────────────────────────────────
    researcher = create_researcher(llm)
    writer     = create_writer(llm)

    # ── Assemble tasks ────────────────────────────────────────────────────────
    research_task = build_research_task(
        agent=researcher,
        topic=args.topic,
        output_file=str(OUTPUT_DIR / "research_notes.md"),
    )
    write_task = build_write_task(
        agent=writer,
        topic=args.topic,
        research_task=research_task,
        word_count=args.word_count,
        output_format="markdown report with a comparison table and recommendations",
        output_file=str(OUTPUT_DIR / "final_report.md"),
    )

    agents = [researcher, writer]
    tasks  = [research_task, write_task]

    # ── Optional reviewer step ────────────────────────────────────────────────
    if args.review:
        reviewer     = create_reviewer(llm)
        review_task  = build_review_task(
            agent=reviewer,
            write_task=write_task,
            output_file=str(OUTPUT_DIR / "review_notes.md"),
        )
        agents.append(reviewer)
        tasks.append(review_task)
        print("[INFO] Reviewer agent added to the crew.\n")

    # ── Assemble and run the crew ─────────────────────────────────────────────
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,  # tasks run in order; switch to hierarchical
                                     # when you want a manager agent to delegate
        verbose=True,                # shows each agent's reasoning steps
        memory=False,                # set True to enable long-term vector memory
                                     # (requires extra setup — see README)
    )

    print(_banner("STARTING CREW RUN"))
    print()

    start = time.perf_counter()
    try:
        result = crew.kickoff()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Run cancelled by user.")
        sys.exit(0)
    elapsed = time.perf_counter() - start

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(_banner("RUN COMPLETE"))
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Outputs :")
    for f in sorted(OUTPUT_DIR.glob("*.md")):
        size = f.stat().st_size
        print(f"    {f}  ({size:,} bytes)")
    print()

    # Print the final result (the last task's output) to stdout
    print(_banner("FINAL OUTPUT"))
    print()

    # crew.kickoff() returns a CrewOutput object; .raw gives the plain string
    final_text = getattr(result, "raw", str(result))
    print(final_text)
    print()
    print(_banner("END"))


if __name__ == "__main__":
    main()
