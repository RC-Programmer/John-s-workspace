"""
agents.py — Agent definitions for the local AI crew.

Each agent is a specialist with:
  - a focused role + goal + backstory  (what the LLM uses to stay in character)
  - a curated set of tools             (what it can actually do)
  - the shared local_llm               (Ollama endpoint)

Adding a new agent later is as simple as:
  1. Define it here with its tools & persona
  2. Add it to the crew in main.py
  3. Give it tasks in tasks.py

Current roster:
  researcher  — gathers facts, searches the web, reads files
  writer      — turns research notes into polished markdown output
  (stub)      — reviewer placeholder for future use
"""

from __future__ import annotations

from crewai import Agent
from langchain_ollama import ChatOllama

from tools import RESEARCHER_TOOLS, WRITER_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm(
    model: str = "ollama/qwen2.5:14b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> ChatOllama:
    """
    Return a LangChain ChatOllama instance pointed at a local Ollama server.

    Uses langchain-ollama directly to avoid the litellm version conflict with
    crewai 1.9.x. CrewAI agents accept any LangChain-compatible LLM.

    Args:
        model       : Ollama model tag; the 'ollama/' prefix is stripped
                      automatically.  e.g. 'ollama/qwen2.5:14b'
        base_url    : Ollama server URL (default: local Ollama)
        temperature : 0.0 = deterministic, 1.0 = creative
        max_tokens  : max tokens in a single LLM response (num_predict)
    """
    model_name = model.removeprefix("ollama/")
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory functions
#
# Using factory functions (instead of module-level singletons) means the LLM
# object is created once in main.py and passed in — easier to swap models or
# run experiments without touching this file.
# ─────────────────────────────────────────────────────────────────────────────

def create_researcher(llm: ChatOllama) -> Agent:
    """
    The Researcher: digs up accurate, up-to-date information.

    Tools available:
      - DuckDuckGo search  (web queries)
      - read_file          (load local documents)
      - append_file        (save intermediate notes)
      - calculate          (quick math while analysing data)
    """
    return Agent(
        role="Senior Research Analyst",
        goal=(
            "Find accurate, comprehensive, and up-to-date information on the "
            "given topic. Gather multiple sources, cross-verify key claims, "
            "and compile structured research notes that give the writer "
            "everything they need — including specific data points, URLs, and "
            "direct quotes where relevant."
        ),
        backstory=(
            "You are a meticulous research analyst with a decade of experience "
            "in technology journalism and AI benchmarking. You never make up "
            "facts — if you can't find something, you say so. You always cite "
            "your sources and prefer recent, primary sources over opinion pieces. "
            "You know how to write concise but information-dense research briefs."
        ),
        tools=RESEARCHER_TOOLS,
        llm=llm,
        verbose=True,          # print each agent's thought process
        allow_delegation=False, # keep it simple; set True for hierarchical crews
        max_iter=8,            # max reasoning iterations before forced answer
        memory=True,           # remember context within a run
    )


def create_writer(llm: ChatOllama) -> Agent:
    """
    The Writer: transforms research notes into polished output.

    Tools available:
      - read_file   (load research notes produced by the researcher)
      - write_file  (save the final document to disk)
    """
    return Agent(
        role="Technical Content Writer",
        goal=(
            "Take raw research notes and craft a clear, well-structured, "
            "engaging document. Match the requested output format exactly "
            "(markdown report, comparison table, email, etc.). Maintain "
            "factual accuracy — do not invent information not present in "
            "the research notes."
        ),
        backstory=(
            "You are a technical writer who specialises in making complex AI "
            "and software topics accessible to practitioners. You have a "
            "strong eye for structure: you know when to use a table vs a list, "
            "how to write an executive summary, and how to end with actionable "
            "recommendations. You always respect word-count targets."
        ),
        tools=WRITER_TOOLS,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )


def create_reviewer(llm: ChatOllama) -> Agent:
    """
    (Optional) The Reviewer: quality-checks the writer's output.

    Uncomment in main.py when you're ready to add a third agent.
    """
    return Agent(
        role="Editorial Reviewer & Fact-Checker",
        goal=(
            "Review the draft document for factual accuracy, logical consistency, "
            "completeness, and clarity. Flag any claims that contradict the "
            "research notes. Suggest concrete, specific improvements — don't "
            "just say 'improve clarity', say exactly what to change and why."
        ),
        backstory=(
            "You are a sharp-eyed editor who worked at a major tech publication "
            "for 15 years. You can spot an unsubstantiated claim from a mile away "
            "and you hold every document to the same high standard regardless of "
            "the topic. Your feedback is direct but constructive."
        ),
        tools=[],          # reviewer only needs to read the draft in-context
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        memory=True,
    )
