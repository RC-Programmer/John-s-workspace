"""
tasks.py — Task definitions for the local AI crew.

A Task tells an agent:
  - WHAT  to do (description)
  - WHAT  success looks like (expected_output)
  - WHO   does it (agent)
  - WHAT  it depends on (context — a list of upstream tasks)

Tasks are assembled into a Crew in main.py.  Changing the kickoff topic only
requires editing the description strings — or better, passing them as
f-strings / templates from main.py (see build_research_task / build_write_task).

Current pipeline:
  [research_task] ──context──▶ [writing_task]
"""

from __future__ import annotations

from crewai import Agent, Task


# ─────────────────────────────────────────────────────────────────────────────
# Task factories
#
# Factory functions (not module-level singletons) so you can parameterise the
# topic / format at runtime from main.py.
# ─────────────────────────────────────────────────────────────────────────────

def build_research_task(
    agent: Agent,
    topic: str,
    output_file: str = "output/research_notes.md",
) -> Task:
    """
    Task 1 — Research.

    The researcher searches the web and compiles structured notes that the
    writer will use in the next task.

    Args:
        agent       : the Researcher agent (from agents.py)
        topic       : plain-language description of what to research
        output_file : where to save the raw research notes (markdown)
    """
    return Task(
        description=f"""\
Conduct thorough research on the following topic:

TOPIC: {topic}

Step-by-step instructions:
1. Break the topic into 3–5 focused sub-questions.
2. Run DuckDuckGo searches for each sub-question.
3. Collect and verify key facts, numbers, model names, benchmark scores, and dates.
4. Cross-check any surprising claims with a second search.
5. Organise your notes under clear headings in Markdown.
6. Save your notes to: {output_file}

Research note requirements:
- Include at least 5 distinct sources with their URLs.
- For any comparison (e.g. models ranked by performance), include a mini table.
- Note anything you could NOT verify so the writer knows not to assert it.
- Keep notes factual and neutral — no editorial spin yet.
""",
        expected_output="""\
A well-structured Markdown research brief saved to disk, containing:
- An executive summary (3–5 sentences)
- Numbered sections, one per sub-question
- A source list with clickable URLs
- Any caveats / unverified items flagged with [UNVERIFIED]
- At least one comparison table if the topic involves ranking or comparing items
""",
        agent=agent,
        output_file=output_file,
    )


def build_write_task(
    agent: Agent,
    topic: str,
    research_task: Task,
    word_count: int = 600,
    output_format: str = "markdown report with a comparison table",
    output_file: str = "output/final_report.md",
) -> Task:
    """
    Task 2 — Writing.

    The writer reads the researcher's notes (passed via context) and produces
    a polished final document.

    Args:
        agent           : the Writer agent (from agents.py)
        topic           : same topic string used in the research task
        research_task   : the upstream Task whose output provides context
        word_count      : target word count for the final document
        output_format   : description of the desired output format
        output_file     : where to save the final document
    """
    return Task(
        description=f"""\
Using the research notes provided in context, write a final document on:

TOPIC: {topic}

Output requirements:
- Format  : {output_format}
- Length  : approximately {word_count} words (±10 %)
- Audience: software practitioners and tech-savvy users
- Tone    : clear, informative, slightly conversational — no jargon without explanation

Document structure (adapt headings to suit the topic):
1. ## Introduction  — 2–3 sentences setting context
2. ## Key Findings  — the most important facts from the research
3. ## Comparison Table — markdown table comparing the main items
4. ## Recommendations — 3–5 concrete, actionable bullet points
5. ## Sources — list all URLs from the research notes

Hard rules:
- Do NOT invent information not present in the research notes.
- If a fact was marked [UNVERIFIED] in the research notes, exclude it or
  explicitly label it as unconfirmed.
- Save the final document to: {output_file}
- The file must be valid Markdown renderable by any standard viewer.
""",
        expected_output=f"""\
A polished ~{word_count}-word Markdown document saved to {output_file}, with:
- A clear introduction
- A well-formatted comparison table
- Concrete recommendations tailored to the stated constraints
- A source list matching the research notes
- No hallucinated facts
""",
        agent=agent,
        context=[research_task],   # writer receives researcher's output as context
        output_file=output_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# (Optional) Reviewer task — uncomment when you add the reviewer agent
# ─────────────────────────────────────────────────────────────────────────────

def build_review_task(
    agent: Agent,
    write_task: Task,
    output_file: str = "output/review_notes.md",
) -> Task:
    """
    Task 3 (optional) — Editorial review.

    The reviewer checks the writer's draft for accuracy and quality, then
    saves a structured feedback document.
    """
    return Task(
        description="""\
Review the draft document provided in context.

Evaluate it on these dimensions:
1. Factual accuracy  — does every claim match the research notes?
2. Completeness      — are all required sections present?
3. Clarity           — is each sentence unambiguous?
4. Table quality     — is the comparison table correct and well-formatted?
5. Recommendations   — are they specific and actionable?

For each issue found, write:
  - LOCATION: (section / paragraph)
  - ISSUE: (what is wrong)
  - FIX: (exactly what to change)

Also give an overall quality score: 1–10 with justification.
""",
        expected_output="""\
A structured review document with:
- An overall score (1–10) and one-line verdict
- A numbered list of issues (can be empty if the draft is solid)
- Each issue with LOCATION / ISSUE / FIX fields
- A final APPROVED / NEEDS REVISION verdict
""",
        agent=agent,
        context=[write_task],
        output_file=output_file,
    )
