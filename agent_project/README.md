# Local AI Agent System — Ollama + CrewAI

100% offline multi-agent AI crew running on your laptop.
No OpenAI key. No bills. No data leaving your machine.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10–3.13 (3.12 recommended) | **3.14+ not yet supported by crewai** |
| [Ollama](https://ollama.ai) | latest | download from ollama.com |
| A pulled model | — | `ollama pull qwen2.5:14b` |

---

## Quick start

```bash
# 1. Clone / enter project
cd agent_project

# 2. Create virtual env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional — defaults work out of the box)
cp .env.example .env
# edit .env if you want a different model or port

# 5. Start Ollama in a separate terminal
ollama serve

# 6. Pull the model (one-time download, ~9 GB for 14b models)
ollama pull qwen2.5:14b

# 7. Run the crew
python main.py
```

The run produces three files in `output/`:
- `research_notes.md`  — raw research gathered by the Researcher
- `final_report.md`    — polished Markdown report by the Writer
- `review_notes.md`    — quality review (only with `--review` flag)

---

## CLI options

```
python main.py --help

  --topic       Research topic (default: local AI models for 16 GB laptop)
  --model       Ollama model tag (default: ollama/qwen2.5:14b)
  --base-url    Ollama URL (default: http://localhost:11434)
  --temp        Temperature 0.0–1.0 (default: 0.7)
  --max-tokens  Max tokens per response (default: 4096)
  --word-count  Target word count for final report (default: 600)
  --review      Add the Reviewer agent as a third step
```

### Examples

```bash
# Custom topic
python main.py --topic "Compare SQLite vs DuckDB for local analytics in Python"

# Use a lighter model on 8 GB RAM
python main.py --model ollama/mistral:7b --max-tokens 2048

# Deep reasoning with DeepSeek + editorial review
python main.py --model ollama/deepseek-r1:14b --review --temp 0.3

# Verbose research run
python main.py --topic "Rust async runtimes compared: Tokio vs async-std 2026" \
               --word-count 800 --review
```

---

## Project structure

```
agent_project/
├── main.py          # entry point — assembles & runs the crew
├── agents.py        # Agent objects (researcher, writer, reviewer)
├── tasks.py         # Task objects (research, write, review)
├── tools.py         # DuckDuckGo search, file I/O, calculator
├── config/
│   ├── agents.yaml  # reference config / future YAML-driven setup
│   └── tasks.yaml
├── output/          # generated at runtime — gitignored
├── .env.example     # copy to .env and edit
├── requirements.txt
└── README.md
```

---

## Adding a new tool

1. Open `tools.py`
2. Add a new function decorated with `@tool("Tool Name")`
3. Write a clear docstring — the LLM reads this to decide when to use it
4. Append it to the relevant list at the bottom (`RESEARCHER_TOOLS`, etc.)

```python
@tool("List Directory")
def list_directory(path: str) -> str:
    """List files in a directory. Input: directory path. Output: file list."""
    from pathlib import Path
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"[Error] Not a directory: {p}"
    return "\n".join(str(f) for f in sorted(p.iterdir()))
```

---

## Adding a new agent

1. Add a `create_<role>` factory function in `agents.py`
2. Add a `build_<role>_task` factory in `tasks.py`
3. Wire it into the `agents` and `tasks` lists in `main.py`

---

## Switching models

Edit `.env` or pass `--model` on the CLI:

| Model | RAM needed | Best for |
|---|---|---|
| `ollama/qwen2.5:14b` | ~10 GB | Reasoning, coding, analysis |
| `ollama/llama3.2:11b` | ~8 GB | Fast general purpose |
| `ollama/deepseek-r1:14b` | ~10 GB | Multi-step reasoning, math |
| `ollama/mistral:7b` | ~5 GB | Lightweight, speed |
| `ollama/phi4:14b` | ~9 GB | Instruction following |

---

## Enabling long-term memory (optional)

CrewAI supports vector-store memory via ChromaDB.  To enable:

```bash
pip install chromadb
```

Then in `main.py`, set `memory=True` on the `Crew` object.
A local Chroma DB will be created in `./memory/`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| `model not found` | Run `ollama pull <model-name>` |
| Slow responses | Use a smaller model (`mistral:7b`) or reduce `--max-tokens` |
| Out of memory | Use `--model ollama/mistral:7b` or close other apps |
| `duckduckgo_search` rate limit | Add a delay or reduce `max_results` in `tools.py` |
