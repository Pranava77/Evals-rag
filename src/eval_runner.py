"""Run the golden set against the RAG seed; print the table; gate the build.

    python -m src.eval_runner                    # healthy → exit 0
    python -m src.eval_runner --break-chunking   # sabotaged → gate FAIL, exit 1
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from .metrics import faithfulness, recall_at_k

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "golden.jsonl")

RECALL_BAR = 0.8
FAITH_BAR = 0.6


@dataclass
class Report:
    rows: list
    recall: float
    faith: float


def load_golden(path: str = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def run_eval(rag, judge_llm, golden: list[dict]) -> Report:
    rows, recalls, faiths = [], [], []
    for case in golden:
        hits = rag.retrieve(case["q"])
        r = recall_at_k(set(case["expected_ids"]), [h.id for h in hits])
        if case["kind"] == "out-of-scope":
            rows.append((case["q"][:38], "—", "n/a"))
            continue
        answer, hits = rag.ask(case["q"])
        f = faithfulness(judge_llm, answer, [h.text for h in hits])
        recalls.append(r)
        faiths.append(1.0 if f else 0.0)
        rows.append((case["q"][:38], f"{r:.2f}", "yes" if f else "NO"))
    return Report(rows,
                  sum(recalls) / len(recalls) if recalls else 0.0,
                  sum(faiths) / len(faiths) if faiths else 0.0)


def gate(report: Report) -> bool:
    """PASS only if recall >= RECALL_BAR AND faith >= FAITH_BAR."""
    return report.recall >= RECALL_BAR and report.faith >= FAITH_BAR


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from .llm_client import FakeLLMClient, ClaudeClient, OllamaClient
    from .rag_seed import RagSeed, broken_seed

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = judge = ClaudeClient()
    else:
        llm = judge = OllamaClient()

    golden = load_golden()
    rag = broken_seed(llm) if "--break-chunking" in sys.argv else RagSeed(llm)

    report = run_eval(rag, judge, golden)
    print(f"{'case':40} {'recall':>7} {'faithful':>9}")
    for q, r, f in report.rows:
        print(f"{q:40} {r:>7} {f:>9}")
    print(f"\nMEAN  recall={report.recall:.2f}  faith={report.faith:.2f}")
    if gate(report):
        print(f"GATE: PASS (≥ {RECALL_BAR}/{FAITH_BAR})")
    else:
        print(f"GATE: FAIL (below {RECALL_BAR}/{FAITH_BAR}) — build goes red")
        sys.exit(1)


if __name__ == "__main__":
    main()
