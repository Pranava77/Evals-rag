"""Prompt A/B on the same golden set — evidence over vibes.

    python -m src.prompt_ab
"""
from __future__ import annotations

import os

from .eval_runner import load_golden, run_eval
from .rag_seed import RagSeed, PROMPT

P_V1 = PROMPT  # baseline
P_V2 = PROMPT.replace("Cite the doc name.",
                      "Cite the doc name in [brackets] after every claim.")


def compare(llm, judge):
    golden = load_golden()
    out = {}
    for name, prompt in {"v1": P_V1, "v2": P_V2}.items():
        rag = RagSeed(llm, prompt=prompt)
        out[name] = run_eval(rag, judge, golden)
    return out


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from .llm_client import ClaudeClient, OllamaClient
    llm = judge = ClaudeClient() if os.getenv("ANTHROPIC_API_KEY") else OllamaClient()
    scores = compare(llm, judge)
    print(f"{'prompt':8} {'recall':>7} {'faith':>7}")
    for name, rep in scores.items():
        print(f"{name:8} {rep.recall:>7.2f} {rep.faith:>7.2f}")
    print("\nSame set, one change apart — the winner is a number, not a feeling.")


if __name__ == "__main__":
    main()
