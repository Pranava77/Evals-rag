"""The two core metrics — deliberately simple, inspectable code.

recall@k     : pure set overlap (no LLM, free, microseconds)
faithfulness : one LLM-judge call, binary verdict, JSON-validated
"""
from __future__ import annotations

import json

VERDICT = 'Reply with ONLY JSON: {"grounded": true or false, "why": "..."}'

JUDGE_PROMPT = ("Is every claim in ANSWER supported by CONTEXT? "
                + VERDICT + "\nANSWER: <answer>\nCONTEXT: <context>")


def recall_at_k(expected: set[str], retrieved: list[str]) -> float:
    """|expected ∩ retrieved| / |expected|.
    Empty expected -> 1.0."""
    if not expected:
        return 1.0
    return len(expected & set(retrieved)) / len(expected)


def _extract_json(text: str) -> dict:
    i, j = text.find("{"), text.rfind("}")
    if i == -1:
        raise ValueError("no JSON in judge reply")
    return json.loads(text[i:j + 1])


def faithfulness(judge_llm, answer: str, chunks: list[str]) -> bool:
    """LLM-as-judge, kept honest: binary, validated. Cheap tier is fine."""
    context = "\n\n".join(chunks)
    prompt = JUDGE_PROMPT.replace("<answer>", answer).replace("<context>", context)
    reply = judge_llm.generate(prompt)
    verdict = _extract_json(reply)
    return verdict["grounded"]
