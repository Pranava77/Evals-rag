import pytest

from src.llm_client import FakeLLMClient
from src.metrics import recall_at_k, faithfulness


def test_recall_full_hit():
    assert recall_at_k({"a", "b"}, ["a", "b", "c"]) == 1.0


def test_recall_partial_and_miss():
    assert recall_at_k({"a", "b"}, ["a", "x", "y"]) == 0.5
    assert recall_at_k({"a"}, ["x", "y"]) == 0.0


def test_recall_empty_expected_is_perfect():
    assert recall_at_k(set(), ["anything"]) == 1.0


def test_faithfulness_parses_judge_verdict():
    judge = FakeLLMClient(scripted=['{"grounded": true, "why": "supported"}'])
    assert faithfulness(judge, "answer", ["context"]) is True
    judge = FakeLLMClient(scripted=['Sure! {"grounded": false, "why": "claim not in context"}'])
    assert faithfulness(judge, "answer", ["context"]) is False


def test_faithfulness_raises_on_garbage_judge():
    judge = FakeLLMClient(scripted=["no json here at all"])
    with pytest.raises(ValueError):
        faithfulness(judge, "a", ["c"])
