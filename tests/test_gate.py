from src.eval_runner import load_golden, run_eval, gate
from src.llm_client import FakeLLMClient
from src.rag_seed import RagSeed, broken_seed

GROUNDED = '{"grounded": true, "why": "ok"}'
ANSWER = "Answer from the docs [policy.md]."


def _llm(n=40):
    # alternate: answer, judge-verdict, answer, judge-verdict...
    return FakeLLMClient(scripted=[ANSWER, GROUNDED] * n)


def test_healthy_pipeline_passes_gate():
    golden = load_golden()
    rag = RagSeed(FakeLLMClient(scripted=[ANSWER] * 20))
    judge = FakeLLMClient(scripted=[GROUNDED] * 20)
    report = run_eval(rag, judge, golden)
    assert report.recall >= 0.8
    assert gate(report) is True


def test_broken_chunking_fails_gate():
    golden = load_golden()
    rag = broken_seed(FakeLLMClient(scripted=[ANSWER] * 20))
    judge = FakeLLMClient(scripted=[GROUNDED] * 20)
    report = run_eval(rag, judge, golden)
    assert report.recall < 0.8          # 120-char chunks crater recall
    assert gate(report) is False


def test_out_of_scope_rows_are_skipped_not_scored():
    golden = [c for c in load_golden() if c["kind"] == "out-of-scope"]
    rag = RagSeed(FakeLLMClient(scripted=[ANSWER] * 5))
    judge = FakeLLMClient(scripted=[GROUNDED] * 5)
    report = run_eval(rag, judge, golden)
    assert report.rows[0][1] == "—"
