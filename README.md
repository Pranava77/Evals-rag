# Session 4 — Evaluation Harness (Take-Home Practice)

**The same exercises your instructor demoed** — make the failing tests pass.

```bash
pip install -r requirements.txt
python -m pytest -q     # RED to start — expected
```

## Your 3 TODOs

1. **`src/metrics.py` → `recall_at_k`** — set overlap / expected size (empty → 1.0).
2. **`src/metrics.py` → `faithfulness`** — build the judge prompt, call the judge,
   parse the JSON verdict.
3. **`src/eval_runner.py` → `gate`** — PASS only if both means clear their bars.

Then run `python -m src.eval_runner` and `--break-chunking` to see your own gate go
green and red. Appendix slides 39–41 are your reference *after* trying.

**Stretch:** add an 11th golden case from YOUR domain and see if the seed passes it.

**Standalone:** the RAG seed is bundled — nothing from other sessions needed.
