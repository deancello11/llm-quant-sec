from src.common import read_jsonl
from src.generate import run_cell
from src.prompts import PromptRow


def rows(n):
    return [PromptRow(f"d:{i}::direct", "d", f"d:{i}", "direct", "c", "p" * (i + 1), True) for i in range(n)]


def test_resume_skips_done_and_batches_identically(tmp_path):
    calls = []

    def gen(texts):
        calls.append(list(texts))
        return [("resp " + t, 3, False) for t in texts]

    meta = {"model": "m", "precision": "fp16", "seed": 1, "run_id": "r"}
    out = tmp_path / "cell.jsonl"
    assert run_cell(rows(5), out, meta, lambda s: s, gen, 2, len) == 5
    assert [len(c) for c in calls] == [2, 2, 1]
    # simulate an interruption: keep only the first 3 lines, then re-run
    lines = out.read_text().splitlines()[:3]
    out.write_text("\n".join(lines) + "\n")
    calls.clear()
    assert run_cell(rows(5), out, meta, lambda s: s, gen, 2, len) == 2
    got = read_jsonl(out)
    assert len(got) == 5 and len({r["prompt_id"] for r in got}) == 5
    assert run_cell(rows(5), out, meta, lambda s: s, gen, 2, len) == 0


def test_oom_halves_batch(tmp_path):
    state = {"n": 0}

    def gen(texts):
        state["n"] += 1
        if len(texts) > 2:
            raise RuntimeError("CUDA out of memory")
        return [("r", 1, False) for _ in texts]

    meta = {"model": "m", "precision": "fp16", "seed": 1, "run_id": "r"}
    assert run_cell(rows(4), tmp_path / "c.jsonl", meta, lambda s: s, gen, 4, len) == 4
