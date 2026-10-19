"""Pure-Python unit tests for judge-record score-scale handling.

Verifies that legacy per-group z-score records are converted to the
absolute 0-10 scale (or dropped when unrecoverable) so they are never
mixed with scores produced by the current pipeline.

Run:
    python tests/test_record_store.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_judge.record import JudgeRecord
from llm_judge.record_store import RecordStore
from llm_judge.score_scale import (
    SCALE_RAW, SCALE_RECOVERED, recover_raw_scores, resolve_scores,
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def _write_jsonl(directory, filename, records):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _legacy_record(cands, zscores, phase3_scores=None, phase1_a=None, phase1_b=None):
    rec = {
        "parent_seq": "ACDEF", "cond": 2, "cond_desc": "i,i+4",
        "candidates": [{"idx": i, "seq": s} for i, s in enumerate(cands)],
        "final_scores": zscores,
        "phase1": {
            "llm_a": {"model": "a", "results": {"scores": phase1_a or []}},
            "llm_b": {"model": "b", "results": {"scores": phase1_b or []}},
        },
        "phase2": {"review_ab": {}, "review_ba": {}},
        "phase3": {"llm_c": {"model": "c", "results": {"final_scores": phase3_scores or []}}},
    }
    return rec


def main():
    tmp = tempfile.mkdtemp(prefix="record_store_test_")

    print("[1] JudgeRecord carries score_scale")
    rec = JudgeRecord(
        epoch=1, global_step=2, parent_seq="ACDEF", cond=2, cond_desc="i,i+4",
        candidates=["ACDEF", "S5CDES5"], eval_a=None, eval_b=None,
        review_ab=None, review_ba=None, final_raw=None,
        final_scores=[7.5, 8.0], score_scale=SCALE_RAW,
    )
    check("raw scale stored", rec.to_dict()["score_scale"] == SCALE_RAW)
    check("scores kept as floats", rec.to_dict()["final_scores"] == [7.5, 8.0])

    print("[2] legacy z-scores are recovered from the phase-3 payload")
    cands = ["ACDEF", "S5CDES5", "ACDGS5"]
    legacy = _legacy_record(
        cands, zscores=[-1.2, 0.0, 1.2],
        phase3_scores=[{"idx": 0, "final_score": 4},
                       {"idx": 1, "final_score": 6},
                       {"idx": 2, "final_score": 9}],
    )
    recovered = recover_raw_scores(legacy)
    check("recovered from phase 3", recovered == [4.0, 6.0, 9.0])
    scores, scale = resolve_scores(legacy)
    check("scale marked recovered", scale == SCALE_RECOVERED)
    check("scores are absolute", scores == [4.0, 6.0, 9.0])

    _write_jsonl(tmp, "judge_records_20240101_000000.jsonl", [legacy])
    store = RecordStore(tmp)
    check("all entries indexed", store.num_records() == 3)
    check("lookup_raw returns recovered value",
          store.lookup_raw("ACDEF", "ACDGS5", 2) == 9.0)
    check("provenance recorded",
          store.lookup_entry("ACDEF", "ACDGS5", 2)["score_source"] == SCALE_RECOVERED)
    check("summarize counts recovery",
          store.summarize()["recovered_records"] == 1)

    print("[3] legacy without phase-3 falls back to the phase-1 mean")
    legacy2 = _legacy_record(
        cands, zscores=[-1.0, 0.0, 1.0],
        phase1_a=[{"idx": 0, "score": 3}, {"idx": 1, "score": 6}, {"idx": 2, "score": 8}],
        phase1_b=[{"idx": 0, "score": 5}, {"idx": 1, "score": 6}, {"idx": 2, "score": 9}],
    )
    check("mean of A and B", recover_raw_scores(legacy2) == [4.0, 6.0, 8.5])

    print("[4] unrecoverable legacy records are dropped")
    bare = {
        "parent_seq": "ACDEF", "cond": 2,
        "candidates": [{"idx": 0, "seq": "ACDEF"}, {"idx": 1, "seq": "S5CDES5"}],
        "final_scores": [-0.71, 0.71],
    }
    check("no scores recovered", recover_raw_scores(bare) is None)
    check("resolve returns None", resolve_scores(bare) == (None, None))

    tmp2 = tempfile.mkdtemp(prefix="record_store_test_bare_")
    _write_jsonl(tmp2, "judge_records_20240201_000000.jsonl", [bare])
    store2 = RecordStore(tmp2)
    check("nothing indexed", store2.num_records() == 0)
    check("counted as unusable", store2.summarize()["unusable_records"] == 1)

    print("[5] raw records pass through unchanged")
    tmp3 = tempfile.mkdtemp(prefix="record_store_test_raw_")
    raw_rec = {
        "parent_seq": "ACDEF", "cond": 3,
        "candidates": [{"idx": 0, "seq": "ACDEF"}, {"idx": 1, "seq": "S5CDEFR8"}],
        "final_scores": [6.0, 8.5],
        "score_scale": SCALE_RAW,
    }
    _write_jsonl(tmp3, "judge_records_20250101_000000.jsonl", [raw_rec])
    store3 = RecordStore(tmp3)
    check("raw hit returned", store3.lookup_raw("ACDEF", "S5CDEFR8", 3) == 8.5)
    check("no recovery needed", store3.summarize()["recovered_records"] == 0)

    print("[6] out-of-range values are clamped to 0-10")
    clamp_rec = _legacy_record(
        ["ACDEF"], zscores=[2.5],
        phase3_scores=[{"idx": 0, "final_score": 14}],
    )
    check("clamped to 10", recover_raw_scores(clamp_rec) == [10.0])

    print("[7] put() defaults to raw scale")
    store3.put("ACDEF", 3, "ACDGS5", 9.0, source="llm_api")
    check("fresh put is raw-scale", store3.lookup_raw("ACDEF", "ACDGS5", 3) == 9.0)
    check("source recorded",
          store3.lookup_entry("ACDEF", "ACDGS5", 3)["source"] == "llm_api")

    print("\nAll record-store tests passed.")


if __name__ == "__main__":
    main()
