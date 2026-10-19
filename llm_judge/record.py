import json
import os
from datetime import datetime
from typing import Optional


class JudgeRecord:
    """Container for a full LLM judge pipeline evaluation on one group."""

    def __init__(self, epoch: int, global_step: int,
                 parent_seq: str, cond: int, cond_desc: str, candidates: list,
                 eval_a: Optional[list], eval_b: Optional[list],
                 review_ab: Optional[dict], review_ba: Optional[dict],
                 final_raw: Optional[list], final_scores: list,
                 llm_a_model: str = "", llm_b_model: str = "",
                 llm_c_model: str = "", score_scale: str = "raw"):
        self.timestamp = datetime.now().isoformat()
        self.epoch = epoch
        self.global_step = global_step
        self.parent_seq = parent_seq
        self.cond = cond
        self.cond_desc = cond_desc
        # "raw" = absolute 0-10 scores (current pipeline).
        # "zscore" = legacy records written before the scale fix; consumers
        # must not mix the two when training a reward proxy.
        self.score_scale = score_scale
        self.candidates = [{"idx": i, "seq": s} for i, s in enumerate(candidates)]
        self.phase1 = {
            "llm_a": {"model": llm_a_model, "results": eval_a or []},
            "llm_b": {"model": llm_b_model, "results": eval_b or []},
        }
        self.phase2 = {
            "review_ab": review_ab or {},
            "review_ba": review_ba or {},
        }
        self.phase3 = {
            "llm_c": {"model": llm_c_model, "results": final_raw or []},
        }
        self.final_scores = [float(s) for s in final_scores]

    def to_dict(self) -> dict:
        return self.__dict__


class RecordWriter:
    """Append judge records to a JSON Lines file."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"judge_records_{ts}.jsonl")
        self._file = open(self.path, "a", encoding="utf-8")
        print(f"  Judge records -> {self.path}")

    def write(self, record: JudgeRecord):
        self._file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
