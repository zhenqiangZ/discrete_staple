import logging
from typing import Optional

import numpy as np

from llm_judge.client import LLMClient, create_default_clients
from llm_judge.record import JudgeRecord, RecordWriter
from llm_judge.prompts import (
    COND_DESCRIPTIONS,
    PHASE1_WITH_DATA_SYSTEM, PHASE1_WITH_DATA_USER,
    format_variants_with_staples,
)
from format_prompt_data import (
    AA_PROPERTIES_TABLE,
    format_parent_alignment,
    format_parent_properties,
)
from compute_properties import compute_all

logger = logging.getLogger(__name__)


def _parse_scores_from_json(data, key: str = "score") -> np.ndarray:
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
        else:
            return np.full(1, 5.0)

    scores = []
    for item in data:
        if isinstance(item, dict):
            scores.append(float(item.get(key, item.get("final_score", 5.0))))
        else:
            try:
                scores.append(float(item))
            except (ValueError, TypeError):
                scores.append(5.0)
    return np.array(scores)


class LLMGroupEvaluator:
    """Full pipeline (sync, pure serial):
       A scores → B scores → A reviews B → B reviews A → C final.
    """

    def __init__(self, client_a: Optional[LLMClient], client_b: Optional[LLMClient],
                 client_c: Optional[LLMClient],
                 record_dir: str = "", epoch: int = 0, global_step: int = 0):
        self.client_a = client_a
        self.client_b = client_b
        self.client_c = client_c
        self.recorder = RecordWriter(record_dir) if record_dir else None
        self.epoch = epoch
        self.global_step = global_step

    def set_step(self, epoch: int, global_step: int):
        self.epoch = epoch
        self.global_step = global_step

    def evaluate_group(self, parent_seq: str, cond: int, candidates: list,
                       extra_data: Optional[dict] = None) -> np.ndarray:
        scores = np.full(len(candidates), np.nan)
        failed = False
        eval_a = eval_b = None
        review_ab = review_ba = None
        final = None

        try:
            scores, eval_a, eval_b, review_ab, review_ba, final = \
                self._evaluate_group_impl(parent_seq, cond, candidates, extra_data)
        except Exception as e:
            failed = True
            logger.warning(f"LLM judge failed, no reward for this group: {e}")

        # Skip recording failed groups so NaN never reaches the JSONL log.
        if self.recorder and not failed:
            model_a = self.client_a.model if self.client_a else ""
            model_b = self.client_b.model if self.client_b else ""
            model_c = self.client_c.model if self.client_c else ""
            record = JudgeRecord(
                epoch=self.epoch, global_step=self.global_step,
                parent_seq=parent_seq, cond=cond,
                cond_desc=COND_DESCRIPTIONS.get(cond, ""),
                candidates=candidates,
                eval_a=eval_a, eval_b=eval_b,
                review_ab=review_ab, review_ba=review_ba,
                final_raw=final,
                final_scores=scores.tolist(),
                llm_a_model=model_a, llm_b_model=model_b, llm_c_model=model_c,
                score_scale="raw",
            )
            self.recorder.write(record)

        return scores

    def _evaluate_group_impl(self, parent_seq, cond, candidates, extra_data=None):
        K = len(candidates)

        # Phase 1: sequential A → B
        if extra_data:
            # Use data-aware prompts with parent gradients + properties
            parent_props = compute_all(parent_seq)
            aa_table = AA_PROPERTIES_TABLE
            align_table = format_parent_alignment(
                parent_seq,
                extra_data.get('amp_alignments', []),
                extra_data.get('tox_alignments', []),
            )
            props_line = format_parent_properties(parent_props)
            vars_table = format_variants_with_staples(candidates)

            sys_prompt = PHASE1_WITH_DATA_SYSTEM
            user_prompt = PHASE1_WITH_DATA_USER.format(
                parent_seq=parent_seq,
                cond_desc=COND_DESCRIPTIONS.get(cond, f"unknown({cond})"),
                aa_properties=aa_table,
                parent_properties=props_line,
                parent_alignment=align_table,
                variants_table=vars_table,
            )
            eval_a = self._safe_custom_prompt(self.client_a, sys_prompt, user_prompt)
            eval_b = self._safe_custom_prompt(self.client_b, sys_prompt, user_prompt)
        else:
            eval_a = self._safe_score(self.client_a, parent_seq, cond, candidates)
            eval_b = self._safe_score(self.client_b, parent_seq, cond, candidates)

        if eval_a is None and eval_b is None:
            raise RuntimeError("both phase-1 evaluators failed")

        score_a = _parse_scores_from_json(eval_a) if eval_a else np.full(K, 5.0)
        score_b = _parse_scores_from_json(eval_b) if eval_b else np.full(K, 5.0)
        eval_a = eval_a or [{"idx": i, "score": float(score_a[i]), "reasoning": ""} for i in range(K)]
        eval_b = eval_b or [{"idx": i, "score": float(score_b[i]), "reasoning": ""} for i in range(K)]

        # Phase 2: sequential A reviews B → B reviews A
        review_ab = self._safe_review(self.client_a, parent_seq, cond, candidates, eval_b, eval_a)
        review_ba = self._safe_review(self.client_b, parent_seq, cond, candidates, eval_a, eval_b)
        review_ab = review_ab or {"disagreements": [], "overall_verdict": "no review available"}
        review_ba = review_ba or {"disagreements": [], "overall_verdict": "no review available"}

        # Phase 3: final judge
        final = None
        if self.client_c is not None:
            final = self._safe_final(self.client_c, parent_seq, cond, candidates,
                                     eval_a, eval_b, review_ab, review_ba)
            final_scores = _parse_scores_from_json(final, "final_score") if final else None
        else:
            final_scores = None

        if final_scores is None or len(final_scores) != K:
            final_scores = (score_a + score_b) / 2.0

        # Raw 0-10 scores; GRPO normalises within each group itself.
        return np.asarray(final_scores, dtype=np.float64), eval_a, eval_b, review_ab, review_ba, final

    def _safe_score(self, client, parent_seq, cond, candidates) -> Optional[list]:
        if client is None:
            return None
        try:
            return client.score_with_reasoning(parent_seq, cond, candidates)
        except Exception as e:
            logger.warning(f"Score call failed for {client.provider}/{client.model}: {e}")
            return None

    def _safe_custom_prompt(self, client, system: str, user: str) -> Optional[dict]:
        if client is None:
            return None
        try:
            return client.score_with_custom_prompt(system, user)
        except Exception as e:
            logger.warning(f"Custom prompt call failed for {client.provider}/{client.model}: {e}")
            return None

    def _safe_review(self, client, parent_seq, cond, candidates, other, self_eval) -> Optional[dict]:
        if client is None:
            return None
        try:
            return client.review(parent_seq, cond, candidates, other, self_eval)
        except Exception as e:
            logger.warning(f"Review call failed for {client.provider}/{client.model}: {e}")
            return None

    def _safe_final(self, client, parent_seq, cond, candidates, ea, eb, rab, rba) -> Optional[list]:
        try:
            return client.final_judge(parent_seq, cond, candidates, ea, eb, rab, rba)
        except Exception as e:
            logger.warning(f"Final judge call failed: {e}")
            return None


class LiteLLMGroupEvaluator:
    """Lite version (sync, pure serial):
       A scores → B scores → C final.  3 calls instead of 5.
    """

    def __init__(self, client_a: Optional[LLMClient], client_b: Optional[LLMClient],
                 client_c: Optional[LLMClient],
                 record_dir: str = "", epoch: int = 0, global_step: int = 0):
        self.client_a = client_a
        self.client_b = client_b
        self.client_c = client_c
        self.recorder = RecordWriter(record_dir) if record_dir else None
        self.epoch = epoch
        self.global_step = global_step

    def set_step(self, epoch: int, global_step: int):
        self.epoch = epoch
        self.global_step = global_step

    def evaluate_group(self, parent_seq: str, cond: int, candidates: list) -> np.ndarray:
        scores = np.full(len(candidates), np.nan)
        failed = False
        eval_a = eval_b = None
        final = None

        try:
            scores, eval_a, eval_b, final = self._evaluate_group_impl(parent_seq, cond, candidates)
        except Exception as e:
            failed = True
            logger.warning(f"Lite LLM judge failed, no reward for this group: {e}")

        if self.recorder and not failed:
            model_a = self.client_a.model if self.client_a else ""
            model_b = self.client_b.model if self.client_b else ""
            model_c = self.client_c.model if self.client_c else ""
            record = JudgeRecord(
                epoch=self.epoch, global_step=self.global_step,
                parent_seq=parent_seq, cond=cond,
                cond_desc=COND_DESCRIPTIONS.get(cond, ""),
                candidates=candidates,
                eval_a=eval_a, eval_b=eval_b,
                review_ab=None, review_ba=None,
                final_raw=final,
                final_scores=scores.tolist(),
                llm_a_model=model_a, llm_b_model=model_b, llm_c_model=model_c,
                score_scale="raw",
            )
            self.recorder.write(record)

        return scores

    def _evaluate_group_impl(self, parent_seq, cond, candidates):
        K = len(candidates)

        eval_a = self._safe_score(self.client_a, parent_seq, cond, candidates)
        eval_b = self._safe_score(self.client_b, parent_seq, cond, candidates)

        if eval_a is None and eval_b is None:
            raise RuntimeError("both phase-1 evaluators failed")

        score_a = _parse_scores_from_json(eval_a) if eval_a else np.full(K, 5.0)
        score_b = _parse_scores_from_json(eval_b) if eval_b else np.full(K, 5.0)

        eval_a = eval_a or [{"idx": i, "score": float(score_a[i]), "reasoning": ""} for i in range(K)]
        eval_b = eval_b or [{"idx": i, "score": float(score_b[i]), "reasoning": ""} for i in range(K)]

        final = None
        if self.client_c is not None:
            final = self._safe_final_lite(self.client_c, parent_seq, cond, candidates, eval_a, eval_b)
            final_scores = _parse_scores_from_json(final, "final_score") if final else None
        else:
            final_scores = None

        if final_scores is None or len(final_scores) != K:
            final_scores = (score_a + score_b) / 2.0

        return np.asarray(final_scores, dtype=np.float64), eval_a, eval_b, final

    def _safe_score(self, client, parent_seq, cond, candidates) -> Optional[list]:
        if client is None:
            return None
        try:
            return client.score_with_reasoning(parent_seq, cond, candidates)
        except Exception as e:
            logger.warning(f"Lite score call failed: {e}")
            return None

    def _safe_final_lite(self, client, parent_seq, cond, candidates, ea, eb) -> Optional[list]:
        try:
            return client.final_judge_lite(parent_seq, cond, candidates, ea, eb)
        except Exception as e:
            logger.warning(f"Lite final judge failed: {e}")
            return None


def create_evaluator(
    client_a=None, client_b=None, client_c=None,
    lite_mode: bool = False,
    record_dir: str = "", epoch: int = 0, global_step: int = 0,
):
    """Factory: create full or lite evaluator."""
    clients = create_default_clients()
    client_a = client_a or clients['a']
    client_b = client_b or clients['b']
    client_c = client_c or clients['c']

    if lite_mode:
        return LiteLLMGroupEvaluator(client_a, client_b, client_c,
                                     record_dir, epoch, global_step)
    return LLMGroupEvaluator(client_a, client_b, client_c,
                             record_dir, epoch, global_step)
