import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

from llm_judge.config import (
    LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_RETRY_DELAY,
    LLM_A_API_KEY, LLM_A_MODEL, LLM_A_PROVIDER,
    LLM_B_API_KEY, LLM_B_MODEL, LLM_B_PROVIDER,
    LLM_C_API_KEY, LLM_C_MODEL, LLM_C_PROVIDER,
    LLM_A_BASE_URL, LLM_B_BASE_URL, LLM_C_BASE_URL,
)
from llm_judge.prompts import (
    PHASE1_SYSTEM, PHASE1_USER,
    PHASE2_REVIEW_SYSTEM, PHASE2_REVIEW_USER,
    PHASE3_SYSTEM, PHASE3_USER,
    PHASE3_LITE_SYSTEM, PHASE3_LITE_USER,
    COND_DESCRIPTIONS, format_variants_table, format_evaluation_json,
)

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base for synchronous LLM API clients."""

    def __init__(self, api_key: str, model: str, provider: str, base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.base_url = base_url

    @abstractmethod
    def _call_api(self, system: str, user: str, temperature: float = 0.3) -> str:
        ...

    def _call_with_retry(self, system: str, user: str, temperature: float = 0.3) -> str:
        last_err = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                return self._call_api(system, user, temperature)
            except Exception as e:
                last_err = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}")
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(LLM_RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_err}")

    def _parse_json(self, text: str):
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        return json.loads(text)

    def _call_json_obj(self, system: str, user: str, temperature: float = 0.3) -> dict:
        return self._parse_json(self._call_with_retry(system, user, temperature))

    def score_with_custom_prompt(self, system: str, user: str, temperature: float = 0.3) -> dict:
        """Score with an arbitrary system/user prompt pair."""
        return self._call_json_obj(system, user, temperature)

    # ── Public API ──────────────────────────────────────

    def score_with_reasoning(self, parent_seq: str, cond: int, candidates: list) -> dict:
        cond_desc = COND_DESCRIPTIONS.get(cond, f"unknown({cond})")
        table = format_variants_table(candidates)
        user = PHASE1_USER.format(parent_seq=parent_seq, cond_desc=cond_desc, variants_table=table)
        return self._call_json_obj(PHASE1_SYSTEM, user)

    def review(self, parent_seq: str, cond: int, candidates: list,
               other_eval: list, self_eval: list) -> dict:
        cond_desc = COND_DESCRIPTIONS.get(cond, f"unknown({cond})")
        table = format_variants_table(candidates)
        user = PHASE2_REVIEW_USER.format(
            parent_seq=parent_seq, cond_desc=cond_desc,
            variants_table=table,
            other_evaluation=format_evaluation_json(other_eval),
            self_evaluation=format_evaluation_json(self_eval),
        )
        return self._call_json_obj(PHASE2_REVIEW_SYSTEM, user)

    def final_judge(self, parent_seq: str, cond: int, candidates: list,
                    eval_a: list, eval_b: list,
                    review_ab: dict, review_ba: dict) -> dict:
        cond_desc = COND_DESCRIPTIONS.get(cond, f"unknown({cond})")
        table = format_variants_table(candidates)
        user = PHASE3_USER.format(
            parent_seq=parent_seq, cond_desc=cond_desc,
            variants_table=table,
            eval_A=format_evaluation_json(eval_a),
            eval_B=format_evaluation_json(eval_b),
            review_AB=format_evaluation_json(review_ab),
            review_BA=format_evaluation_json(review_ba),
        )
        return self._call_json_obj(PHASE3_SYSTEM, user)

    def final_judge_lite(self, parent_seq: str, cond: int, candidates: list,
                          eval_a: list, eval_b: list) -> dict:
        cond_desc = COND_DESCRIPTIONS.get(cond, f"unknown({cond})")
        table = format_variants_table(candidates)
        user = PHASE3_LITE_USER.format(
            parent_seq=parent_seq, cond_desc=cond_desc,
            variants_table=table,
            eval_A=format_evaluation_json(eval_a),
            eval_B=format_evaluation_json(eval_b),
        )
        return self._call_json_obj(PHASE3_LITE_SYSTEM, user)


class OpenAIClient(LLMClient):
    """OpenAI API client using synchronous OpenAI client."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = ""):
        super().__init__(api_key, model, "openai", base_url)

    def _call_api(self, system: str, user: str, temperature: float = 0.3) -> str:
        from openai import OpenAI
        kwargs = {"api_key": self.api_key, "timeout": LLM_TIMEOUT}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Anthropic API client using synchronous Anthropic client."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", base_url: str = ""):
        super().__init__(api_key, model, "anthropic", base_url)

    def _call_api(self, system: str, user: str, temperature: float = 0.3) -> str:
        from anthropic import Anthropic
        kwargs = {"api_key": self.api_key, "timeout": LLM_TIMEOUT}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = Anthropic(**kwargs)
        resp = client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=4096,
        )
        return resp.content[0].text if resp.content else ""


def create_client(provider: str, model: str, api_key: str, base_url: str = "") -> Optional[LLMClient]:
    """Factory: create synchronous LLM client by provider name."""
    if not api_key:
        return None
    if provider == "openai":
        return OpenAIClient(api_key, model, base_url)
    elif provider == "anthropic":
        return AnthropicClient(api_key, model, base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_default_clients() -> dict:
    """Create clients from config defaults. Returns dict of {'a', 'b', 'c'} or None entries."""
    return {
        'a': create_client(LLM_A_PROVIDER, LLM_A_MODEL, LLM_A_API_KEY, LLM_A_BASE_URL),
        'b': create_client(LLM_B_PROVIDER, LLM_B_MODEL, LLM_B_API_KEY, LLM_B_BASE_URL),
        'c': create_client(LLM_C_PROVIDER, LLM_C_MODEL, LLM_C_API_KEY, LLM_C_BASE_URL),
    }
