import logging
from typing import Optional

import torch

from llm_judge.client import LLMClient
from data.tokenizer import encode, pad

logger = logging.getLogger(__name__)


class ProxyRewardClient(LLMClient):
    """Implements LLMClient interface using a trained RewardProxyModel.

    This allows drop-in replacement of LLM API calls with local
    model inference for scoring.
    """

    def __init__(self, nn_model, device: torch.device,
                 api_key: str = '', model_name: str = 'reward_proxy',
                 provider: str = 'proxy', base_url: str = ''):
        super().__init__(api_key, model_name, provider, base_url)
        self._nn_model = nn_model
        self.device = device
        self._nn_model.eval()
        self._max_len = 50

    def _call_api(self, system: str, user: str, temperature: float = 0.3) -> str:
        raise NotImplementedError('ProxyRewardClient does not support raw text API calls')

    @torch.no_grad()
    def score_with_reasoning(self, parent_seq: str, cond: int,
                              candidates: list) -> list:
        parent_tokens = encode(parent_seq)
        parent_padded = torch.tensor(
            [pad(parent_tokens, self._max_len)], dtype=torch.long, device=self.device
        )
        parent_batch = parent_padded.expand(len(candidates), -1)

        cand_tokens_list = [encode(c) for c in candidates]
        cand_padded = torch.stack([
            torch.tensor(pad(t, self._max_len), dtype=torch.long, device=self.device)
            for t in cand_tokens_list
        ])

        cond_tensor = torch.full((len(candidates),), cond, dtype=torch.long, device=self.device)

        scores = self._nn_model(parent_batch, cand_padded, cond_tensor)
        scores = scores.cpu().tolist()

        results = []
        for i, s in enumerate(scores):
            clipped = max(0.0, min(10.0, s))
            results.append({
                'idx': i,
                'score': clipped,
                'reasoning': '[proxy]',
            })
        return results

    def review(self, parent_seq: str, cond: int, candidates: list,
               other_eval: list, self_eval: list) -> dict:
        return {'disagreements': [], 'overall_verdict': '[proxy] no cross-review available'}

    def final_judge(self, parent_seq: str, cond: int, candidates: list,
                    eval_a: list, eval_b: list,
                    review_ab: dict, review_ba: dict) -> list:
        return self.score_with_reasoning(parent_seq, cond, candidates)

    def final_judge_lite(self, parent_seq: str, cond: int, candidates: list,
                          eval_a: list, eval_b: list) -> list:
        return self.score_with_reasoning(parent_seq, cond, candidates)


def load_proxy_client(checkpoint_path: str,
                      device: torch.device) -> Optional[ProxyRewardClient]:
    """Load a trained RewardProxyModel from checkpoint and wrap in ProxyRewardClient."""
    try:
        import sys
        sys.path.insert(0, '.')
        from train_reward_proxy import RewardProxyModel

        nn_model = RewardProxyModel()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        nn_model.load_state_dict(sd)
        nn_model.to(device)
        nn_model.eval()
        logger.info(f'Loaded reward proxy from {checkpoint_path}')
        return ProxyRewardClient(nn_model, device)
    except Exception as e:
        logger.warning(f'Failed to load reward proxy: {e}')
        return None
