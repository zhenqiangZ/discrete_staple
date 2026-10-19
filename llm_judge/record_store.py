import json
import os
import logging
from glob import glob
from typing import Optional

from llm_judge.score_scale import (
    SCALE_RAW, SCALE_RECOVERED, resolve_scores,
)

logger = logging.getLogger(__name__)


class RecordStore:
    """In-memory index + JSON Lines persistence for judge records.

    Supports fast lookup by (parent, candidate, cond) and group-level
    queries with source tracking ("llm_api" vs "proxy").

    Every stored score is normalised to the absolute 0-10 scale.  Legacy
    z-score records are recovered from their embedded phase-1/phase-3
    payloads when possible; unrecoverable ones are dropped.
    """

    def __init__(self, directory: str):
        self.directory = directory
        # (parent, candidate, cond) -> {"score": float, "source": str, ...}
        self._by_key: dict[tuple[str, str, int], dict] = {}
        # (parent, cond) -> set of candidate strings
        self._groups: dict[tuple[str, int], set[str]] = {}
        # Flat log for sequential replay
        self._records: list[dict] = []
        # Migration bookkeeping
        self.n_records_seen = 0
        self.n_recovered_records = 0
        self.n_unusable_records = 0

        self._load_existing()

    def _load_existing(self):
        """Load all judge_records_*.jsonl files into memory on startup."""
        pattern = os.path.join(self.directory, 'judge_records_*.jsonl')
        files = sorted(glob(pattern))
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f'skipping malformed record in {filepath}')
                        continue
                    self._ingest_record(record)
        if self.n_records_seen:
            logger.info(f'RecordStore loaded {self.n_records_seen} records '
                        f'from {len(files)} file(s)')

    def _ingest_record(self, record: dict):
        self.n_records_seen += 1
        self._records.append(record)

        scores, scale = resolve_scores(record)
        if scores is None:
            self.n_unusable_records += 1
            return
        if scale == SCALE_RECOVERED:
            self.n_recovered_records += 1

        parent = record.get('parent_seq', '')
        cond = record.get('cond', 0)
        candidates = record.get('candidates', [])
        source = record.get('source', 'llm_api')

        group_key = (parent, cond)
        if group_key not in self._groups:
            self._groups[group_key] = set()

        for cand_info, score in zip(candidates, scores):
            cand_seq = cand_info.get('seq', '') if isinstance(cand_info, dict) else str(cand_info)
            if not cand_seq:
                continue
            key = (parent, cand_seq, cond)
            self._by_key[key] = {
                'score': float(score),
                'source': source,
                'score_scale': SCALE_RAW,
                'score_source': scale,
                'idx': cand_info.get('idx', -1) if isinstance(cand_info, dict) else -1,
            }
            self._groups[group_key].add(cand_seq)

    def lookup(self, parent: str, candidate: str, cond: int) -> Optional[float]:
        """Return cached score, or None if not found."""
        entry = self._by_key.get((parent, candidate, cond))
        return entry['score'] if entry else None

    def lookup_raw(self, parent: str, candidate: str, cond: int) -> Optional[float]:
        """Return cached score only when it is on the absolute 0-10 scale.

        Legacy z-score entries are ignored so a cached value can never be
        mixed with a freshly computed absolute reward.
        """
        entry = self._by_key.get((parent, candidate, cond))
        if entry and entry.get('score_scale') == 'raw':
            return entry['score']
        return None

    def lookup_entry(self, parent: str, candidate: str, cond: int) -> Optional[dict]:
        """Return full entry dict {score, source, idx}, or None."""
        return self._by_key.get((parent, candidate, cond))

    def get_group_candidates(self, parent: str, cond: int) -> list[str]:
        """Return list of candidate sequences for a (parent, cond) group."""
        group_key = (parent, cond)
        return list(self._groups.get(group_key, set()))

    def get_group_entries(self, parent: str, cond: int) -> list[dict]:
        """Return all entries for a (parent, cond) group."""
        entries = []
        for cand_seq in self.get_group_candidates(parent, cond):
            entry = self.lookup_entry(parent, cand_seq, cond)
            if entry:
                entries.append({'seq': cand_seq, **entry})
        return entries

    def put(self, parent: str, cond: int, candidate: str, score: float,
            source: str = 'proxy', score_scale: str = 'raw'):
        """Add or update a single (parent, candidate, cond) entry."""
        key = (parent, candidate, cond)
        self._by_key[key] = {'score': score, 'source': source,
                             'score_scale': score_scale}

        group_key = (parent, cond)
        if group_key not in self._groups:
            self._groups[group_key] = set()
        self._groups[group_key].add(candidate)

    def put_group(self, parent: str, cond: int, candidates: list[str],
                  scores: list[float], source: str = 'proxy'):
        """Add or update all candidates for a group."""
        for cand, score in zip(candidates, scores):
            self.put(parent, cond, cand, score, source)

    def get_proxy_marked_groups(self, threshold: int = 5) -> list[tuple[str, int]]:
        """Return all (parent, cond) groups with >= threshold proxy-marked entries."""
        groups = []
        for (parent, cond), candidates in self._groups.items():
            n_proxy = 0
            for cand in candidates:
                entry = self._by_key.get((parent, cand, cond))
                if entry and entry.get('source') == 'proxy':
                    n_proxy += 1
            if n_proxy >= threshold:
                groups.append((parent, cond))
        return groups

    def num_records(self) -> int:
        return len(self._by_key)

    def num_groups(self) -> int:
        return len(self._groups)

    def summarize(self) -> dict:
        n_llm = sum(1 for e in self._by_key.values() if e.get('source') == 'llm_api')
        n_proxy = sum(1 for e in self._by_key.values() if e.get('source') == 'proxy')
        n_recovered = sum(1 for e in self._by_key.values()
                          if e.get('score_source') == SCALE_RECOVERED)
        return {
            'total_entries': len(self._by_key),
            'total_groups': len(self._groups),
            'llm_api': n_llm,
            'proxy': n_proxy,
            'recovered_entries': n_recovered,
            'recovered_records': self.n_recovered_records,
            'unusable_records': self.n_unusable_records,
        }
