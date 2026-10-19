"""Lightweight package facade.

Submodules are imported lazily so that ``import llm_judge.record`` does not
drag in numpy / HTTP clients / the full evaluator stack.  Public names stay
available as ``from llm_judge import LLMGroupEvaluator``.
"""

_EXPORTS = {
    'LLMGroupEvaluator': ('llm_judge.evaluator', 'LLMGroupEvaluator'),
    'LiteLLMGroupEvaluator': ('llm_judge.evaluator', 'LiteLLMGroupEvaluator'),
    'LLMClient': ('llm_judge.client', 'LLMClient'),
    'JudgeRecord': ('llm_judge.record', 'JudgeRecord'),
    'RecordWriter': ('llm_judge.record', 'RecordWriter'),
    'RecordStore': ('llm_judge.record_store', 'RecordStore'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
    import importlib
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
