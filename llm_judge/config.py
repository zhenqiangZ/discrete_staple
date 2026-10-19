import os
import sys

# 确保能找到项目根目录的 config.py
_llm_judge_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_llm_judge_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 共享配置统一从项目 config.py 导入
from config import (
    LLM_A_PROVIDER, LLM_A_MODEL, LLM_A_BASE_URL, LLM_A_API_KEY,
    LLM_B_PROVIDER, LLM_B_MODEL, LLM_B_BASE_URL, LLM_B_API_KEY,
    LLM_C_PROVIDER, LLM_C_MODEL, LLM_C_BASE_URL, LLM_C_API_KEY,
)

# 内部请求控制
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2.0
