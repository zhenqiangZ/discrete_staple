import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PARENT_DATA_DIR = DATA_DIR


def _load_env_file(path: str) -> None:
    """Minimal .env loader (no external dependency).

    Existing environment variables take precedence, so CI/shell settings
    always win over the local file.
    """
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(BASE_DIR, '.env'))

# Vocab
PAD_TOKEN = 0
MASK_TOKEN = 23
VOCAB_SIZE = 24
MAX_LEN = 50

# Model
D_MODEL = 512
NUM_LAYERS = 12
NUM_HEADS = 8
FFN_DIM = 1024
DROPOUT = 0.1
NUM_CLASS = 5

# Training
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-4
WARMUP_STEPS = 1000
MASK_RATE_MAX = 0.6

# Sampling
NUM_SAMPLE_STEPS = 10
TEMPERATURE = 0.5

# Paths
PAIRED_CSV = os.path.join(PARENT_DATA_DIR, 'paired_data.csv')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# RL
RL_EPOCHS = 20
RL_BATCH_SIZE = 16
RL_STEPS_PER_EPOCH = 20
RL_LR = 5e-6
RL_ADV_COEFF = 0.2
RL_K = 8
RL_MIX_GT = 0.1
RL_SAMPLE_STEPS = 10
RL_SAMPLE_TEMP = 0.8
RL_MASK_RATE_MAX = 0.6

# GRPO
GRPO_CLIP_EPS = 0.2
GRPO_KL_BETA = 0.05
GRPO_DESIGN_TOP_K = 10
GRPO_MASK_RATE = 0.1

# Data selection
SUBSAMPLE_PER_EPOCH = 2000     # AMPs sampled per epoch for RL (125 batches at B=16)
MIN_SEQ_LEN = 10               # Minimum parent peptide length (tokens)
MAX_SEQ_LEN_DESIGN = 20        # Maximum parent peptide length (tokens)

# LLM Judge
W_LLM = 0.3                    # LLM reward weight (0 = disabled)
RECORD_DIR = "logs/llm_judge"  # LLM evaluation record output dir
JUDGE_EVERY_N_STEPS = 5        # Run LLM judge every N steps
JUDGE_SUBSAMPLE_GROUPS = 4     # Number of groups evaluated per step
JUDGE_LITE_MODE = False        # Skip cross-review (3 calls vs 5 per group)
LLM_A_PROVIDER = "openai"
LLM_B_PROVIDER = "openai"
LLM_C_PROVIDER = "openai"

# LLM_A_MODEL = "deepseek-v4-flash"
# LLM_B_MODEL = "deepseek-v4-flash"
# LLM_C_MODEL = "deepseek-v4-flash"

LLM_C_MODEL = "deepseek-v4-flash"
LLM_B_MODEL = "deepseek-v4-pro"
LLM_A_MODEL = "gpt-5.1"

LLM_C_BASE_URL = "https://api.deepseek.com"
LLM_B_BASE_URL = "https://opencode.ai/zen/go/v1/"
LLM_A_BASE_URL = "https://mydamoxing.cn/v1"

# LLM API Keys (loaded from environment / .env, never hard-code them here)
LLM_A_API_KEY = os.environ.get("LLM_A_API_KEY", "")
LLM_B_API_KEY = os.environ.get("LLM_B_API_KEY", "")
LLM_C_API_KEY = os.environ.get("LLM_C_API_KEY", "")

# Reward Proxy
PROXY_MODEL_PATH = "checkpoints/reward_proxy/best.pt"  # trained proxy checkpoint
PROXY_ENABLED = False                                  # use proxy instead of LLM API
REVIEW_THRESHOLD = 5                                   # proxy entries needed to trigger LLM review
REVIEW_EVERY_N_EPOCHS = 1                              # run review every N epochs

