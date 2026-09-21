# metric_cfg.py
"""指标配置：Judge 模型、RAGAS/DeepEval 指标开关、阈值。"""
import os

# =============================================================================
# Judge LLM（OpenAI 兼容接口，当前用 DeepSeek）
# =============================================================================
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")
JUDGE_EMBED_MODEL = os.getenv("JUDGE_EMBED_MODEL")

# =============================================================================
# 评分阈值
# =============================================================================
DEFAULT_THRESHOLD = 0.5

# =============================================================================
# RAGAS 指标开关
# answer_relevancy 默认关闭：需 embeddings，MiniMax 非完全 OpenAI 兼容
# =============================================================================
RAGAS_METRICS_ENABLED = {
    "faithfulness": False,
    "answer_relevancy": False,
    "context_recall": False,
    "context_precision": False,
}

# =============================================================================
# DeepEval 指标开关
# answer_relevancy 同样因 embeddings 问题默认关闭
# =============================================================================
DEEPEVAL_METRICS_ENABLED = {
    "faithfulness": False,
    "answer_relevancy": True,
    "contextual_recall": False,
    "contextual_precision": False,
    "geval": True,
}


def get_judge_env() -> dict:
    """返回 DeepEval 需要的 OPENAI_* 环境变量字典。"""
    return {
        "OPENAI_API_KEY": JUDGE_API_KEY,
        "OPENAI_API_BASE": JUDGE_BASE_URL,
        "OPENAI_BASE_URL": JUDGE_BASE_URL,
    }
