# pipeline_cfg.py
"""流程配置：Agent 接口地址、并发、超时、路径。"""
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# Agent 接口
# =============================================================================
# 业务接口 base（stream / conversation 等都挂在它下面）
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL")
# 登录接口 base（与业务接口前缀不同，独立配置）
AGENT_LOGIN_BASE_URL = os.getenv("AGENT_LOGIN_BASE_URL")
AGENT_LOGIN_PATH = "/api/damp-app-api/agent/legal-agent/auth/loginNew"

# 登录账号
AGENT_ACCOUNT = os.getenv("AGENT_ACCOUNT")
AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")

# 选择使用哪个已注册的 agent 客户端（见 audit_agent/registry.py）
AGENT_NAME = os.getenv("AGENT_NAME")

# =============================================================================
# 检索 / Trace（STUB 模块预留）
# =============================================================================
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K"))
TRACE_QUERY_URL = os.getenv("TRACE_QUERY_URL")

# =============================================================================
# 评估流程
# =============================================================================
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS"))
EVAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT"))

# =============================================================================
# 路径常量
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "testset_agent.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BADCASE_DIR = PROJECT_ROOT / "data" / "badcase"
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dirs():
    """一次性创建运行时目录。"""
    for d in (PROCESSED_DIR, BADCASE_DIR, RUNS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def new_run_dir() -> Path:
    """按 run_YYYYMMDD_HHMMSS 生成子目录并返回。"""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
