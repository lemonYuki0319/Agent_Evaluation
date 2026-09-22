# langfuse_eval.py
"""
拉取 Langfuse god_case 数据集 → 调审计 Agent。
业务侧 LangGraph 自动上报 trace，平台 rule 命中 synthesize 子节点触发评估。

用法:
    python langfuse_eval.py                  # 全量
    python langfuse_eval.py --max-cases 3    # 调试
"""
import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langfuse import get_client  # noqa: E402
from config.logger import get_logger  # noqa: E402
from config.pipeline_cfg import AGENT_NAME  # noqa: E402
from audit_agent import get_agent  # noqa: E402

DATASET_NAME = "god_case"


def _extract_input(item) -> str:
    """提取 item.input，兼容 messages / text / str 三种结构。"""
    data = getattr(item, "input", None)
    if isinstance(data, dict):
        msgs = data.get("messages")
        if msgs:
            users = [m for m in msgs if m.get("role") == "user"]
            if users:
                return users[-1]["content"]
        if "text" in data:
            return data["text"]
    return data if isinstance(data, str) else ""


def run(max_cases: int = None):
    langfuse = get_client()
    if not langfuse.auth_check():
        print("Langfuse 连接失败")
        return

    items = langfuse.get_dataset(DATASET_NAME).items
    items = items[:max_cases] if max_cases else items
    call_agent = get_agent(AGENT_NAME)
    print(f"[INFO] 开始评估: agent={AGENT_NAME} dataset={DATASET_NAME} n={len(items)}\n")

    ok = 0
    for i, item in enumerate(items, 1):
        q = _extract_input(item)
        if not q:
            continue
        try:
            print(f"[{i}/{len(items)}] 调用 agent...")
            out = call_agent(q)
            print(f"  ✓ trace_id={out.get('trace_id', '')}")
            print(f"  answer={out.get('answer', '')[:100]}\n")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(items)}] ✗ 失败: {e}\n")
        time.sleep(1)

    langfuse.flush()
    print(f"[INFO] 完成: {ok}/{len(items)} 已上报 Langfuse")
    get_logger().info(f"Langfuse 数据集跑批完成: 成功 {ok}/{len(items)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-cases", type=int)
    run(p.parse_args().max_cases)
