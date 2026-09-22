# main.py
"""
入口脚本：加载数据集 → 批量调用 Agent → 双框架打分 → 生成报告。

用法:
    python main.py                          # 全量运行
    python main.py --max-cases 3            # 只跑前 3 条
    python main.py --skip-deepeval          # 只跑 RAGAS
    python main.py --skip-ragas             # 只跑 DeepEval
"""
import argparse
import json
import sys
from pathlib import Path

# 保证项目根在 sys.path，使 audit_agent / config / src 均可导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.pipeline_cfg import (
    AGENT_NAME,
    PROCESSED_DIR,
    RAW_DATA_FILE,
    ensure_dirs,
    new_run_dir,
)
from config.logger import get_logger
from audit_agent import get_agent
from src.evaluator_deepeval import run_deepeval
from src.evaluator_ragas import run_ragas
from src.report_builder import build_report


def load_dataset(path):
    """读取数据集，兼容单行 JSONL 与多行展开的 JSON 对象。

    每条记录可为单轮 ({user_input, ground_truth}) 或多轮
    ({messages: [...], ground_truth})，由调用方按字段分流。
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    idx = 0
    n = len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(content, idx)
        records.append(obj)
        idx = end
    return records


def _run_agent_batch(records, call_agent, cache_path):
    """批量调用 Agent，把 answer/trace_id/retrieved_contexts 写回每条 record 并落盘缓存。

    每条 record 按字段自动分流：
        - 含 messages：多轮，把完整对话历史传给 agent；同时填充 user_input 字段
                       （取最后一条 user 内容）供 evaluator_ragas 等下游评测器使用
        - 否则：单轮，传 user_input 字符串
    """
    with open(cache_path, "w", encoding="utf-8") as cache_f:
        for i, rec in enumerate(records, 1):
            if "messages" in rec:
                prompt = rec["messages"]
                user_msgs = [m for m in rec["messages"] if m.get("role") == "user"]
                rec.setdefault("user_input", user_msgs[-1]["content"])
                preview = f"多轮({len(user_msgs)} 轮) 首轮: {user_msgs[0]['content'][:40]}"
            else:
                prompt = rec["user_input"]
                preview = rec["user_input"][:60]
            print(f"\n[{i}/{len(records)}] {preview}...")
            out = call_agent(prompt)
            rec["actual_output"] = out["answer"]
            rec["trace_id"] = out["trace_id"]
            rec["retrieved_contexts"] = out.get("retrieved_contexts", [])
            print(f"  trace_id={out['trace_id']}")
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_f.flush()


def main():
    parser = argparse.ArgumentParser(description="RAG 混合评测（DeepEval + RAGAS）")
    parser.add_argument("--data", default=str(RAW_DATA_FILE), help="数据集路径")
    parser.add_argument("--max-cases", type=int, help="最大用例数（调试用）")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS")
    parser.add_argument("--skip-deepeval", action="store_true", help="跳过 DeepEval")
    args = parser.parse_args()

    # .env 已由 config 包在导入时加载
    ensure_dirs()
    run_dir = new_run_dir()

    # 1. 加载数据集
    records = load_dataset(args.data)
    if args.max_cases:
        records = records[: args.max_cases]
    print(f"[INFO] 加载 {len(records)} 条样本")
    get_logger().info(f"评测启动: agent={AGENT_NAME} run_id={run_dir.name} 样本数={len(records)}")

    # 2. 批量调用 Agent，边调边落盘
    call_agent = get_agent(AGENT_NAME)
    print(f"[INFO] 使用 agent: {AGENT_NAME}")
    cache_path = PROCESSED_DIR / f"agent_outputs_{run_dir.name}.jsonl"
    _run_agent_batch(records, call_agent, cache_path)
    print(f"\n[OK] Agent 调用完成，缓存: {cache_path}")

    # 3. RAGAS 评测
    ragas_scores = []
    if not args.skip_ragas:
        ragas_scores = run_ragas(records)
    else:
        ragas_scores = [{"index": i, "ragas": {}} for i in range(len(records))]

    # 4. DeepEval 评测
    deepeval_scores = []
    if not args.skip_deepeval:
        deepeval_scores = run_deepeval(records)
    else:
        deepeval_scores = [{"index": i, "deepeval": {}} for i in range(len(records))]

    # 5. 生成报告
    summary = build_report(records, ragas_scores, deepeval_scores, run_dir)

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print(f"评测完成  run_id={summary['run_id']}  样本数={summary['total']}")
    print("=" * 60)
    for metric, stats in summary.get("per_metric", {}).items():
        print(
            f"  {metric}: mean={stats['mean']:.4f}  "
            f"pass_rate={stats['pass_rate']:.2%}  n={stats['n']}"
        )
    print(f"\n  bad_case: {summary.get('bad_case_count', 0)} 条")
    print(f"\n  输出目录: {run_dir}")
    print("=" * 60)
    get_logger().info(f"评测完成: run_id={summary['run_id']} 样本数={summary['total']} bad_case={summary.get('bad_case_count', 0)}")


if __name__ == "__main__":
    main()
