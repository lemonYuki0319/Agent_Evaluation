# report_builder.py
"""汇总双框架分数，输出 details.csv / summary.json / bad_cases.jsonl。"""
import json

import pandas as pd

from config.metric_cfg import DEFAULT_THRESHOLD


def _extract_scores(score_entry, prefix):
    """
    从单条 RAGAS 或 DeepEval 分数字典中抽出扁平列。

    RAGAS: {"faithfulness": 0.85, "context_recall": 0.7, ...}
    DeepEval: {"faithfulness": (0.8, True, "reason..."), ...}
    """
    flat = {}
    if not score_entry:
        return flat

    for name, val in score_entry.items():
        col = f"{prefix}_{name}"
        if isinstance(val, tuple):
            # DeepEval: (score, success, reason)
            score = val[0] if val[0] is not None else None
            flat[col] = score
            flat[f"{col}_pass"] = val[1]
            flat[f"{col}_reason"] = val[2]
        else:
            # RAGAS: 直接是数值或 None
            flat[col] = val
    return flat


def build_report(records, ragas_scores, deepeval_scores, run_dir):
    """
    合并三路数据，写出 details.csv / summary.json / bad_cases.jsonl。

    Args:
        records: 流程记录列表，每条含 user_input/ground_truth/actual_output/trace_id
        ragas_scores: RAGAS 分数列表，每条 {"index": i, "ragas": {...}}
        deepeval_scores: DeepEval 分数列表，每条 {"index": i, "deepeval": {...}}
        run_dir: 输出目录 Path

    Returns:
        summary 字典供 main.py 打印
    """
    run_id = run_dir.name

    # 1. 构建明细 DataFrame
    rows = []
    for i, rec in enumerate(records):
        row = {
            "index": i,
            "user_input": rec.get("user_input", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "actual_output": rec.get("actual_output", ""),
            "trace_id": rec.get("trace_id", ""),
        }

        # RAGAS 分数
        if i < len(ragas_scores):
            row.update(_extract_scores(ragas_scores[i].get("ragas", {}), "ragas"))

        # DeepEval 分数
        if i < len(deepeval_scores):
            row.update(_extract_scores(deepeval_scores[i].get("deepeval", {}), "deepeval"))

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = run_dir / "details.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 明细已写出: {csv_path}")

    # 2. 统计摘要
    summary = {"run_id": run_id, "total": len(records), "per_metric": {}}

    # 找出所有分数列（以 ragas_ 或 deepeval_ 开头，不含 _pass/_reason）
    score_cols = [
        c
        for c in df.columns
        if (c.startswith("ragas_") or c.startswith("deepeval_"))
        and not c.endswith("_pass")
        and not c.endswith("_reason")
    ]

    for col in score_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        summary["per_metric"][col] = {
            "mean": round(float(vals.mean()), 4),
            "pass_rate": round(float((vals >= DEFAULT_THRESHOLD).mean()), 4),
            "n": int(len(vals)),
        }

    # bad case 筛选：任一分数列 < 阈值
    bad_mask = pd.Series([False] * len(df))
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad_mask = bad_mask | (vals < DEFAULT_THRESHOLD)

    bad_df = df[bad_mask]
    summary["bad_case_count"] = int(len(bad_df))

    json_path = run_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] 摘要已写出: {json_path}")

    # 3. bad_cases.jsonl
    bad_path = run_dir / "bad_cases.jsonl"
    with open(bad_path, "w", encoding="utf-8") as f:
        for _, row in bad_df.iterrows():
            f.write(row.to_json(force_ascii=False) + "\n")
    print(f"[OK] bad case 已写出: {bad_path} ({summary['bad_case_count']} 条)")

    return summary
