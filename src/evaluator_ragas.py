# evaluator_ragas.py
"""RAGAS 评测主逻辑：忠实度、上下文精确度、上下文召回。"""
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

from config.metric_cfg import (
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    RAGAS_METRICS_ENABLED,
)
from config.pipeline_cfg import EVAL_MAX_WORKERS, EVAL_TIMEOUT


def build_llm():
    """构建 Judge LLM 包装器。"""
    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        temperature=0,
        max_retries=2,
        timeout=120,
    )
    return LangchainLLMWrapper(chat)


def build_metrics(llm):
    """按开关构造 RAGAS 指标列表。"""
    metrics = []
    if RAGAS_METRICS_ENABLED.get("faithfulness"):
        metrics.append(Faithfulness(llm=llm))
    if RAGAS_METRICS_ENABLED.get("answer_relevancy"):
        metrics.append(ResponseRelevancy(llm=llm))
    if RAGAS_METRICS_ENABLED.get("context_recall"):
        metrics.append(LLMContextRecall(llm=llm))
    if RAGAS_METRICS_ENABLED.get("context_precision"):
        metrics.append(LLMContextPrecisionWithReference(llm=llm))
    return metrics


def build_samples(records):
    """把流程记录转成 RAGAS SingleTurnSample 列表。"""
    samples = []
    for rec in records:
        contexts = rec.get("retrieved_contexts") or ["(无检索上下文)"]
        samples.append(
            SingleTurnSample(
                user_input=rec["user_input"],
                response=rec["actual_output"],
                retrieved_contexts=contexts,
                reference=rec["ground_truth"],
            )
        )
    return samples


def run_ragas(records):
    """
    对一批记录跑 RAGAS 指标，返回每条的分数字典。

    Args:
        records: list[dict]，每条含 user_input/ground_truth/actual_output/retrieved_contexts

    Returns:
        list[dict]，每条形如:
        {"index": i, "ragas": {"faithfulness": 0.85, "context_recall": 0.7, ...}}
        单条失败时分数为 None。
    """
    if not records:
        return []

    llm = build_llm()
    metrics = build_metrics(llm)
    if not metrics:
        print("[WARN] 未启用任何 RAGAS 指标，跳过")
        return [{"index": i, "ragas": {}} for i in range(len(records))]

    samples = build_samples(records)
    dataset = EvaluationDataset(samples)
    run_config = RunConfig(
        timeout=EVAL_TIMEOUT, max_workers=EVAL_MAX_WORKERS, max_retries=1
    )

    print(f"[INFO] RAGAS 评估开始，共 {len(records)} 条样本，{len(metrics)} 个指标")
    try:
        result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    except Exception as e:
        print(f"[ERROR] RAGAS 整体失败: {e}")
        return [
            {"index": i, "ragas": {m.name: None for m in metrics}}
            for i in range(len(records))
        ]

    df = result.to_pandas()

    # 指标名 -> 列名映射
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_recall": "context_recall",
        "context_precision": "llm_context_precision_with_reference",
    }

    scores = []
    for i in range(len(df)):
        row_scores = {}
        for metric_name, col in col_map.items():
            if not RAGAS_METRICS_ENABLED.get(metric_name):
                continue
            val = df.iloc[i].get(col)
            row_scores[metric_name] = float(val) if val is not None else None
        scores.append({"index": i, "ragas": row_scores})

    print("[OK] RAGAS 评估完成")
    return scores
