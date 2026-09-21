# evaluator_deepeval.py
"""DeepEval 评测主逻辑：忠实度、上下文召回、上下文精确度 + GEval。"""
import os

# DeepEval 通过 OpenAI SDK 调 Judge LLM，需先注入环境变量
from config.metric_cfg import (
    DEEPEVAL_METRICS_ENABLED,
    DEFAULT_THRESHOLD,
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    get_judge_env,
)

os.environ.update(get_judge_env())

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def _build_judge_model() -> OpenAIModel:
    """创建 Judge 模型实例，注入 response_format 确保 JSON 输出。

    DeepSeek 等非 OpenAI 原生模型不在 DeepEval 已知模型表里，
    supports_json_mode() 返回 None 导致 JSON mode 路径被跳过。
    通过 generation_kwargs 注入 response_format 补齐。
    """
    return OpenAIModel(
        model=JUDGE_MODEL,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        generation_kwargs={"response_format": {"type": "json_object"}},
        temperature=0.0
    )


def build_metrics():
    """按开关构造 DeepEval 指标列表。"""
    model = _build_judge_model()
    metrics = []
    if DEEPEVAL_METRICS_ENABLED.get("faithfulness"):
        metrics.append(FaithfulnessMetric(threshold=DEFAULT_THRESHOLD, model=model))
    if DEEPEVAL_METRICS_ENABLED.get("answer_relevancy"):
        metrics.append(AnswerRelevancyMetric(threshold=DEFAULT_THRESHOLD, model=model))
    if DEEPEVAL_METRICS_ENABLED.get("contextual_recall"):
        metrics.append(ContextualRecallMetric(threshold=DEFAULT_THRESHOLD, model=model))
    if DEEPEVAL_METRICS_ENABLED.get("contextual_precision"):
        metrics.append(ContextualPrecisionMetric(threshold=DEFAULT_THRESHOLD, model=model))
    if DEEPEVAL_METRICS_ENABLED.get("geval"):
        metrics.append(_build_geval(model))
    return metrics


def _build_geval(model: OpenAIModel):
    """自定义 GEval：Agent 语义对齐度。"""
    return GEval(
        name="语义对齐度",
        criteria=(
            "你是一个严格的 Agent 评估专家。请根据 input(用户输入)判断 "
            "actual_output(Agent 实际回答) 与 expected_output(期望回答) "
            "在语义层面的对齐程度。综合考虑:\n"
            "1. 意图理解: 是否正确识别用户真实意图\n"
            "2. 关键点覆盖: expected_output 中的关键处理点是否在 actual_output 中体现\n"
            "3. 行为合理性: 是否避免了编造信息、绕过权限、滥用工具等问题\n"
            "打分范围 0~1: 1.0=完全对齐, 0.7=基本合格, <0.5=明显偏差"
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=DEFAULT_THRESHOLD,
    )


def _format_input(rec):
    """构造 LLMTestCase.input。

    单轮：直接返回 user_input；
    多轮：把完整 messages 格式化为 "用户/助手: ..." 文本，让 Judge 能看到
          上下文（含指代、上下文记忆等），actual_output/expected_output/retrieval_context
          仍对应最后一轮。
    """
    if "messages" in rec:
        role_label = {"user": "用户", "assistant": "助手"}
        lines = [
            f"{role_label.get(m['role'], m['role'])}: {m['content']}"
            for m in rec["messages"]
        ]
        return "\n".join(lines)
    return rec["user_input"]


def build_test_cases(records):
    """把流程记录转成 DeepEval LLMTestCase 列表。

    单轮记录（含 user_input）和多轮记录（含 messages）都构造为 LLMTestCase：
    多轮时 input 为格式化对话历史，actual_output/expected_output/retrieval_context
    对应最后一轮（模型回答 / 标注答案 / 检索切片）。
    """
    test_cases = []
    for rec in records:
        contexts = rec.get("retrieved_contexts") or ["(无检索上下文)"]
        test_cases.append(
            LLMTestCase(
                input=_format_input(rec),
                actual_output=rec["actual_output"],
                expected_output=rec["ground_truth"],
                retrieval_context=contexts,
            )
        )
    return test_cases


def run_deepeval(records):
    """
    对一批记录跑 DeepEval 指标，返回每条的分数与原因。

    Args:
        records: list[dict]，单轮含 user_input/ground_truth/actual_output/retrieved_contexts；
                  多轮含 messages/ground_truth/actual_output/retrieved_contexts

    Returns:
        list[dict]，每条形如:
        {"index": i, "deepeval": {"faithfulness": (score, success, reason), ...}}
    """
    if not records:
        return []

    metrics = build_metrics()
    if not metrics:
        print("[WARN] 未启用任何 DeepEval 指标，跳过")
        return [{"index": i, "deepeval": {}} for i in range(len(records))]

    test_cases = build_test_cases(records)

    print(f"[INFO] DeepEval 评估开始，共 {len(records)} 条样本，{len(metrics)} 个指标")
    try:
        results = evaluate(test_cases=test_cases, metrics=metrics)
    except Exception as e:
        print(f"[ERROR] DeepEval 整体失败: {e}")
        return [
            {"index": i, "deepeval": {getattr(m, "name", type(m).__name__): (None, False, str(e)) for m in metrics}}
            for i in range(len(records))
        ]

    scores = []
    for i, tr in enumerate(results.test_results):
        row = {}
        for md in tr.metrics_data:
            row[md.name] = (md.score, md.success, md.reason)
        scores.append({"index": i, "deepeval": row})

    print("[OK] DeepEval 评估完成")
    return scores
