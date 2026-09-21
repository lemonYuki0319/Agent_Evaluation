# DeepEval 官方教程

> 基于 DeepEval 官方文档整理,覆盖安装、核心概念、指标、单元测试、自定义指标、自定义 LLM、CI/CD 集成。
>
> 官方文档: https://deepeval.com/docs

---

## 目录

1. [概述](#1-概述)
2. [安装](#2-安装)
3. [核心概念](#3-核心概念)
4. [LLMTestCase 详解](#4-llmtestcase-详解)
5. [指标体系](#5-指标体系)
6. [RAG 指标详解](#6-rag-指标详解)
7. [自定义指标 GEval](#7-自定义指标-geval)
8. [单元测试与断言](#8-单元测试与断言)
9. [批量评估 evaluate()](#9-批量评估-evaluate)
10. [数据集管理](#10-数据集管理)
11. [自定义 LLM](#11-自定义-llm)
12. [CLI 命令](#12-cli-命令)
13. [CI/CD 集成](#13-cicd-集成)
14. [本项目代码讲解](#14-本项目代码讲解)
15. [常见问题](#15-常见问题)

---

## 1. 概述

DeepEval 是一个开源的 LLM 评估框架,核心特点:

- **像 pytest 一样写 LLM 测试** — `assert_test()` + `deepeval test run`
- **50+ 预置指标** — 覆盖 RAG、Agent、多轮对话、安全、多模态
- **LLM-as-a-Judge** — 使用 LLM 对输出打分(0-1)
- **自定义指标** — GEval(自然语言定义)、DAG(决策树)
- **Confident AI** — 云端报告与监控(免费)

### DeepEval vs RAGAS 对比

| 维度 | DeepEval | RAGAS |
|------|---------|-------|
| 定位 | 综合 LLM 评估框架 | 专注 RAG 评估 |
| 测试风格 | pytest 单元测试 | 数据集评估 |
| 指标数量 | 50+ | ~20 |
| 断言机制 | `assert_test` + threshold | 无(只出分数) |
| 失败原因 | 每个指标带 reason | 部分有 |
| CI/CD | 原生支持 | 需自己集成 |
| 自定义指标 | GEval(自然语言) | 自定义 Prompt |

---

## 2. 安装

```bash
pip install -U deepeval
```

可选:登录 Confident AI 获取云端报告

```bash
deepeval login
```

> 浏览器会打开认证页面,注册/登录后自动创建项目。

---

## 3. 核心概念

```
┌─────────────────────────────────────────────────────┐
│                   评估工作流                          │
│                                                      │
│  Golden(测试数据)                                     │
│    ↓                                                 │
│  LLMTestCase(测试用例)                               │
│    ├── input          用户输入                        │
│    ├── actual_output  LLM 实际输出                    │
│    ├── expected_output 金标答案                      │
│    └── retrieval_context 检索上下文                   │
│    ↓                                                 │
│  Metric(指标)                                        │
│    ├── score  0-1                                    │
│    ├── reason  评分原因                              │
│    └── threshold 阈值                               │
│    ↓                                                 │
│  assert_test() / evaluate()                          │
│    → PASS / FAIL                                     │
└─────────────────────────────────────────────────────┘
```

三种评估方式:

| 方式 | 用途 | 调用 |
|------|------|------|
| `metric.measure(test_case)` | 单个指标独立打分 | 脚本 |
| `assert_test(test_case, [metrics])` | pytest 断言 | 单元测试 |
| `evaluate(test_cases, metrics)` | 批量评估 | 脚本/Notebook |

---

## 4. LLMTestCase 详解

`LLMTestCase` 是 DeepEval 的核心数据结构:

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="数据安全法是什么时候施行的?",       # 必填:用户输入
    actual_output="2021年9月1日起施行。",       # 必填:LLM 实际输出
    expected_output="2021年9月1日起施行。",      # 可选:金标答案
    retrieval_context=[                           # 可选:检索上下文(RAG)
        "《数据安全法》于2021年6月10日通过,9月1日起施行。"
    ],
    context=[                                      # 可选:额外上下文
        "附加信息..."
    ],
    tools_called=[                                 # 可选:Agent 调用的工具
        "search_law"
    ],
    expected_tools=[                               # 可选:期望调用的工具
        "search_law"
    ],
)
```

### 字段与指标的对应关系

| 字段 | 哪些指标需要 |
|------|------------|
| `input` | 几乎所有指标 |
| `actual_output` | 几乎所有指标 |
| `expected_output` | AnswerCorrectness, ContextualRecall, ContextualPrecision |
| `retrieval_context` | Faithfulness, ContextualPrecision, ContextualRecall, ContextualRelevancy |

### 多轮对话:ConversationalTestCase

```python
from deepeval.test_case import ConversationalTestCase, Turn

convo = ConversationalTestCase(
    turns=[
        Turn(role="user", content="你好"),
        Turn(role="assistant", content="你好!有什么可以帮你?"),
        Turn(role="user", content="介绍下数据安全法"),
        Turn(role="assistant", content="..."),
    ]
)
```

---

## 5. 指标体系

DeepEval 提供 50+ 指标,按用途分类:

### RAG 指标

| 指标 | 评估对象 | 说明 |
|------|---------|------|
| `AnswerRelevancyMetric` | 生成器 | 回答是否切题 |
| `FaithfulnessMetric` | 生成器 | 回答是否基于检索上下文(无幻觉) |
| `ContextualPrecisionMetric` | 检索器 | 相关片段是否排在前面 |
| `ContextualRecallMetric` | 检索器 | 金标要点是否被检索到 |
| `ContextualRelevancyMetric` | 检索器 | 检索内容是否与问题相关 |

### 自定义指标

| 指标 | 说明 |
|------|------|
| `GEval` | 自然语言定义评估标准(LLM-as-Judge + CoT) |
| `DAGMetric` | 决策树结构,更确定性 |
| `ConversationalGEval` | 多轮对话版 GEval |
| `BaseMetric` | 完全自定义(继承实现) |

### Agent 指标

| 指标 | 说明 |
|------|------|
| `TaskCompletionMetric` | 是否完成任务 |
| `ToolCallAccuracyMetric` | 工具调用是否正确 |
| `ToolCallF1Metric` | 工具调用 F1 |
| `StepEfficiencyMetric` | 步骤效率 |

### 安全指标

| 指标 | 说明 |
|------|------|
| `BiasMetric` | 偏见检测 |
| `ToxicityMetric` | 毒性检测 |
| `PIILeakageMetric` | PII 泄露 |
| `MisuseMetric` | 滥用检测 |

### 其他

| 指标 | 说明 |
|------|------|
| `HallucinationMetric` | 幻觉检测(与 Faithfulness 类似) |
| `SummarizationMetric` | 摘要质量 |
| `JsonCorrectnessMetric` | JSON 格式正确性 |

> **官方建议**: 每个项目不超过 5 个指标(2-3 个通用 + 1-2 个自定义)。

---

## 6. RAG 指标详解

### FaithfulnessMetric(忠实度)

评估 `actual_output` 是否与 `retrieval_context` 事实一致。

```python
from deepeval.metrics import FaithfulnessMetric

metric = FaithfulnessMetric(
    threshold=0.7,        # 通过阈值
    model="gpt-4",        # 评估模型
    include_reason=True,  # 返回评分原因
)
# 需要: input, actual_output, retrieval_context
```

**计算方式**: 将 actual_output 拆成事实陈述,逐条检查是否被 retrieval_context 支持。

### AnswerRelevancyMetric(答案相关性)

评估 `actual_output` 是否回应了 `input`。

```python
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=True,
)
# 需要: input, actual_output
```

**计算方式**: 从 actual_output 反向生成问题,计算与原始 input 的语义相似度。

### ContextualPrecisionMetric(上下文精确度)

评估 `retrieval_context` 中相关片段是否排在前面。

```python
from deepeval.metrics import ContextualPrecisionMetric

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4",
)
# 需要: input, expected_output, retrieval_context
```

### ContextualRecallMetric(上下文召回)

评估 `expected_output` 中的要点是否都出现在 `retrieval_context` 中。

```python
from deepeval.metrics import ContextualRecallMetric

metric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4",
)
# 需要: input, expected_output, retrieval_context
```

### ContextualRelevancyMetric(上下文相关性)

评估 `retrieval_context` 是否与 `input` 相关。

```python
from deepeval.metrics import ContextualRelevancyMetric

metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-4",
)
# 需要: input, retrieval_context
```

---

## 7. 自定义指标 GEval

GEval 是 DeepEval 最强大的功能,用自然语言定义任何评估标准:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness = GEval(
    name="准确性",                                    # 指标名
    criteria="判断实际输出与期望输出在事实层面是否一致",  # 评估标准(自然语言)
    evaluation_params=[                               # 用哪些字段评估
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,          # 通过阈值
    model="gpt-4",         # 评估模型
)
```

### criteria vs evaluation_steps

`criteria` 和 `evaluation_steps` 二选一:

```python
# 方式一:criteria(自然语言,GEval 自动生成步骤)
correctness = GEval(
    name="准确性",
    criteria="判断实际输出与期望输出在事实层面是否一致",
    evaluation_params=[...],
)

# 方式二:evaluation_steps(明确指定步骤,更可控)
correctness = GEval(
    name="准确性",
    evaluation_steps=[
        "检查实际输出中的事实是否与期望输出矛盾",
        "遗漏细节应严重扣分",
        "模糊表述或观点差异可以接受",
    ],
    evaluation_params=[...],
)
```

### GEval 参数完整列表

| 参数 | 必填 | 说明 |
|------|:---:|------|
| `name` | 是 | 指标名 |
| `criteria` | 二选一 | 自然语言评估标准 |
| `evaluation_steps` | 二选一 | 明确的评估步骤列表 |
| `evaluation_params` | 是 | 评估使用哪些字段 |
| `threshold` | 否 | 通过阈值,默认 0.5 |
| `model` | 否 | 评估模型,默认 gpt-5.4 |
| `strict_mode` | 否 | True=只打 0 或 1 |
| `async_mode` | 否 | 异步并发,默认 True |
| `verbose_mode` | 否 | 打印中间步骤 |
| `evaluation_template` | 否 | 自定义 Prompt 模板 |
| `rubric` | 否 | 限定分数范围 |

### Rubric 示例

```python
from deepeval.metrics.g_eval import Rubric

correctness = GEval(
    name="准确性",
    criteria="...",
    evaluation_params=[...],
    rubric=[
        Rubric(score_range=[0.8, 1.0], description="完全正确"),
        Rubric(score_range=[0.5, 0.79], description="部分正确"),
        Rubric(score_range=[0.0, 0.49], description="错误"),
    ],
)
```

---

## 8. 单元测试与断言

### assert_test()

`assert_test()` 是 DeepEval 的 pytest 集成核心:

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

def test_faithfulness():
    test_case = LLMTestCase(
        input="数据安全法何时施行?",
        actual_output="2021年9月1日起施行。",
        retrieval_context=["数据安全法2021年9月1日起施行。"],
    )
    metric = FaithfulnessMetric(threshold=0.7)
    assert_test(test_case, [metric])
    # score >= 0.7 → PASS
    # score <  0.7 → pytest 报红 FAIL
```

### 参数化测试

```python
TEST_DATA = [
    {"input": "...", "actual_output": "...", "expected_output": "..."},
    {"input": "...", "actual_output": "...", "expected_output": "..."},
]

@pytest.mark.parametrize("data", TEST_DATA)
def test_rag(data):
    test_case = LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data["expected_output"],
        retrieval_context=data["retrieval_context"],
    )
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ])
```

### 运行测试

```bash
# 用 deepeval CLI(推荐,带报告)
deepeval test run test_rag.py

# 用 pytest(标准)
pytest test_rag.py -v
```

### threshold 机制

```python
# 所有指标默认 threshold=0.5
metric = AnswerRelevancyMetric()
# score >= 0.5 → PASS

# 自定义阈值
metric = AnswerRelevancyMetric(threshold=0.8)
# score >= 0.8 → PASS

# threshold=None:不打分只记录,不断言
metric = AnswerRelevancyMetric(threshold=None)

# strict_mode:只打 0 或 1
metric = AnswerRelevancyMetric(strict_mode=True)
```

### flaky 测试

```python
# flaky=True:失败不影响整体结果,只发警告
metric = AnswerRelevancyMetric(threshold=0.7, flaky=True)
```

---

## 9. 批量评估 evaluate()

`evaluate()` 适合脚本和 Notebook,不做断言:

```python
from deepeval import evaluate

results = evaluate(
    test_cases=[test_case1, test_case2],
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ],
)

# 查看结果
for result in results.test_results:
    print(f"\n问题: {result.input}")
    for mr in result.metrics_data:
        status = "PASS" if mr.success else "FAIL"
        print(f"  [{status}] {mr.name}: {mr.score:.4f}")
        print(f"         原因: {mr.reason}")
```

### assert_test vs evaluate

| 维度 | `assert_test()` | `evaluate()` |
|------|-----------------|--------------|
| 用途 | pytest 单元测试 | 脚本批量评估 |
| 结果 | PASS/FAIL 断言 | 返回结果对象 |
| 适用 | CI/CD | Notebook/脚本 |
| 退出码 | 失败时非零 | 无 |

---

## 10. 数据集管理

### Golden 和 EvaluationDataset

```python
from deepeval.dataset import Golden, EvaluationDataset

# Golden = 测试数据的前体(只有 input,没有 output)
goldens = [
    Golden(input="数据安全法何时施行?"),
    Golden(input="数据安全法建立了什么制度?"),
]

dataset = EvaluationDataset(goldens=goldens)

# 遍历 goldens,调用你的应用,构建 test cases
for golden in dataset.goldens:
    answer, retrieved_chunks = my_rag_app(golden.input)
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=answer,
            retrieval_context=retrieved_chunks,
        )
    )

# 批量评估
evaluate(test_cases=dataset.test_cases, metrics=[...])
```

### 从文件加载

```python
# CSV
dataset = EvaluationDataset()
dataset.add_goldens_from_csv_file(
    file_path="test_data.csv",
    input_col_name="query",
)

# JSON
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path="test_data.json",
    input_key_name="query",
)
```

### 自动生成测试数据(Synthesizer)

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["知识库.txt"],
    chunk_size=500,
    chunk_overlap=50,
)
```

---

## 11. 自定义 LLM

### 方式一:OpenAI 兼容接口(最简单)

DeepEval 默认用 OpenAI。对于 OpenAI 兼容的 API(如 MiniMax),设环境变量即可:

```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "https://api.minimaxi.com/v1"

# 然后直接用 model 参数指定模型名
metric = AnswerRelevancyMetric(model="MiniMax-M3")
```

### 方式二:LocalModel(本地/自部署模型)

```python
from deepeval.models import LocalModel

model = LocalModel(
    model="your-model-name",
    base_url="http://localhost:1234/v1/",
    api_key="placeholder",   # 无认证可填任意值
    temperature=0,
)

metric = AnswerRelevancyMetric(model=model)
```

### 方式三:CLI 设置(持久化)

```bash
# 设置本地模型
deepeval set-local-model --model=llama3 --base-url="http://localhost:1234/v1/"

# 设置 Ollama
deepeval set-ollama --model=deepseek-r1:1.5b

# 设置 Gemini
deepeval set-gemini --model=gemini-2.0-flash-001

# 切回 OpenAI
deepeval unset-local-model
```

### 方式四:DeepEvalBaseLLM(完全自定义)

继承 `DeepEvalBaseLLM` 实现任意模型:

```python
from deepeval.models import DeepEvalBaseLLM

class MyCustomLLM(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        # 调用你的模型 API
        response = my_api_call(prompt)
        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# 使用
model = MyCustomLLM("my-model")
metric = AnswerRelevancyMetric(model=model)
```

---

## 12. CLI 命令

| 命令 | 说明 |
|------|------|
| `deepeval login` | 登录 Confident AI |
| `deepeval test run test_file.py` | 运行测试(等同 pytest + 报告) |
| `deepeval set-local-model` | 设置本地模型 |
| `deepeval set-ollama` | 设置 Ollama 模型 |
| `deepeval set-gemini` | 设置 Gemini 模型 |
| `deepeval unset-local-model` | 切回 OpenAI |

---

## 13. CI/CD 集成

### GitHub Actions 示例

```yaml
name: LLM Evaluation
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -U deepeval
      - run: deepeval test run test_rag.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

> 测试失败时 CI 报红,阻止合并。登录 Confident AI 后还可查看在线报告。

---

## 14. 本项目代码讲解

[deepeval_test.py](file:///e:/python_project/llm_test/RAG/deepeval_test.py) 结构讲解:

### 14.1 配置区(L29-35)

```python
os.environ["OPENAI_API_KEY"] = "sk-cp-..."
os.environ["OPENAI_API_BASE"] = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M3"
THRESHOLD = 0.7
```

用环境变量配置 MiniMax(OpenAI 兼容接口),所有指标共享同一个模型。

### 14.2 测试数据(L38-59)

```python
TEST_DATA = [
    {
        "input": "数据安全法是什么时候施行的?",
        "actual_output": "《数据安全法》于2021年9月1日起施行。",
        "expected_output": "2021年9月1日起施行。",
        "retrieval_context": ["..."],
    },
    ...
]
```

用字典列表存储测试数据,`@pytest.mark.parametrize` 逐条遍历。

### 14.3 构建函数(L62-98)

```python
def make_test_case(data):
    """字典 → LLMTestCase"""
    return LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data["expected_output"],
        retrieval_context=data["retrieval_context"],
    )

def build_metrics():
    """4 个内置 RAG 指标,统一阈值"""
    return [
        AnswerRelevancyMetric(threshold=THRESHOLD, model=MODEL),
        FaithfulnessMetric(threshold=THRESHOLD, model=MODEL),
        ContextualRecallMetric(threshold=THRESHOLD, model=MODEL),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=MODEL),
    ]

def build_custom_metric():
    """自定义 GEval 指标"""
    return GEval(
        name="准确性",
        criteria="判断实际输出与期望输出在事实层面是否一致",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=MODEL,
        threshold=THRESHOLD,
    )
```

### 14.4 三种测试方式(L101-163)

**方式一:AI 单元测试(单指标)**

```python
@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_faithfulness(data):
    test_case = make_test_case(data)
    metric = FaithfulnessMetric(threshold=THRESHOLD, model=MODEL)
    assert_test(test_case, [metric])
```

每条数据跑一次,只测 Faithfulness。`assert_test` 自动断言,score >= 0.7 才 PASS。

**方式二:综合测试(多指标 + 自定义)**

```python
@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_full_suite(data):
    test_case = make_test_case(data)
    metrics = build_metrics() + [build_custom_metric()]
    assert_test(test_case, metrics)
```

一次跑 5 个指标(4 内置 + 1 GEval),全部通过才算 PASS。

**方式三:批量评估(不断言)**

```python
def main():
    results = evaluate(test_cases=..., metrics=...)
    for result in results.test_results:
        for mr in result.metrics_data:
            status = "PASS" if mr.success else "FAIL"
            print(f"[{status}] {mr.name}: {mr.score:.4f}")
```

只看分数不做断言,适合探索性评估。

---

## 15. 常见问题

### Q: 指标分数不稳定/随机怎么办?

1. 开启 `verbose_mode=True` 查看中间推理
2. 用 `strict_mode=True` 强制 0/1 评分
3. 换更强的评估模型(如 gpt-4)
4. 用 DAG 指标替代 GEval(更确定性)
5. 降低 `temperature=0`

### Q: threshold 设多少合适?

| 场景 | 建议 |
|------|------|
| 开发阶段 | 0.5(宽松) |
| 上线前 | 0.7(标准) |
| 安全/合规 | 0.9+(严格) |

### Q: 必须用 OpenAI 吗?

不是。可通过环境变量、`LocalModel`、`DeepEvalBaseLLM` 或 CLI 切换到任何模型。

### Q: deepeval test run 和 evaluate() 有什么区别?

| | `deepeval test run` | `evaluate()` |
|---|---|---|
| 类型 | CLI 命令(基于 pytest) | Python 函数 |
| 断言 | 有(assert_test 语义) | 无 |
| 退出码 | 失败时非零 | 无 |
| 场景 | CI/CD | 脚本/Notebook |

### Q: 指标选几个?

官方建议不超过 5 个:2-3 个通用 + 1-2 个自定义(GEval)。

---

## 参考链接

- [DeepEval 官方文档](https://deepeval.com/docs)
- [5 分钟快速入门](https://deepeval.com/docs/getting-started)
- [RAG 评估指南](https://deepeval.com/docs/getting-started-rag)
- [GEval 文档](https://deepeval.com/docs/metrics-llm-evals)
- [指标总览](https://deepeval.com/docs/metrics-introduction)
- [自定义 LLM 指南](https://deepeval.com/guides/guides-using-custom-llms)
- [Confident AI](https://www.confident-ai.com/)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
