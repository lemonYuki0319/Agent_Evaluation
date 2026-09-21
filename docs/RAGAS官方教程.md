# RAGAS 官方教程

> 基于 RAGAS v0.4.x 官方文档整理,涵盖核心概念、基本语法、常用方法与实战示例。
> 官方文档: https://docs.ragas.io/

---

## 目录

- [1. RAGAS 概述](#1-ragas-概述)
- [2. 安装与环境配置](#2-安装与环境配置)
- [3. 核心概念](#3-核心概念)
- [4. 数据结构:SingleTurnSample 与 EvaluationDataset](#4-数据结构singleturnsample-与-evaluationdataset)
- [5. LLM 初始化](#5-llm-初始化)
- [6. 核心指标详解](#6-核心指标详解)
- [7. 基本评估流程](#7-基本评估流程)
- [8. 单指标评分(独立调用)](#8-单指标评分独立调用)
- [9. 批量评估:evaluate() 与 experiment()](#9-批量评估evaluate-与-experiment)
- [10. 自定义 Prompt](#10-自定义-prompt)
- [11. RunConfig 配置](#11-runconfig-配置)
- [12. 结合本项目的实战示例](#12-结合本项目的实战示例)
- [13. 常见问题与排坑](#13-常见问题与排坑)
- [14. v0.3 → v0.4 API 变更速查](#14-v03--v04-api-变更速查)

---

## 1. RAGAS 概述

RAGAS(Retrieval-Augmented Generation Assessment)是一个用于评估 RAG 系统的 Python 框架。它通过 LLM-as-Judge 的方式,自动量化 RAG 系统的检索质量和生成质量,无需人工逐条标注。

### 四大核心指标

| 指标 | 评估什么 | 需要哪些字段 | 需要金标答案? |
|------|---------|-------------|:---:|
| **Faithfulness(忠实度)** | 回答中的每个论断是否都能在检索到的上下文中找到依据 | `response` + `retrieved_contexts` | 否 |
| **Answer Relevancy(答案相关性)** | 回答是否真正回答了用户的问题 | `user_input` + `response` | 否 |
| **Context Precision(上下文精确度)** | 检索到的片段中,相关的排在前面了吗 | `user_input` + `retrieved_contexts` + `reference` | 是 |
| **Context Recall(上下文召回)** | 标准答案的要点是否都出现在检索到的上下文中 | `user_input` + `retrieved_contexts` + `reference` | 是 |

> **关键优势**: Faithfulness 和 Answer Relevancy 不需要人工标注的标准答案,可以直接用于线上流量评估。只有 Context Precision/Recall 需要 `reference`(金标答案)。

### 指标组合诊断逻辑

```
context_recall 低 → 检索环节出问题(切分/embedding/top_k)
faithfulness 低   → 生成环节出问题(prompt 没限制住模型发散)
answer_relevancy 低 → 问题理解环节(query rewrite)
```

---

## 2. 安装与环境配置

```bash
pip install ragas
```

如果你使用 LangChain 生态:

```bash
pip install ragas langchain-openai
```

设置 API Key(以 OpenAI 为例):

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

---

## 3. 核心概念

### 3.1 架构演进

RAGAS 经历了几个重要版本:

| 版本 | 数据结构 | 评估方式 | 状态 |
|------|---------|---------|------|
| ≤0.1.x | HuggingFace Dataset | `evaluate(dataset, metrics)` | 已废弃 |
| 0.2.x~0.3.x | `SingleTurnSample` + `EvaluationDataset` | `evaluate(dataset=..., metrics=...)` | 兼容可用 |
| **0.4.x** | `SingleTurnSample` + `EvaluationDataset` | `experiment()` 装饰器 + `ascore()` | **当前版本** |

### 3.2 两种 API 风格

**Legacy API(本项目当前使用,仍兼容)**:

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall

# 创建样本 → 创建数据集 → 批量评估
samples = [SingleTurnSample(user_input=..., response=..., retrieved_contexts=..., reference=...)]
dataset = EvaluationDataset(samples)
result = evaluate(dataset=dataset, metrics=[Faithfulness(llm=llm), ...])
```

**v0.4 新 API(推荐新项目使用)**:

```python
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# 用 experiment 装饰器定义评估流程
@experiment(MyResultModel)
async def run_eval(row):
    scorer = Faithfulness(llm=llm)
    result = await scorer.ascore(response=row.response, retrieved_contexts=row.contexts)
    return MyResultModel(faithfulness=result.value)
```

> **本项目说明**: 当前 `evaluator.py` 使用 Legacy API + `evaluate()`,在 v0.4 中仍可用(标记为 deprecated),功能正常。新项目建议直接用 v0.4 API。

---

## 4. 数据结构:SingleTurnSample 与 EvaluationDataset

### 4.1 SingleTurnSample

`SingleTurnSample` 是单轮对话的评估样本,是 RAGAS 的核心数据结构。

```python
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="数据安全法是什么时候施行的?",       # 用户问题
    retrieved_contexts=["《数据安全法》于2021年9月1日起施行。"],  # 检索到的上下文(列表)
    response="《数据安全法》于2021年9月1日起施行。",   # RAG 系统的回答
    reference="2021年9月1日",                          # 标准答案(金标,部分指标需要)
)
```

### 字段说明

| 字段 | 类型 | 说明 | 哪些指标需要 |
|------|------|------|-------------|
| `user_input` | `str` | 用户提问 | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall |
| `retrieved_contexts` | `List[str]` | 检索到的上下文片段列表 | Faithfulness, ContextPrecision, ContextRecall |
| `response` | `str` | RAG 系统生成的回答 | Faithfulness, AnswerRelevancy |
| `reference` | `str` | 标准答案(金标) | ContextPrecision, ContextRecall |

> **注意**: v0.2+ 将旧版的 `question` → `user_input`、`answer` → `response`、`contexts` → `retrieved_contexts`、`ground_truth` → `reference`,语义更清晰。

### 4.2 EvaluationDataset

`EvaluationDataset` 是样本集合,用于批量评估。

```python
from ragas import EvaluationDataset, SingleTurnSample

samples = [
    SingleTurnSample(
        user_input="问题1",
        retrieved_contexts=["上下文1"],
        response="回答1",
        reference="标准答案1",
    ),
    SingleTurnSample(
        user_input="问题2",
        retrieved_contexts=["上下文2"],
        response="回答2",
        reference="标准答案2",
    ),
]

dataset = EvaluationDataset(samples)
```

---

## 5. LLM 初始化

### 5.1 v0.4 统一工厂:llm_factory()

```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# OpenAI
llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI(api_key="sk-..."))

# Anthropic
from anthropic import AsyncAnthropic
llm = llm_factory("claude-3-5-sonnet-20241022", client=AsyncAnthropic(api_key="..."))

# Google Gemini
llm = llm_factory("gemini-2.0-flash", client=...)

# 本地 Ollama
llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")
```

**优势**:
- 自动检测 provider(无需手动传 provider 字符串)
- 自动处理 GPT-5/o-series 的 temperature/top_p 约束
- 返回 `InstructorBaseRagasLLM`,支持结构化输出

### 5.2 LangChain 方式(Legacy,本项目使用)

```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# 创建 LangChain LLM 实例
chat_llm = ChatOpenAI(
    model_name="gpt-4o",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    temperature=0,
)

# 包装成 RAGAS 可用的 LLM
evaluator_llm = LangchainLLMWrapper(chat_llm)

# Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
```

> **注意**: `LangchainLLMWrapper` 在 v0.4 中标记为 deprecated,但仍然可用。本项目 `evaluator.py` 使用的就是这个方式。

### 5.3 embedding_factory()(用于 Embedding 模型)

```python
from ragas.embeddings import embedding_factory

# 使用 OpenAI 兼容接口
modern_emb = embedding_factory(
    provider="openai",
    model="text-embedding-ada-002",
    client=openai_client,  # 自定义 client
)
```

---

## 6. 核心指标详解

### 6.1 Faithfulness(忠实度)

**评估什么**: 回答中的每个论断是否都能在检索到的上下文中找到依据。防止模型"编造"(幻觉)。

**计算方式**:
1. 将回答拆分为多个"论断"(statement)
2. 逐个检查每个论断是否能从上下文推断出来
3. `忠实度 = 被支持的论断数 / 总论断数`

```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=evaluator_llm)

# v0.4 新方式:直接传关键字参数
result = await scorer.ascore(
    user_input="第一届超级碗是什么时候?",
    response="第一届超级碗于1967年1月15日举行。",
    retrieved_contexts=["第一届AFL-NFL世界冠军赛于1967年1月15日在洛杉矶举行。"],
)
print(result.value)   # 分数: 1.0
print(result.reason)   # 可选的解释
```

### 6.2 AnswerRelevancy(答案相关性)

**评估什么**: 回答是否真正回答了用户的问题,而非答非所问。

**计算方式**:
1. 从回答反向生成若干"可能的问题"
2. 计算这些生成问题与原始问题的语义相似度(需 Embedding)
3. 相似度越高 = 回答越切题

```python
from ragas.metrics.collections import AnswerRelevancy

scorer = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb, strictness=1)
# strictness: 生成几个候选问题(默认3,越大越严格但更慢)

result = await scorer.ascore(
    user_input="数据安全法什么时候施行?",
    response="《数据安全法》于2021年9月1日起施行。",
)
print(result.value)  # 0~1,越接近1越相关
```

### 6.3 ContextRecall(上下文召回)

**评估什么**: 标准答案中的要点是否都出现在检索到的上下文中。

**计算方式**:
1. 将标准答案拆分为句子
2. 逐句检查是否能从检索上下文中找到支持
3. `召回率 = 被支持的句子数 / 总句子数`

```python
from ragas.metrics.collections import ContextRecall

scorer = ContextRecall(llm=evaluator_llm)

result = await scorer.ascore(
    user_input="数据安全法的主要内容?",
    retrieved_contexts=["《数据安全法》规定了数据分类分级保护制度..."],
    reference="数据安全法建立了数据分类分级保护、数据安全审查等制度。",
)
print(result.value)
```

### 6.4 ContextPrecision(上下文精确度)

**评估什么**: 检索到的片段中,相关的片段是否排在前面(top-k 精确度)。

```python
from ragas.metrics.collections import ContextPrecision

scorer = ContextPrecision(llm=evaluator_llm)

result = await scorer.ascore(
    user_input="数据安全法什么时候施行?",
    retrieved_contexts=["施行日期相关片段...", "不相关片段..."],
    reference="2021年9月1日",
)
print(result.value)
```

### 指标速查表

| 指标 | 类名 | 需要字段 | 需要 LLM | 需要 Embedding |
|------|------|---------|:---:|:---:|
| Faithfulness | `Faithfulness` | `user_input`, `response`, `retrieved_contexts` | 是 | 否 |
| AnswerRelevancy | `AnswerRelevancy` | `user_input`, `response` | 是 | 是 |
| ContextRecall | `ContextRecall` | `user_input`, `retrieved_contexts`, `reference` | 是 | 否 |
| ContextPrecision | `ContextPrecision` | `user_input`, `retrieved_contexts`, `reference` | 是 | 否 |
| AnswerCorrectness | `AnswerCorrectness` | `user_input`, `response`, `reference` | 是 | 否 |
| SemanticSimilarity | `SemanticSimilarity` | `response`, `reference` | 否 | 是 |
| FactualCorrectness | `FactualCorrectness` | `response`, `reference` | 是 | 否 |
| BleuScore | `BleuScore` | `response`, `reference` | 否 | 否 |
| RougeScore | `RougeScore` | `response`, `reference` | 否 | 否 |

---

## 7. 基本评估流程

### 完整流程(三步走)

```python
# ===== 第一步:准备样本 =====
from ragas import SingleTurnSample, EvaluationDataset

samples = [
    SingleTurnSample(
        user_input="数据安全法是什么?",
        retrieved_contexts=["《数据安全法》是规范数据处理活动的法律..."],
        response="数据安全法是规范数据处理活动的法律。",
        reference="数据安全法是规范数据处理活动的法律。",
    ),
]
dataset = EvaluationDataset(samples)

# ===== 第二步:准备 LLM 和指标 =====
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key="sk-..."))

metrics = [
    Faithfulness(llm=llm),
    ResponseRelevancy(llm=llm, embeddings=emb),
    LLMContextRecall(llm=llm),
]

# ===== 第三步:运行评估 =====
from ragas import evaluate

result = evaluate(dataset=dataset, metrics=metrics)

# 查看结果
print(result)               # 打印各项指标均值
df = result.to_pandas()     # 转成 DataFrame,逐条查看
print(df.head())
```

### 结果对象常用方法

```python
result = evaluate(dataset=dataset, metrics=metrics)

# 1. 打印摘要(各指标均值)
print(result)

# 2. 转 DataFrame(每行一个样本,每列一个指标)
df = result.to_pandas()

# 3. 访问单条结果
print(result.scores)  # 嵌套字典 {sample_index: {metric_name: score}}
```

---

## 8. 单指标评分(独立调用)

不运行批量评估,只对单个样本跑一个指标:

### v0.4 新方式(关键字参数 + ascore)

```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

# 异步
result = await scorer.ascore(
    user_input="问题",
    response="回答",
    retrieved_contexts=["上下文"],
)
print(result.value)   # 分数(float)
print(result.reason)  # 可选解释

# 同步
result = scorer.score(
    user_input="问题",
    response="回答",
    retrieved_contexts=["上下文"],
)
```

### Legacy 方式(SingleTurnSample + single_turn_score)

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
    user_input="问题",
    response="回答",
    retrieved_contexts=["上下文"],
)

scorer = Faithfulness(llm=llm)
score = scorer.single_turn_score(sample)  # 返回 float
```

> **v0.4 变化**: `single_turn_score()` / `single_turn_ascore()` 标记为 deprecated。新方式直接传关键字参数,返回 `MetricResult` 对象(含 `.value` 和 `.reason`)。

---

## 9. 批量评估:evaluate() 与 experiment()

### 9.1 evaluate()(Legacy,简单直接)

```python
from ragas import evaluate

result = evaluate(
    dataset=dataset,          # EvaluationDataset
    metrics=metrics,          # 指标列表
    llm=llm,                  # (可选)全局 LLM
    embeddings=emb,           # (可选)全局 Embedding
    run_config=run_config,    # (可选)运行配置
)

df = result.to_pandas()
```

> v0.4 中 `evaluate()` 标记为 deprecated,但仍可用。适合简单场景。

### 9.2 experiment()(v0.4 推荐)

```python
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from pydantic import BaseModel

# 定义结果结构
class EvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float

# 定义实验函数
@experiment(EvalResult)
async def run_evaluation(row):
    faith_scorer = Faithfulness(llm=llm)
    rel_scorer = AnswerRelevancy(llm=llm, embeddings=emb)

    faith_result = await faith_scorer.ascore(
        response=row.response,
        retrieved_contexts=row.contexts,
    )
    rel_result = await rel_scorer.ascore(
        user_input=row.user_input,
        response=row.response,
    )

    return EvalResult(
        faithfulness=faith_result.value,
        answer_relevancy=rel_result.value,
    )

# 运行实验
exp_results = await run_evaluation(dataset)
```

**experiment() 的优势**:
1. 结构化结果 — 用 Pydantic 模型定义输出
2. 逐行控制 — 可以对每个样本定制评估逻辑
3. 版本追踪 — 可选集成 git 版本记录
4. 迭代友好 — 修改指标后重新运行很方便

---

## 10. 自定义 Prompt

当默认 Prompt 效果不好(如中文场景 JSON 输出不规范),可以继承并修改:

### 10.1 继承并追加约束

```python
from ragas.metrics._faithfulness import StatementGeneratorPrompt, NLIStatementPrompt

# 给原始 Prompt 追加 JSON 输出约束
JSON_CONSTRAINT = """
【输出格式硬性约束】
1. 只输出纯净 JSON,禁止解释
2. 禁止 markdown 代码块
3. 键名必须英文
"""

class StrictStatementGeneratorPrompt(StatementGeneratorPrompt):
    instruction = StatementGeneratorPrompt.instruction + "\n" + JSON_CONSTRAINT

class StrictNLIStatementPrompt(NLIStatementPrompt):
    instruction = NLIStatementPrompt.instruction + "\n" + JSON_CONSTRAINT

# 使用自定义 Prompt 创建指标
from ragas.metrics import Faithfulness

metric = Faithfulness(
    llm=llm,
    statement_generator_prompt=StrictStatementGeneratorPrompt(),
    nli_statements_prompt=StrictNLIStatementPrompt(),
)
```

### 10.2 查看原始 Prompt

```python
from ragas.metrics._faithfulness import StatementGeneratorPrompt

# 查看完整 Prompt 内容
print(StatementGeneratorPrompt.instruction)
```

### 10.3 v0.4 Prompt 适配(多语言)

```python
from ragas.metrics._faithfulness import StatementGeneratorPrompt

# 自动适配到目标语言
StatementGeneratorPrompt.adapt(language="chinese", llm=llm)
```

> 本项目 `evaluator.py` 已使用自定义 Prompt 方式(`StrictStatementGeneratorPrompt` 等)来约束中文 JSON 输出。

---

## 11. RunConfig 配置

`RunConfig` 控制评估的并发、超时、重试:

```python
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=180,        # 单次 LLM 调用超时(秒)
    max_workers=2,      # 并发线程数
    max_retries=1,      # 失败重试次数
)

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config,
)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | 60 | 单次 LLM 调用超时秒数 |
| `max_workers` | 16 | 并发线程数(调低避免限流) |
| `max_retries` | 3 | 失败重试次数 |
| `thread_timeout` | - | 线程级超时 |
| `max_retries_wait` | - | 重试间隔 |
| `log_tenacity` | False | 是否打印重试日志 |

> **调优建议**: MiniMax 等国产 API 限流较严,建议 `max_workers=2`、`timeout=180`。

---

## 12. 结合本项目的实战示例

本项目(`rag-evaluation-system/`)已实现了完整的 RAGAS 评估流程。以下是对应的代码映射:

### 12.1 项目评估流程

```
testset.jsonl (测试集)
    ↓ load_testset()
List[Dict] (字典格式)
    ↓ 逐条调用 Agent API (call_rag_agent)
    ↓ 获取 RAG 系统的回答 + 检索到的上下文
SingleTurnSample (组装评估样本)
    ↓
EvaluationDataset (样本集合)
    ↓ evaluate(dataset, metrics, run_config)
评分结果 DataFrame
    ↓ build_report_dataframe() + save_evaluation_report()
evaluation_report.txt (报告)
```

### 12.2 关键代码对应

**创建样本**(evaluator.py):

```python
sample = SingleTurnSample(
    user_input=question,
    retrieved_contexts=retrieved_contexts,
    response=agent_output["answer"],
    reference=case.get("reference", ""),
)
```

**创建指标 + 自定义 Prompt**(evaluator.py):

```python
metrics = [
    Faithfulness(
        llm=evaluator_llm,
        statement_generator_prompt=StrictStatementGeneratorPrompt(),
        nli_statements_prompt=StrictNLIStatementPrompt(),
    ),
    ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        strictness=1,
        question_generation=StrictResponseRelevancePrompt(),
    ),
    LLMContextRecall(
        llm=evaluator_llm,
        context_recall_prompt=StrictContextRecallPrompt(),
    ),
]
```

**运行评估**(evaluator.py):

```python
run_config = RunConfig(
    timeout=EVAL_TIMEOUT, max_workers=EVAL_MAX_WORKERS, max_retries=EVAL_MAX_RETRIES
)
eval_results = evaluate(
    dataset=EvaluationDataset(samples),
    metrics=metrics,
    run_config=run_config,
)
score_df = eval_results.to_pandas()
```

### 12.3 运行命令

```bash
cd rag-evaluation-system

# 生成测试集(从知识库文档生成问答对)
python main.py generate

# 运行评估(调 Agent + RAGAS 评分 + 报告)
python main.py evaluate

# 完整流程
python main.py full
```

---

## 13. 常见问题与排坑

### Q1: LLM 返回的 JSON 解析失败怎么办?

**原因**: 部分 LLM(如 MiniMax-M3)会带思考块 `<think>` 或 markdown 代码块,破坏 JSON 解析。

**解决**: 参考 `evaluator.py` 的 `sanitize_json_text()` 函数,在 RAGAS 解析前清洗输出:

```python
import re

def sanitize_json_text(text):
    # 去掉 <think>...</think> 思考块
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # 去掉 markdown 代码块标记
    text = re.sub(r"```(?:json)?\s*", "", text)
    # 中文标点转英文
    text = text.replace("，", ",").replace("：", ":").replace("、", ",")
    return text
```

### Q2: 评分全是 0 或 None?

**排查清单**:
1. 检查 `retrieved_contexts` 是否为空列表 → 空上下文会导致 Faithfulness 无法计算
2. 检查 `response` 是否为空字符串
3. 检查 LLM API Key 是否正确、网络是否通
4. 检查 `run_config.timeout` 是否太小(默认60秒,国产 API 建议设180)

### Q3: Embedding 报错 "embed_query not found"?

**原因**: 新版 RAGAS 用 `embed_text`/`embed_texts`,旧版指标可能需要 `embed_query`/`embed_documents`。

**解决**: 用适配器桥接(参考 `evaluator.py` 的 `LegacyEmbeddingAdapter`):

```python
class LegacyEmbeddingAdapter:
    def __init__(self, modern_emb):
        self._emb = modern_emb

    def embed_query(self, text):
        return self._emb.embed_text(text)

    def embed_documents(self, texts):
        return self._emb.embed_texts(texts)
```

### Q4: 评估太慢?

- 降低 `max_workers`(减少并发,避免 API 限流)
- 减少 `strictness` 参数(AnswerRelevancy 的候选问题数)
- 使用更快的 LLM 模型
- 只评估关键指标(Faithfulness + AnswerRelevancy 不需要 reference)

### Q5: 样本太少导致分数不稳定?

建议每个场景准备 20~50 条覆盖边界情况的样本。1~2 条分数波动很大,不具备统计意义。

---

## 14. v0.3 → v0.4 API 变更速查

| 方面 | v0.3(Legacy) | v0.4(新) |
|------|-------------|---------|
| 导入指标 | `from ragas.metrics import Faithfulness` | `from ragas.metrics.collections import Faithfulness` |
| LLM 初始化 | `LangchainLLMWrapper(chat)` | `llm_factory("gpt-4o", client=...)` |
| 单指标评分 | `scorer.single_turn_score(sample)` | `await scorer.ascore(user_input=..., response=...)` |
| 返回类型 | `float` | `MetricResult`(`.value` + `.reason`) |
| 批量评估 | `evaluate(dataset, metrics)` | `@experiment(Model)` 装饰器 |
| Embedding | `LangchainEmbeddingsWrapper(...)` | `embedding_factory(provider, model, client)` |
| Prompt 系统 | Dataclass(`PydanticPrompt`) | 函数式 Prompt + `BasePrompt.adapt()` |
| 字段名 | `question/answer/contexts/ground_truth` | `user_input/response/retrieved_contexts/reference` |

> **本项目状态**: 使用 v0.3 兼容 API(Legacy),在 RAGAS 0.4.3 下功能正常。如需迁移到 v0.4 新 API,参考官方迁移指南: https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/

---

## 参考资源

- 官方文档: https://docs.ragas.io/
- 快速开始: https://docs.ragas.io/en/latest/getstarted/quickstart/
- 指标总览: https://docs.ragas.io/en/latest/concepts/metrics/overview/
- v0.4 迁移指南: https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
- GitHub 仓库: https://github.com/explodinggradients/ragas
