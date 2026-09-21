# Langfuse 详细教程：Agent 全链路监控 + 提示词对比 + Prompt 打分

> 基于 Langfuse 官方文档（https://langfuse.com/docs）整理，结合本 RAG 项目落地。
> SDK 版本：Python SDK v3（OpenTelemetry-based）

---

## 1. Langfuse 是什么

Langfuse 是一个开源的 AI 工程平台（[GitHub](https://github.com/langfuse/langfuse)），核心能力：

| 模块 | 解决的问题 |
|---|---|
| **Observability（可观测性）** | Agent 全链路追踪：每次请求的 prompt、模型回复、工具调用、检索步骤、token 用量、延迟 |
| **Prompts（提示词管理）** | 提示词版本化、灰度发布、按版本追踪指标、无需重新部署即可更新 |
| **Evaluation（评估）** | LLM-as-a-Judge、人工标注、用户反馈、自定义评分；离线实验 + 在线监控 |

三种部署方式：
- **Langfuse Cloud**（EU / US region，免费层，零基础设施）
- **Self-hosting**（自托管，开源，Docker Compose / K8s）
- **公开 Demo 项目**（只读访问体验）

---

## 2. 快速开始

### 2.1 安装 Python SDK v3

```bash
pip install langfuse
```

### 2.2 配置凭据

`.env`：

```
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 2.3 初始化 Client

```python
from langfuse import get_client

langfuse = get_client()  # 单例

# 验证连接（不要在生产里调，会增加同步延迟）
if langfuse.auth_check():
    print("Langfuse connected")
```

### 2.4 关键配置项

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `LANGFUSE_HOST` | API 地址 | `https://cloud.langfuse.com` |
| `LANGFUSE_TIMEOUT` | 请求超时秒 | `5` |
| `LANGFUSE_FLUSH_AT` | 批量发送的 span 数 | `512` |
| `LANGFUSE_FLUSH_INTERVAL` | 批量发送间隔秒 | `5` |
| `LANGFUSE_TRACING_ENVIRONMENT` | 环境名（development/staging/production） | `default` |
| `LANGFUSE_RELEASE` | 应用版本/hash | — |
| `LANGFUSE_SAMPLE_RATE` | 采样率 0.0~1.0 | `1.0` |
| `LANGFUSE_TRACING_ENABLED` | 关掉则所有调用变 no-op | `True` |

---

## 3. Agent 全链路监控（Tracing）

### 3.1 数据模型

Langfuse 把数据分为三层：

```
Session（会话，可选，用于多轮对话）
  └── Trace（一次完整请求）
        └── Observation（具体步骤：LLM 调用 / 工具调用 / 检索）
              └── Observation（嵌套子步骤）
```

- **Observation**：应用执行的单个步骤
  - 类型：`span`（通用步骤）、`generation`（LLM 调用）、`event`（无耗时事件）
- **Trace**：一次完整请求，所有共享 `trace_id` 的 observation 集合
- **Session**：可选，把多个 trace 聚合为一个会话（多轮对话场景）

Trace 级属性（`user_id` / `session_id` / `tags` / `metadata`）会自动写到该 trace 下每条 observation 上，便于聚合查询。

### 3.2 三种接入方式

#### 方式一：`@observe` 装饰器（最简）

```python
from langfuse import observe, get_client

@observe()
def my_data_processing(data, parameter):
    return {"processed": data, "status": "ok"}

@observe(name="llm-call", as_type="generation")
async def my_async_llm_call(prompt_text):
    return "LLM response"

my_data_processing(...)
langfuse = get_client()
langfuse.flush()  # 短生命周期应用必须 flush
```

装饰器自动捕获函数名、入参、返回值、执行时间；嵌套调用自动构成父子 observation。

#### 方式二：Context Manager（推荐，控制更细）

```python
from langfuse import get_client
langfuse = get_client()

with langfuse.start_as_current_span(
    name="user-request-pipeline",
    input={"user_query": "Tell me a joke"},
) as root_span:
    # 设置 trace 级属性
    root_span.update_trace(
        user_id="user_123",
        session_id="session_abc",
        tags=["experimental", "comedy"],
    )

    # 嵌套 generation（LLM 调用）
    with langfuse.start_as_current_generation(
        name="joke-generation",
        model="gpt-4o",
        input=[{"role": "user", "content": "Tell me a joke"}],
        model_parameters={"temperature": 0.7},
    ) as generation:
        response = "Why did the chicken..."
        generation.update(
            output=response,
            usage_details={"input_tokens": 10, "output_tokens": 25},
        )

    root_span.update(output={"final_joke": response})
```

#### 方式三：Manual（最细粒度，需手动 `.end()`）

```python
span = langfuse.start_span(name="my-span")
span.update(output="...")
span.end()  # 必须手动结束
```

### 3.3 多轮对话（Sessions）

把多轮 trace 归到一个 session 即可：

```python
def chat_round(thread_id: str, user_msg: str):
    with langfuse.start_as_current_span(name=f"chat-round-{thread_id}") as span:
        # 关键：所有同 thread_id 的 trace 会自动聚合到一个 session
        span.update_trace(
            session_id=thread_id,
            user_id="user_42",
        )
        # ... 调用 LLM
```

Langfuse UI 的 Sessions 视图会显示完整的对话时间线。

### 3.4 自定义 Trace ID（分布式追踪）

如果你已有外部 trace_id（如业务侧的 `conversation_id`），可以转换成合法的 W3C trace ID：

```python
from langfuse import Langfuse, observe

@observe()
def handle_request(user_msg, *, langfuse_trace_id=None):
    # 业务逻辑
    ...

# 把业务 conversation_id 转成 Langfuse trace_id
external_id = "4622064354806726656"
trace_id = Langfuse.create_trace_id(seed=external_id)
handle_request(user_msg, langfuse_trace_id=trace_id)
```

线上 trace 与离线评测共用同一 ID，便于回溯。

### 3.5 第三方库自动埋点

Langfuse 通过 OpenTelemetry 自动捕获任何 OTel-instrumented 库的 span：

```python
# OpenAI 自动埋点
from langfuse.openai import openai  # 替换 import openai
# 后续 client.chat.completions.create 都会被自动 trace
```

```python
# LangChain 自动埋点
from langfuse.callback import CallbackHandler
handler = CallbackHandler()
chain.invoke({"input": "..."}, config={"callbacks": [handler]})
```

任何 OTel 原生 instrumented 库（requests / httpx / redis 等）也会自动并入 trace 树。

### 3.6 Token 与成本追踪

在 generation 上传 `usage_details`：

```python
generation.update(
    output=response_text,
    usage_details={"input_tokens": 10, "output_tokens": 25},
)
```

Langfuse 后台按模型价格表自动算成本，UI 显示每条 trace 总成本。

### 3.7 短生命周期应用必须 flush

评测脚本是一次性进程，退出前必须 flush：

```python
from langfuse import get_client
langfuse = get_client()
# ... 所有 trace 写完
langfuse.flush()  # 强制发送缓冲区
```

否则后台线程还没发完，进程就退出了，会丢 trace。

### 3.8 按属性过滤 Trace

可用属性（在 UI 或 API filter 里）：

| 属性 | 用途 |
|---|---|
| `environment` | 区分 production / staging / development |
| `tags` | 自由标签，按 feature / endpoint / workflow 分类 |
| `user_id` | 终端用户追踪 |
| `metadata` | 灵活 key-value，自定义信息 |
| `release` / `version` | 应用版本追踪 |

---

## 4. 提示词管理（Prompt Management）

### 4.1 工作流

1. **创建/更新** 提示词（UI / SDK）
2. **promote** 某版本到 production
3. 应用侧 `get_prompt()` 取当前生产版本
4. `prompt.compile()` 注入变量
5. 把 prompt 对象附在 generation 上，按版本追踪指标

### 4.2 创建提示词版本

```python
langfuse.create_prompt(
    name="legal-qa-system",
    prompt="你是企业知识库顾问。仅依据以下知识库内容回答：\n{{retrieved_context}}\n\n用户问题：{{user_input}}",
    config={
        "model": "deepseek-chat",
        "temperature": 0.0,
        "provider": "deepseek",
    },
    labels=["production"],  # 标记为生产版本
)
```

更新时用相同 `name` 创建新版本，旧版本仍可回滚。

### 4.3 取生产版本并注入变量

```python
# 取当前 production 版本
prompt = langfuse.get_prompt("legal-qa-system")

# 取指定版本（如回归测试）
prompt_v3 = langfuse.get_prompt("legal-qa-system", version=3)

# 注入变量
compiled = prompt.compile(
    retrieved_context="[文档1]...[文档2]...",
    user_input="数据资源如何入表？",
)
```

### 4.4 把 prompt 关联到 generation（按版本追踪指标）

```python
with langfuse.start_as_current_generation(
    name="legal-qa-llm",
    model=prompt.config["model"],
    input=compiled,
) as generation:
    answer = call_llm(compiled)
    generation.update(
        output=answer,
        usage_details=token_usage,
        prompt=prompt,  # 关键：把 prompt 对象附上
    )
```

UI 里就能按 `prompt name + version` 切片看延迟、token、score 分布。

### 4.5 性能注意

`get_prompt()` 每次都打 Langfuse API，会增加约 170ms 延迟。Langfuse 计划加缓存，当前对热路径需要自己缓存（如本地 LRU 5 秒过期）。

---

## 5. 提示词对比（Datasets + Experiments）

### 5.1 创建 Dataset

```python
langfuse.create_dataset(
    name="legal-rag-benchmark",
    description="审计法律知识库 RAG 评测集",
    metadata={"owner": "zhangsan", "type": "regression"},
)
```

### 5.2 上传 Dataset Items

```python
langfuse.create_dataset_item(
    dataset_name="legal-rag-benchmark",
    input={
        "messages": [
            {"role": "user", "content": "你好，我叫张三"},
            {"role": "assistant", "content": "你好，张三。"},
            {"role": "user", "content": "我叫什么？另外审计证据如何判断？"},
        ]
    },
    expected_output={"text": "你叫张三。\n\n关于审计证据，依据《国家审计准则》..."},
    metadata={"topic": "multi-turn"},
)
```

也可以从生产 trace 转换为 dataset item（bad case 回流）：

```python
langfuse.create_dataset_item(
    dataset_name="legal-rag-benchmark",
    input={"text": "..."},
    expected_output={"text": "..."},
    source_trace_id="<trace_id>",          # 关联到原 trace
    source_observation_id="<observation_id>",  # 可选
)
```

### 5.3 Schema 校验

```python
langfuse.create_dataset(
    name="qa-conversations",
    input_schema={
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            }
        },
        "required": ["messages"],
    },
    expected_output_schema={
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"],
    },
)
```

不合规 item 会被拒绝，保证数据集质量。

### 5.4 版本化（按时间点快照）

每次 add/update/delete 都产生新版本，可按时间点取快照：

```python
from datetime import datetime, timezone

version_ts = datetime(2025, 12, 15, 6, 30, 0, tzinfo=timezone.utc)
dataset_v1 = langfuse.get_dataset(
    name="legal-rag-benchmark",
    version=version_ts,
)
```

回归测试时锁定版本，保证不同 run 的数据集一致。

### 5.5 跑实验（SDK）

```python
dataset = langfuse.get_dataset("legal-rag-benchmark")

def my_app(*, item, **kwargs):
    # 你的 RAG agent 调用
    return {"response": call_legal_agent(item["input"]["messages"])}

# 实验：每个 item 跑一次，自动产出 trace + 自动跑 evaluator
result = dataset.run_experiment(
    name="prompt-v3-baseline",
    description="换 GPT-4o-mini + 新 system prompt",
    task=my_app,
)
```

实验产出的 trace 自动带 `experiment_id`，UI 上直接对比多次实验的指标均值与 bad case 列表。

### 5.6 按版本快照跑实验

```python
versioned_dataset = langfuse.get_dataset(
    "legal-rag-benchmark",
    version=datetime(2025, 12, 15, 6, 30, 0, tzinfo=timezone.utc),
)

result = versioned_dataset.run_experiment(
    name="baseline-v1",
    description="Running on dataset v1",
    task=my_app,
)
```

可重现历史结果，便于对比"模型在 v1 数据集 vs v2 数据集"的表现。

### 5.7 从生产 trace 批量入集

UI 操作：`Observations` 表 → 多选 → `Actions` → `Add to dataset`，灵活字段映射。适合从生产线上挑 bad case 加入回归集。

---

## 6. Prompt 打分（Evaluation）

### 6.1 Score 数据模型

所有评分统一为 `score` 对象，三种类型：

| 类型 | 用途 | 例子 |
|---|---|---|
| **Numeric** | 连续打分 | helpfulness 0~1, faithfulness 0~1 |
| **Categorical** | 离散标签 | `correct` / `partially_correct` / `incorrect` |
| **Boolean** | 二元判断 | 是否越狱、是否超出知识库、是否违规 |

Score 可挂在 trace 或 observation 上，UI / API 都可访问。

### 6.2 四种评测方法

| 方法 | 触发方式 | 适用场景 |
|---|---|---|
| **LLM-as-a-Judge** | 自动按 rule 触发 / 离线实验 | 大规模自动评测，可重复 |
| **人工标注**（Annotation Queue） | 标注员在 UI 操作 | 建立基线、对齐 LLM-as-Judge |
| **用户反馈** | 前端埋点 / SDK | 真实用户体验信号 |
| **自定义 SDK score** | 调 `langfuse.score()` | 规则式检查、外部评测器结果回写 |

### 6.3 通过 SDK 写入自定义 score

最常用——把外部评测框架（如 RAGAS / DeepEval）的分数回写到 Langfuse：

```python
from langfuse import get_client
langfuse = get_client()

# 把 RAGAS 的 faithfulness 分数回写到对应 trace
langfuse.score(
    trace_id="<trace_id>",
    name="ragas_faithfulness",
    value=0.85,
    comment="RAGAS faithfulness, 自动评测",
    data_type="NUMERIC",
)

# 布尔类型（如：是否命中知识库）
langfuse.score(
    trace_id="<trace_id>",
    name="in_knowledge_base",
    value=True,
    data_type="BOOLEAN",
)

# 类别类型（如：回答质量分级）
langfuse.score(
    trace_id="<trace_id>",
    name="answer_quality",
    value="partially_correct",
    data_type="CATEGORICAL",
)
```

挂到 observation 上（更细粒度）：

```python
langfuse.score(
    trace_id="<trace_id>",
    observation_id="<observation_id>",  # 如某个 retrieval span
    name="retrieval_relevance",
    value=0.7,
    data_type="NUMERIC",
)
```

### 6.4 配置 LLM-as-a-Judge evaluator

#### 步骤一：配置 LLM Connection

`项目设置 → LLM Connection`，添加 OpenAI 兼容端点（如 DeepSeek / 通义 / 自建 OpenAI-compatible API）。

#### 步骤二：创建 Evaluator

`Evaluators → New evaluator → LLM-as-a-Judge`，配置：

- **Judge model**：选择项目默认或专用模型
- **Messages**：
  - `system`：rubric 与约束（必须是第一条）
  - `user`：评测任务 + `{{variables}}`
  - `assistant`（可选）：few-shot 示例
- **Score type**：Numeric / Categorical / Boolean
- **Variable mapping**：把 prompt 里的 `{{input}}` `{{output}}` `{{ground_truth}}` 映射到 observation 字段

典型 Judge prompt 模板：

```
[System]
你是一个严格的 Agent 评估专家。打分范围 0~1：1.0=完全对齐，0.7=基本合格，<0.5=明显偏差。

[User]
用户输入：{{input}}
Agent 回答：{{output}}
期望回答：{{ground_truth}}

请综合判断：1) 意图理解 2) 关键点覆盖 3) 行为合理性
输出 JSON：{"score": float, "reason": "..."}
```

#### 步骤三：测试 evaluator

右侧选 sample observation → 运行 → 查看 score + reasoning → 迭代 prompt 直到满意。

#### 步骤四：保存 + 关联 rule

- **在线（production）**：创建 rule，filter 命中后自动跑 evaluator
- **离线（experiments）**：在实验里直接选 evaluator 跑

#### 步骤五：在线 rule 的 filter 示例

`isRootObservation = true` + `type = generation` + `name contains "legal-qa-llm"`，只评"根节点 LLM 输出"，避免每条 trace 跑多个内部 LLM 调用都被评分。

### 6.5 在线 vs 离线评测

| 维度 | 在线（observation-level） | 离线（experiments） |
|---|---|---|
| **数据** | 实时生产 trace | 受控数据集 |
| **触发** | rule 自动 | 手动 / SDK / UI |
| **延迟** | 秒级异步，每分钟数千条 | 实验级，按数据集规模 |
| **用途** | 监控、告警、趋势 | 模型对比、prompt 对比、回归 |

推荐组合：**开发期**用 experiments 做决策，**上线后**用 observation-level evaluator 做监控。

### 6.6 调试 LLM-as-Judge 执行

每次 evaluator 执行本身也是一条 trace，环境标记为 `langfuse-llm-as-a-judge`。Tracing 表里 filter 这个 environment 就能看到所有 evaluator 的 prompt、模型回复、token 用量。

### 6.7 API 程序化配置

```http
POST /api/public/evaluators
{
  "name": "faithfulness-judge",
  "prompt_config": {...},
  "score_type": "NUMERIC",
  ...
}

POST /api/public/evaluation-rules
{
  "evaluator_id": "<id>",
  "filters": [
    {"type": "boolean", "column": "isRootObservation", "operator": "=", "value": true}
  ],
  "sample_rate": 1.0,
}
```

适合把评测配置纳入版本控制 / CI 部署。

---

## 7. 完整 Agent 评测链路（结合本 RAG 项目）

把上述能力串起来，本 RAG 项目的 Agent 评测链路：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 业务侧 Agent（legal_agent）每次 SSE stream 自动上报 trace 到  │
│    Langfuse，含 model/retriever/tool 各 observation              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 生产 trace → 选 bad case → 加入 dataset                       │
│    langfuse.create_dataset_item(source_trace_id=...)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 评测脚本（main.py）跑 dataset.run_experiment，产出新 trace      │
│    并自动关联实验 ID                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Evaluator 自动跑：                                            │
│    - LLM-as-Judge faithfulness（在线 rule 触发）                 │
│    - DeepEval/RAGAS 分数回写：langfuse.score(trace_id=...)        │
│    - 人工标注（Annotation Queue）抽样复核                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. UI 对比实验：prompt v1 vs v2 vs v3 的指标分布、bad case 列表  │
│    按 prompt version 切片看延迟 / token / score                  │
└─────────────────────────────────────────────────────────────────┘
```

### 落地清单（接入本项目的最小步骤）

1. **业务侧 legal_agent**：在 `audit_agent/api/legal_agent/stream_request.py` 的 `_stream_query` 里包一层 `langfuse.start_as_current_span`，把 SSE 事件按节点拆 observation。或直接用 Langfuse 的 LangGraph / OpenAI 集成自动埋点。
2. **评测脚本 main.py**：在 `_run_agent_batch` 里把每条 record 的 `trace_id` 与 Langfuse 对齐（用 `Langfuse.create_trace_id(seed=conversation_id)`）。
3. **DeepEval 评测器**：在 `run_deepeval` 出分后，调 `langfuse.score(trace_id=..., name=metric_name, value=score)` 把分数回写。
4. **数据集**：把 `data/raw/testset_agent.jsonl` 通过 `langfuse.create_dataset_item` 导入。
5. **prompt**：把当前 system prompt 用 `langfuse.create_prompt` 纳入版本管理，agent 调用时 `get_prompt` 拿生产版本。

---

## 8. 自托管 vs 云

| 维度 | Langfuse Cloud | Self-hosting |
|---|---|---|
| 数据驻留 | EU / US region | 自定 |
| 维护 | 零 | Docker Compose / K8s |
| 成本 | 免费层 + 按用量计费 | 仅服务器成本 |
| 升级 | 自动 | 手动 |
| 安全 | 项目级 RBAC、SSO | 完全自有 |

合规要求高 / 数据不出本司 → self-hosting；快速试用 / 团队小 → Cloud 免费层。

---

## 9. 关键参考链接

- 官方文档：https://langfuse.com/docs
- Python SDK v3：https://langfuse.com/docs/sdk/python/decorators
- LLM-as-a-Judge：https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge
- Datasets & Experiments：https://langfuse.com/docs/evaluation/experiments/datasets
- Prompt Management：https://langfuse.com/docs/prompts
- Tracing Core Concepts：https://langfuse.com/docs/observability/data-model
- GitHub：https://github.com/langfuse/langfuse
- API Reference：https://api.reference.langfuse.com

---

## 10. 常见坑

| 坑 | 解法 |
|---|---|
| 评测脚本退出后 trace 丢失 | 进程结束前必须 `langfuse.flush()` |
| `get_prompt()` 拖慢热路径 | 本地 LRU 缓存，5~10 秒过期 |
| LLM-as-Judge 评分不稳 | system message 写死 rubric + 加 few-shot assistant 示例 |
| 在线 evaluator 不触发 | rule filter 加 `isRootObservation=true`，只评根节点 |
| 多轮对话 trace 散开 | `span.update_trace(session_id=thread_id)` 关联 |
| 多模态 dataset 不被实验支持 | 用 SDK 跑（UI 暂不支持带 media 的实验） |
| Trace 级 evaluator 已废弃 | 迁移到 observation-level（v4 cutover 2026-11-16 后停用） |
