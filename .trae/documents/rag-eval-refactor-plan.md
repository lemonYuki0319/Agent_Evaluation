# RAG 评测系统重构方案 (Hybrid DeepEval + RAGAS)

## Context

当前项目存在两套独立的评测代码：`rag-evaluation-system/`（RAGAS，单文件 732 行，含大量 MiniMax JSON 清洗逻辑）和 `rag-evaluation-system/deepeval/`（DeepEval demo），两者割裂、代码冗长。用户要求重构为 **DeepEval + RAGAS 混合评测**，目录扁平化，代码精简易懂，接口封装保留在 `audit_agent/`。

核心简化：统一用 `MiniMax-M2.5`（不破坏 JSON），**砍掉全部 JSON 清洗/Prompt 重写/system 注入/embeddings 适配器**，用各框架默认行为。

---

## 目标结构

```
e:\python_project\RAG\
├── README.md                       (新建)
├── requirements.txt                (新建)
├── .env / .env.example             (新建)
├── .gitignore                      (新建)
├── audit_agent/                    (保留原样,接口封装)
├── docs/                           (新建,教程)
│   ├── RAGAS官方教程.md            (从根目录移入)
│   └── DeepEval官方教程.md         (从 rag-evaluation-system/deepeval/ 移入)
├── config/
│   ├── __init__.py
│   ├── metric_cfg.py               (judge 模型 + 指标开关 + 阈值)
│   └── pipeline_cfg.py             (Agent 地址 + 并发 + 路径)
├── data/
│   ├── raw/testset_agent.jsonl     (从 testset.jsonl 迁移,字段 user_input/ground_truth)
│   ├── processed/                  (空)
│   └── badcase/                    (空)
├── src/
│   ├── __init__.py
│   ├── agent_client.py             (调 Agent SSE,返回 answer+trace_id)
│   ├── retriever_client.py         (STUB)
│   ├── trace_loader.py             (STUB)
│   ├── evaluator_ragas.py          (RAGAS 3 指标,answer_relevancy 默认关)
│   ├── evaluator_deepeval.py       (DeepEval 4+1 指标 + GEval)
│   └── report_builder.py           (details.csv + summary.json + bad_cases.jsonl)
├── runs/                           (空,按 run_id 子目录输出)
├── logs/                           (空)
└── main.py                         (入口)
```

**删除清单**（新流程跑通后）：`rag-evaluation-system/`、根 `test.py`、根 `RAGAS官方教程.md`、`.deepeval/`

---

## 各文件实施要点

### 配置层

**`config/metric_cfg.py`**
- `JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_EMBED_MODEL`（`os.getenv`，默认 `MiniMax-M2.5`）
- `DEFAULT_THRESHOLD = 0.7`
- `RAGAS_METRICS_ENABLED = {"faithfulness": True, "answer_relevancy": False, "context_recall": True, "context_precision": True}`（answer_relevancy 关：需 embeddings，MiniMax 非完全 OpenAI 兼容）
- `DEEPEVAL_METRICS_ENABLED = {"faithfulness": True, "answer_relevancy": False, "contextual_recall": True, "contextual_precision": True, "geval": True}`
- `get_judge_env() -> dict`：返回 `{"OPENAI_API_KEY": ..., "OPENAI_API_BASE": ...}` 供 DeepEval 注入环境变量

**`config/pipeline_cfg.py`**
- `AGENT_BASE_URL / AGENT_AUTH_TOKEN`（`os.getenv`）
- `EVAL_MAX_WORKERS=2 / EVAL_TIMEOUT=180 / RETRIEVER_TOP_K=5 / TRACE_QUERY_URL=""`
- 路径常量（`pathlib.Path`）：`PROJECT_ROOT / RAW_DATA_FILE / PROCESSED_DIR / BADCASE_DIR / RUNS_DIR / LOGS_DIR`
- `ensure_dirs()`、`new_run_dir() -> Path`

### 数据层

**数据迁移** `testset.jsonl → data/raw/testset_agent.jsonl`
- 字段映射：`question → user_input`，`reference → ground_truth`，丢弃其余字段
- 格式：每行 `{"user_input": "...", "ground_truth": "..."}`

### 接口层

**`src/agent_client.py`**（< 60 行）
- 复用 `audit_agent.stream_request` 的 `BASE_URL / AUTH_TOKEN / HEADERS` 常量
- `call_agent(question, timeout=120) -> dict`：自实现 SSE 调用，先 `create_conversation` 拿 thread_id（=trace_id），再流式取 model 节点 answer
- 返回 `{"answer": str, "trace_id": str, "retrieved_files": []}`
- `retrieved_files` 暂空，等 retriever_client 实现
- 注意：`main.py` 顶部加 `sys.path.insert(0, str(Path(__file__).resolve().parent))` 保证 `audit_agent` 可导入

**`src/retriever_client.py` / `src/trace_loader.py`**（STUB）
- 文件头注释说明目标 + 待实现要点
- 函数体 `raise NotImplementedError(...)`

### 评测层

**`src/evaluator_ragas.py`**（< 80 行）
- `build_llm()`：`LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, base_url=..., api_key=..., temperature=0))`
- `build_metrics()`：按开关构造 `Faithfulness / LLMContextRecall / LLMContextPrecisionWithReference`（answer_relevancy 默认关）
- `build_samples(records)`：`SingleTurnSample(user_input, response=actual_output, retrieved_contexts or ["(无检索上下文)"], reference=ground_truth)`
- `run_ragas(records) -> list[dict]`：`evaluate(dataset=EvaluationDataset(samples), metrics=..., run_config=RunConfig(...))` → 抽分数成 `{"index": i, "ragas": {"faithfulness": 0.85, ...}}`，单条异常置 None

**`src/evaluator_deepeval.py`**（< 100 行）
- 模块顶部 `os.environ.update(get_judge_env())` 注入环境变量
- `build_metrics()`：`FaithfulnessMetric / ContextualRecallMetric / ContextualPrecisionMetric / GEval`（answer_relevancy 默认关；GEval criteria 复用旧 `deepeval/test.py` 文本）
- `build_test_cases(records)`：`LLMTestCase(input, actual_output, expected_output, retrieval_context)`
- `run_deepeval(records) -> list[dict]`：`evaluate(test_cases, metrics)` → 抽 `metrics_data` 成 `{"index": i, "deepeval": {"faithfulness": (score,success,reason), ...}}`

### 报告层

**`src/report_builder.py`**（< 120 行）
- `build_report(records, ragas_scores, deepeval_scores, run_dir) -> dict`
- 输出三件套到 `runs/<run_id>/`：
  1. `details.csv`：每行一条样本，列含 user_input/ground_truth/actual_output/trace_id + 双框架所有分数
  2. `summary.json`：`{run_id, total, per_metric: {name: {mean, pass_rate, n}}}`
  3. `bad_cases.jsonl`：任一指标 < DEFAULT_THRESHOLD 的样本明细

### 入口

**`main.py`**（< 100 行）
- `load_dotenv()` → `ensure_dirs()` → 解析 `--data / --max-cases / --skip-ragas / --skip-deepeval`
- `load_dataset(path) -> list[dict]`：读 JSONL
- 循环每条：`call_agent(user_input)` → 填 `actual_output / trace_id / retrieved_contexts=[]`，边调边落 `data/processed/agent_outputs_<run_id>.jsonl`
- `run_ragas(records)` → `run_deepeval(records)` → `build_report(...)` → 打印 summary

---

## 数据流

```
data/raw/testset_agent.jsonl  ({user_input, ground_truth})
    │
    ▼ call_agent()
records: [{user_input, ground_truth, actual_output, trace_id, retrieved_contexts:[]}]
    │
    ├──► run_ragas(records)   → [{index, ragas:{faithfulness, context_recall, context_precision}}]
    └──► run_deepeval(records)→ [{index, deepeval:{faithfulness, contextual_recall, contextual_precision, geval}}]
    │
    ▼ build_report()
runs/<run_id>/{details.csv, summary.json, bad_cases.jsonl}
```

---

## 实施顺序

1. 建目录骨架（docs/ config/ data/{raw,processed,badcase}/ src/ runs/ logs/）
2. 移教程到 docs/，迁移测试数据
3. 写 requirements.txt / .env.example / .env / .gitignore
4. 写 config/（metric_cfg.py, pipeline_cfg.py, __init__.py）
5. 写 src/__init__.py + 两个 STUB（retriever_client.py, trace_loader.py）
6. 写 src/agent_client.py
7. 写 src/evaluator_ragas.py
8. 写 src/evaluator_deepeval.py
9. 写 src/report_builder.py
10. 写 main.py + README.md
11. 冒烟测试 `python main.py --max-cases 2`
12. 删除旧目录（rag-evaluation-system/、根 test.py、根 RAGAS官方教程.md）

---

## 简化对照

| 旧 evaluator.py 复杂点 | 新方案 |
|---|---|
| sanitize_json_text + _CN_KEY_MAP（~100 行） | 删除，用 M2.5 |
| CleanJsonLLMWrapper（~40 行） | 删除，直接 LangchainLLMWrapper |
| Strict*Prompt 子类（~30 行） | 删除，用默认 prompt |
| SystemPrefixedChat（~30 行） | 删除 |
| MiniMaxEmbeddings 适配器（~70 行） | 删除，answer_relevancy 默认关 |
| SSE retrieved_documents 解析（~50 行） | 删除，retrieved 留空 |
| resolve_contexts 多级 fallback | 删除 |
| load_testset 旧格式兼容 | 删除，干净 JSONL |

新 src/ 总计约 400 行，远小于旧单文件 732 行。

---

## 风险与对策

1. **RAGAS/DeepEval answer_relevancy 依赖 embeddings**：MiniMax embeddings 非完全 OpenAI 兼容 → 两个框架的 answer_relevancy 默认关闭，留开关供后续启用
2. **audit_agent 无 `__init__.py`**：隐式命名空间包 → main.py 顶部 `sys.path.insert(0, project_root)`
3. **MiniMax-M2.5 偶发非纯 JSON**：`run_ragas` / `run_deepeval` 包 try/except，单条失败置 None，不阻塞整体
4. **SSE 偶发不返回 end 事件**：沿用旧 stream_request 的 warn 行为，EVAL_TIMEOUT 兜底

---

## 验证方式

1. **接口层冒烟**：`python -c "from src.agent_client import call_agent; print(call_agent('数据安全是什么?'))"`
2. **RAGAS 冒烟**：造 2 条假 record，`run_ragas(records)` 应返回分数列表
3. **DeepEval 冒烟**：同上，`run_deepeval(records)`
4. **端到端**：`python main.py --max-cases 2`，检查 `runs/run_*/` 下三件套文件
5. **全量**：`python main.py`
