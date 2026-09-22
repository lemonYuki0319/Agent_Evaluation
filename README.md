# RAG 混合评测系统（DeepEval + RAGAS）

用 DeepEval + RAGAS 双框架对 Agent 流式接口做 RAG 评测，统一用 MiniMax-M2.5 作为 Judge LLM。

## 目录结构

```
RAG/
├── audit_agent/          # Agent SSE 接口封装（保留）
├── config/               # 配置
│   ├── metric_cfg.py     # Judge 模型、指标开关、阈值
│   └── pipeline_cfg.py   # Agent 地址、并发、路径
├── data/
│   ├── raw/              # 原始评测集（user_input + ground_truth）
│   ├── processed/        # 运行后生成（Agent 输出缓存）
│   └── badcase/          # 失败样本沉淀
├── src/
│   ├── agent_client.py   # 调 Agent，返回 answer + trace_id
│   ├── retriever_client.py  # 检索函数（STUB）
│   ├── trace_loader.py   # Trace 加载（STUB）
│   ├── evaluator_ragas.py   # RAGAS 评测
│   ├── evaluator_deepeval.py # DeepEval 评测
│   └── report_builder.py # 报告生成
├── docs/                # 官方教程
├── runs/                # 每次实验输出（按 run_id）
├── logs/                # 运行日志
└── main.py              # 入口脚本
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 填入 API Key
   ```

3. 小样本冒烟：
   ```bash
   python main.py --max-cases 3
   ```

4. 全量运行：
   ```bash
   python main.py
   ```

## 输出说明

每次运行在 `runs/run_YYYYMMDD_HHMMSS/` 下生成：
- `details.csv`：每条样本一行，含双框架所有分数
- `summary.json`：各指标均值、通过率
- `bad_cases.jsonl`：低于阈值的样本明细

## 指标说明

| 框架 | 指标 | 说明 |
|------|------|------|
| RAGAS | faithfulness | 忠实度（回答是否基于上下文） |
| RAGAS | context_recall | 上下文召回（金标要点是否被检索到） |
| RAGAS | context_precision | 上下文精确度（相关片段排前） |
| DeepEval | faithfulness | 同上 |
| DeepEval | contextual_recall | 同上 |
| DeepEval | contextual_precision | 同上 |
| DeepEval | geval | 语义对齐度（自定义 GEval） |

> `answer_relevancy` 默认关闭（需 embeddings，MiniMax 非完全 OpenAI 兼容）。

## 已知限制

- `src/retriever_client.py` 与 `src/trace_loader.py` 为 STUB，待实现
- Judge 模型统一用 `Deepseek`（不用MiniMax，避免思考块破坏 JSON）
# Agent_Evaluation
