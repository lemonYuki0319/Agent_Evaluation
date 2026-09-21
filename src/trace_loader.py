# trace_loader.py
"""
Trace 日志加载器（STUB）。

目标：通过 otel 平台查询接口，按 trace_id 拉取一次 Agent 运行的
     完整 trace（包括每步检索片段、tool call、LLM 输入输出），
     用于离线复盘与上下文回填。

待实现要点：
    - 调用 TRACE_QUERY_URL（pipeline_cfg.py）
    - 解析 span 树，抽出 retrieval 节点的 contexts
    - 返回 {"contexts": List[str], "spans": List[dict]}
"""
from typing import Dict, List


def load_trace(trace_id: str) -> Dict[str, List]:
    """按 trace_id 拉取 trace 详情。未实现。"""
    raise NotImplementedError("trace_loader.load_trace 尚未实现")
