# retriever_client.py
"""
检索函数封装（STUB）。

目标：给定 question，调用独立检索服务返回 top_k 上下文片段，
     用于在 Agent 未返回 retrieved_documents 时回填 retrieved_contexts。

待实现要点：
    - 调用配置中的检索服务地址（RETRIEVER_* 环境变量）
    - 返回 List[str]，每项为一段检索正文
    - 失败时返回空列表，由上层决定是否用占位
"""
from typing import List


def retrieve(question: str, top_k: int = 5) -> List[str]:
    """检索相关上下文片段。未实现。"""
    raise NotImplementedError("retriever_client.retrieve 尚未实现")
