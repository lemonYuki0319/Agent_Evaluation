# stream_request.py
"""Legal Agent SSE 流式客户端。

拆分为三步：创建会话 / 流式提问 / 解析回答，
通过 @register_agent("legal_agent") 注册到工厂。
"""
import json
import time

import requests

from audit_agent.api.legal_agent.login import login
from audit_agent.registry import register_agent
from config.logger import get_logger
from config.pipeline_cfg import (
    AGENT_ACCOUNT,
    AGENT_BASE_URL,
    AGENT_PASSWORD,
)


def _auth_headers() -> dict:
    """登录并组装鉴权头。"""
    return {
        "Authorization": f"Bearer {login(AGENT_ACCOUNT, AGENT_PASSWORD)}",
        "Content-Type": "application/json",
    }


def _create_conversation(headers: dict, retries: int = 1) -> str:
    """创建会话，返回 conversation_id。失败按 retries 重试，应对瞬时网络抖动。"""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{AGENT_BASE_URL}/history/conversation/create",
                headers=headers,
                json={"chat_type": "100", "title": "评测"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["data"]["conversation_id"]
        except Exception:
            if attempt < retries:
                time.sleep(3)
            else:
                raise


def _stream_query(headers: dict, thread_id: str, messages: list, timeout: int) -> list:
    """发起流式提问，返回 SSE 事件列表。

    messages 为完整对话历史（[{role, content}, ...]，最后一条为 user），
    交由 LangGraph runtime 覆盖式重建上下文后生成最后一条 user 的回复。
    """
    payload = {
        "input": {
            "messages": messages,
            "task_type": "knowledge",
        },
        "config": {
            "recursion_limit": 100,
            "configurable": {"thread_id": thread_id},
        },
        "stream_mode": ["updates"],
        "stream_subgraphs": True,
    }
    events = []
    with requests.post(
        f"{AGENT_BASE_URL}/stream",
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    pass
    return events


def _extract_answer(events: list) -> str:
    """从 model 节点的 messages 中提取回答文本。"""
    for event in events:
        model = event.get("model")
        if isinstance(model, dict):
            messages = model.get("messages", [])
            if messages and messages[0].get("content"):
                return messages[0]["content"]
    return ""


@register_agent("legal_agent")
def stream_chat(prompt, timeout: int = 120) -> dict:
    """Legal Agent 流式提问，返回 {answer, trace_id, retrieved_contexts}。

    Args:
        prompt: 单轮为 str（问题文本）；多轮为 list[dict]（完整对话历史
                [{role, content}, ...]，最后一条须为 user）。

    多轮实现：按 user 消息顺序逐次发起 stream 调用，每次只传当前 user
    message；复用同一 thread_id，让业务侧 LangGraph checkpoint 自动累积
    上下文（历史 assistant 不需要本地传入）。最终轮的模型回复作为
    actual_output 返回。

    登录/会话/流式调用失败时返回错误占位，不抛异常，保证批跑不中断。
    """
    try:
        if isinstance(prompt, str):
            preview = prompt[:60]
        else:
            user_msgs = [m for m in prompt if m.get("role") == "user"]
            preview = f"多轮 {len(user_msgs)} 轮"
        get_logger().info(f"调用审计智能体 agent 接口: prompt={preview}")
        headers = _auth_headers()
        thread_id = _create_conversation(headers)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        final_answer = ""
        user_turns = [m for m in messages if m.get("role") == "user"]
        for turn_idx, m in enumerate(user_turns, 1):
            print(f"    [turn {turn_idx}/{len(user_turns)}] {m['content'][:60]}...")
            events = _stream_query(headers, thread_id, [m], timeout)
            final_answer = _extract_answer(events) or ""
    except Exception as e:
        return {"answer": f"(调用失败: {e})", "trace_id": "", "retrieved_contexts": []}

    return {
        "answer": final_answer or "(无响应内容)",
        "trace_id": thread_id,
        "retrieved_contexts": [],
    }
