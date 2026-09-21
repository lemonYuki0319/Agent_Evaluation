# registry.py
"""Agent 客户端注册表。

通过 @register_agent("name") 注册一个 agent 实现函数，
main.py 通过 get_agent(AGENT_NAME) 按配置取用。

Agent 客户端签名：
    question: str -> dict
    返回字典需含字段：answer / trace_id / retrieved_contexts
"""
from typing import Callable, Dict

AgentClient = Callable[[str], Dict]

_REGISTRY: Dict[str, AgentClient] = {}


def register_agent(name: str):
    """装饰器：把函数注册为指定名称的 agent 客户端。"""
    def decorator(fn: AgentClient) -> AgentClient:
        if name in _REGISTRY:
            raise ValueError(f"agent '{name}' 已注册")
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_agent(name: str) -> AgentClient:
    """按名称取已注册的 agent 客户端。"""
    if name not in _REGISTRY:
        raise KeyError(
            f"未注册的 agent: '{name}'，已注册: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_agents() -> list:
    """列出所有已注册 agent 名称。"""
    return list(_REGISTRY)
