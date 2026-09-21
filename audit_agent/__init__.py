"""audit_agent 包：Agent 客户端注册入口。

import 本包即触发所有内置 agent 模块注册（通过 @register_agent 装饰）。
外部使用者只需：from audit_agent import get_agent, list_agents
"""
from .registry import get_agent, list_agents, register_agent  # noqa: F401

# 触发内置 agent 模块注册。新增 agent 时在此追加一行 import 即可。
from .api.legal_agent import stream_request  # noqa: F401
