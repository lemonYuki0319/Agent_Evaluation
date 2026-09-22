# login.py
"""登录接口：登录获取 accessToken。"""
import requests

from config.logger import get_logger
from config.pipeline_cfg import (
    AGENT_LOGIN_BASE_URL,
    AGENT_LOGIN_PATH,
)


def login(mobile: str, password: str) -> str:
    """登录，返回 accessToken。"""
    get_logger().info(f"调用审计智能体登录接口: mobile={mobile}")
    resp = requests.post(
        f"{AGENT_LOGIN_BASE_URL}{AGENT_LOGIN_PATH}",
        json={"mobile": mobile, "password": password, "loginIdentity": "PERSONAL"},
        headers={"Tenant-Id": "1"},
        timeout=30,
    )
    return resp.json()["data"]["accessToken"]
