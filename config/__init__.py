"""评估系统配置包：导入任意配置模块前，先加载项目根目录的 .env。"""
from pathlib import Path

from dotenv import load_dotenv

# 显式指定 .env 路径（项目根目录），避免受运行时 cwd 影响
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
