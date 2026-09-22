# logger.py
"""日志封装：仅文件输出到 logs/ 目录，文件名按启动时刻年月日_时分命名。"""
import logging
from datetime import datetime

from config.pipeline_cfg import LOGS_DIR

_LOGGER = None


def get_logger():
    """获取全局文件日志器（懒加载，进程内复用同一文件）。

    日志仅写入 LOGS_DIR/YYYYMMDD_HHMM.log，不输出到控制台。
    多次调用返回同一 logger 实例。
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M')}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger = logging.getLogger("rag_eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False  # 阻断向 root 传播，避免控制台输出
    _LOGGER = logger
    return logger
