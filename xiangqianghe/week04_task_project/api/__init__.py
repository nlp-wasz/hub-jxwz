"""
API 包初始化文件

此文件使 api 目录成为一个 Python 包
"""

# 定义包级别的导入
from .main import app  # 导入 FastAPI 应用实例
from .schemas import TextClassifyRequest, TextClassifyResponse  # 导入数据模型

# 定义包版本
__version__ = "1.0.0"

# 控制导入的内容
__all__ = ["app", "TextClassifyRequest", "TextClassifyResponse"]

# 初始化日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("API package initialized")