import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""
    
    # OpenAI API配置
    openai_api_key: str = "sk-78cc4e9ac8f44efdb207b7232ed8"
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 模型配置
    default_model: str = "qwen-plus"
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 日志配置
    log_level: str = "INFO"
    
    # 数据路径
    data_dir: str = "data"
    
    class Config:
        env_file = "config/.env"
        case_sensitive = False

# 创建全局配置实例
settings = Settings()