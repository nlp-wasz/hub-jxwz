import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_OPENAI_API_KEY = os.getenv("LLM_OPENAI_API_KEY", "sk-")
    LLM_OPENAI_SERVER_URL = os.getenv("LLM_OPENAI_SERVER_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")

settings = Settings()