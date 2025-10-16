from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import homework02_prompt
import homework02_tools
import uvicorn

# FastAPI Python后端开发的框架： 用来部署模型、部署项目的代码
# 创建 FastAPI 应用实例
app = FastAPI(
    title="意图识别API",
    description="一个简单的 FastAPI 应用，用于识别用户输入意图。",
    version="1.0.0"
)
@app.get("/get_answer_prompt")
def get_answer_prompt(content:str)->str:
    return homework02_prompt.PromptClass(
        api_key="sk-e35a94e9130a4a71b9a9c99389275eaa",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus").get_answer_client(content);

@app.get("/get_answer_tools")
def get_answer_tools(content:str)->str:
    return homework02_tools.ExtractionAgent(
        api_key="sk-e35a94e9130a4a71b9a9c99389275eaa",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus").call(content,homework02_tools.Text)


if __name__ == "__main__":
    uvicorn.run(app,workers=1)
