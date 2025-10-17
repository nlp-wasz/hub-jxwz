from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from week08_2_1 import analyze_user_input
from week08_2_2 import analyze_user_input_tools

app = FastAPI(
    title="智能助手",
)



class AnalysisResult(BaseModel):
    """用户输入分析结果模型"""
    intent: List[str]
    domain: List[str]
    entities: List[str]


@app.get("/prompt")
async def prompt(question: str):
    return analyze_user_input(question)


@app.get("/tools")
async def prompt(question: str):
    return analyze_user_input_tools(question)





