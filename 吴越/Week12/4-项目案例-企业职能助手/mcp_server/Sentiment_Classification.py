from fastmcp import FastMCP
import os
from typing import Annotated, Union
from pydantic import BaseModel,Field
from typing_extensions import Literal
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-fad1550b59d547ee83006bde2452e7bc"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"



mcp=FastMCP(
    name="Sentiment Classification",
    instructions="This server contains sentiment Classification.",
)


class Text(BaseModel):
    """Analysize the sentiment  type of text."""
    sentiment:Literal["正向","反向"]=Field(description="情感类型")


tools=[Text]
llm=init_chat_model("qwen-plus",model_provider="openai")
llm_with_tools=llm.bind_tools(tools)

@mcp.tool
def sentiment_classification(text:Annotated[str,"The text to analyze"]):
    """Classifies the sentiment of a given text."""
    result=llm_with_tools.invoke(text)

    if result.tool_calls:
        tool_call = result.tool_calls[0]
        sentiment = tool_call["args"]["sentiment"]
        # 只返回情感类型，不包含其他信息
        return f"情感类型：{sentiment}"
    else:
        return "情感类型: 未知"

