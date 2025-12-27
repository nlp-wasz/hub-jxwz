import asyncio

import openai
from fastmcp import FastMCP
from typing import Annotated

mcp = FastMCP(
    name="sentiment-classification-MCP-Server",
    instructions="""This server contains some api of sentiment-classification.""",
)


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    # 构造提示词让大模型进行情感分析
    system_prompt = """
    你是一个专业的情感分析专家。请分析给定文本的情感倾向。

    请严格按照以下格式输出结果：
    情感类别：[positive/negative/neutral]
    置信度：[0-100]%
    分析理由：[简要说明判断依据]

    要求：
    1. 情感类别只能从 positive(积极)、negative(消极)、neutral(中性) 中选择
    2. 置信度表示分析的可信程度
    3. 分析理由要简洁明了
    """

    user_prompt = f"请对以下文本进行情感分析：\n\n文本内容：\"{text}\""

    # 使用新版 OpenAI Chat Completions API
    client = openai.OpenAI(
        api_key="sk-4935fc3535a64938b35cc4b36a98137c",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 如果使用阿里云模型
    )

    try:
        response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        return f"情感分析失败: {str(e)}"



from fastmcp import Client


async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8902)
