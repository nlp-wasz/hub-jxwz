# 情感分析 mcp
import logging
import os
from typing import Annotated

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from agents import (
    Agent, Runner, set_tracing_disabled, set_default_openai_api, trace,
    InputGuardrail, GuardrailFunctionOutput
)
from agents.exceptions import InputGuardrailTripwireTriggered

mcp = FastMCP("Emotion analysis")

# 环境配置
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
os.environ["OPENAI_API_KEY"] = "sk-04ab3d7290e243dda1badc5a1d5ac858"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmotionModel(BaseModel):
    """ 用于情感分析，判断是正面情绪还是负面情绪 """
    is_positive: bool = Field(..., description="是否是正面情绪")
    is_emotion: bool = Field(..., description="是否是情感分析方面的问题")


async def emotion_check_function(ctx, agent, input_data):
    # 定义 情感分析Agent
    emotion_agent = Agent(
        name="Emotion analysis",
        model="qwen-max",
        instructions="""
        你必须严格判断用户输入是否包含情感倾向，并以**严格的JSON格式**返回结果，不能包含任何额外内容或解释。

        规则：
        1. 如果文本明显表达正面情绪（如开心、满意、赞扬），则 "is_positive": true
        2. 如果文本明显表达负面情绪（如愤怒、失望、批评），则 "is_positive": false
        3. 如果文本不涉及情感（如提问、事实陈述、命令等），则 "is_emotion": false，此时 "is_positive" 仍必须为 false（不能为 null）
        4. **所有字段必须存在，且值只能是 true 或 false，禁止使用 null、undefined、空字符串等**

        输出示例（仅输出 JSON，不要 markdown、不要解释）：
        {"is_positive": true, "is_emotion": true}
        """,
        output_type=EmotionModel
    )
    emotion_res = await Runner.run(emotion_agent, input_data, context=ctx.context)

    if emotion_res.final_output.is_emotion:
        output_info = f"情感分析结果为：{emotion_res.final_output.is_positive}，正面情感为True，负面情感为False"
        tripwire_triggered = False
    else:
        output_info = "不是情感分析方面的问题，不予回答"
        tripwire_triggered = True

    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=tripwire_triggered
    )


@mcp.tool
async def sentiment_classification(emotion_text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    print(emotion_text)
    # 根据 emotion_agent 返回结果，使用Agent生成最终回答
    # 加入 守护检查Agent，用于检查输入是否符合规则
    try:
        generator_agent = Agent(
            name="Result Generator",
            model="qwen-max",
            instructions="对emotion_agent的分析结果进行简略总结",
            input_guardrails=[
                InputGuardrail(guardrail_function=emotion_check_function)
            ]
        )

        runner_res = await Runner.run(generator_agent, emotion_text)
        generator_res = runner_res.final_output
    except InputGuardrailTripwireTriggered as e:
        generator_res = "不是情感分析方面的问题，不予回答"

    return generator_res
