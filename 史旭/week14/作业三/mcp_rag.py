# 通过获取MCP工具信息，筛选与用户提问的问题最相似的3个工作 作为白名单，交给LLM使用
import asyncio

import torch
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_api, trace, ModelSettings, AsyncOpenAI, \
    OpenAIChatCompletionsModel
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from sentence_transformers import SentenceTransformer

# agent 环境配置
set_tracing_disabled(True)
set_default_openai_api("chat_completions")


# 1.获取 MCP 所有工具信息
async def get_mcp_tools():
    async with MCPServerSse(params={"url": "http://127.0.0.1:8000/sse"}) as mcp_sse:
        tools = await mcp_sse.list_tools()

        # 获取工具名称 和 工具描述信息
        tools_info = {
            "tool_name": [],
            "tool_desc": []
        }

        for tool in tools:
            tool_name = tool.name
            tool_desc = tool.description

            tools_info["tool_name"].append(tool_name)
            tools_info["tool_desc"].append(tool_desc)

    return tools_info


# 2.对 MCP 工具信息进行编码
async def get_top3(question, tools_info):
    # 使用 Qwen3-Embedding 模型编码
    embedding_model = SentenceTransformer("../../../models/Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    tool_info_embedding = embedding_model.encode_document(tools_info["tool_desc"], show_progress_bar=True,
                                                          convert_to_tensor=True)

    question_embedding = embedding_model.encode_query(question, convert_to_tensor=True)

    # 相似度计算
    similarity = embedding_model.similarity(question_embedding, tool_info_embedding)[0]

    # 获取 TOP3
    top_3_score, top_3_index = torch.topk(similarity, 3)

    # 返回结果
    top_3_res = [{"tool_name": tools_info["tool_name"][idx.item()], "tool_desc": tools_info["tool_desc"][idx.item()]}
                 for idx in top_3_index]

    return top_3_res


# 3.解析 工具名称
async def get_tools_name(top_3_res):
    tools_name = [tool_inof["tool_name"] for tool_inof in top_3_res]

    return tools_name


# 4.构建提示词
async def get_prompt(question, tools_name):
    # 根据 top_3_index 构建prompt提示词工程
    prompt = f"""你是一位专业的科学建模专家，擅长通过数学模型和计算工具解决实际问题。

    ### 用户问题
    「{question}」
    
    """

    # prompt += "\n### 可用计算工具（仅限以下三个）"

    # for i, tool in enumerate(tools_name, start=1):
    #     prompt += f"\n**工具 {i}：`{tool['tool_name']}`**\n"
    #     prompt += f"功能描述：{tool['tool_desc']}\n"
    #
    # prompt += """
    #
    # ### 你的任务
    # 1. **先分析**用户问题是否匹配某个工具的数学模型；
    # 2. **清晰写出**所用模型的公式、参数含义及代入值；
    # 3. **然后调用**对应的 MCP 工具进行实际数值计算；
    # 4. **严禁自行编程或手算数值结果**，所有计算必须通过工具完成；
    # 5. 如果工具不匹配，请回复：“当前无合适计算工具可用。”
    #
    # 请按以下格式回答：
    # - **模型选择**：...
    # - **公式说明**：...
    # - **参数代入**：...
    # - **工具调用**：[由系统自动触发]
    # - **结果解释**：...
    #
    # 开始你的分析：
    # """
    return prompt


# Agent LLM 回答
async def agent_llm(question):
    # 1.获取所有工具
    tools_info = await get_mcp_tools()

    # 2.获取 top3 工具
    top_3_res = await get_top3(question, tools_info)

    # 3.解析 工具名称
    tools_name = await get_tools_name(top_3_res)

    # 4.构建提示词
    prompt = await get_prompt(question, top_3_res)
    message = [
        {"role": "user", "content": prompt}
    ]

    model_client = AsyncOpenAI(
        api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    model = OpenAIChatCompletionsModel(
        model="qwen-plus",
        openai_client=model_client
    )

    # MCP
    mcp_sse = MCPServerSse(
        params={"url": "http://127.0.0.1:8000/sse"},
        tool_filter=ToolFilterStatic(allowed_tool_names=tools_name)
    )
    async with mcp_sse:
        # 创建 Agent
        agent_llm = Agent(
            name="ScientificModelingAgent",
            instructions="你是一位科学建模专家。你可以解释模型公式、参数含义和计算逻辑并给出最终结果。",
            model=model,
            mcp_servers=[mcp_sse],
            model_settings=ModelSettings(parallel_tool_calls=False),
            tool_use_behavior="run_llm_again"
        )

        # 流式输出
        llm_res = Runner.run_streamed(agent_llm, input=message)

        async for event in llm_res.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseOutputItemDoneEvent) and isinstance(event.data.item,
                                                                                      ResponseFunctionToolCall):
                    yield '\n```\njson\n' + event.data.item.name + ":" + event.data.item.arguments + "\n```\n\n"

                if isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta


async def consume():
    async for chunk in agent_llm(
            """在一个水产养殖池中，初始溶解氧释放量为 8.0 mg/L，衰减系数为 0.1 /h，环境扰动振幅为 2.5 mg/L，扰动频率为 0.5 rad/h。
            请计算在 t = 0, 2, 4, 6 小时时的溶解氧浓度。"""):
        print(chunk, end="")


if __name__ == "__main__":
    asyncio.run(consume())
