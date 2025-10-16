# 通过使用 Prompt+LLM 或者 tools + LLM 完成意图识别、领域识别、实体识别任务
import os

from langchain_core.prompts import ChatPromptTemplate
from Arguments import Arguments

args = Arguments()

# 提示词模板
prompt = args.prompt

# 自定义tools（手写 参数信息）
tools = args.tools

# pydantic + tools（自动解析 pydantic 类中的属性信息）
pydanticTools = args.pydanticTools


# 1.原始OpenAI
def Prompt_LLM_OpenAI(pre_texts):
    from openai import OpenAI

    # prompt 提示词信息
    global prompt
    filled_prompt = prompt.format(inputs=pre_texts)

    openAI = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 调用LLM
    res = openAI.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "user", "content": filled_prompt}
        ]
    )

    # print(res.model_dump_json(indent=2))
    return res


# 2.LangChain 框架实现（ChatOpenAI 或者 init_chat_model  +  ChatPromptTemplate）
def Prompt_LLM_LangChain(pre_texts):
    from langchain_openai import ChatOpenAI
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate

    # prompt 提示词信息
    global prompt
    filled_prompt = prompt.format(inputs=pre_texts)

    # 创建 ChatOpenAI
    chatOpenAI = ChatOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model_name
    )

    # 创建提示词模板
    prompt_template = ChatPromptTemplate(
        [("system", "你是一个专业的assistant"), ("user", "{content}")]
    )
    message = prompt_template.invoke({"content": filled_prompt})

    # 调用LLM
    res = chatOpenAI.invoke(message)

    # print(res.model_dump_json(indent=2))
    return res


# 3.使用 LLM + tools 方式完成任务
# 3.1 自定义tools（手写 参数信息）
# ①原始 OpenAI
def Prompt_LLM_OpenAI_Tools(pre_texts):
    from openai import OpenAI

    # prompt 提示词信息
    global prompt
    filled_prompt = prompt.format(inputs=pre_texts)

    openAI = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 调用LLM
    res = openAI.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "user", "content": filled_prompt}
        ],
        tools=tools,
        tool_choice="auto"
    )

    # print(res.model_dump_json(indent=2))
    return res


# 3.1 自定义tools（手写 参数信息）
# ②LangChain 框架的 ChatOpenAI 或者 init_chat_model()
def Promt_LLM_ChatOpenAI_Tools(pre_texts):
    from langchain.chat_models import init_chat_model

    # prompt 提示词信息
    global prompt
    filled_prompt = prompt.format(inputs=pre_texts)

    # os.environ 和 dotenv.load_dotenv() 加载 OPENAI_API_KEY  OPENAI_BASE_URL
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url
    chatLLM = init_chat_model(model=args.model_name, model_provider="openai")
    chatLLM_tools = chatLLM.bind_tools(tools=tools, tool_choice="auto")

    # 构建 提示词模板
    prompt_template = ChatPromptTemplate(
        [("user", "{content}")]
    )
    messages = prompt_template.invoke({"content": filled_prompt})

    res = chatLLM_tools.invoke(messages)

    # print(res.model_dump_json(indent=2))
    return res


# 3.2 pydantic + tools（自动解析 pydantic 类中的属性信息）
def Promt_LLM_OpenAI_PydanticTools(pre_texts):
    from openai import OpenAI

    openAI = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 调用LLM
    res = openAI.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": "你是一个专业的assistant"},
            {"role": "user", "content": args.prompt.format(inputs=pre_texts)}
        ],
        tools=pydanticTools,
        tool_choice="auto"
    )

    # print(res.model_dump_json(indent=2))
    return res


# 3.3 自定义Agent，完成 pydantic + tools + llm 流程（或者 普通的tools + llm）
def Promt_LLM_Agent(pre_texts):
    from Arguments import AutoAgent

    res = AutoAgent(args).call(pre_texts)

    # print(res.model_dump_json(indent=2))
    return res


a = ["你去玩手机了吧用英语怎么说", " 分手时，背一首诗吧？ ", " 本期七星彩的中奖号码是多少？ ", " 何以解忧的下一句是什么？ ",
     " 请帮我调频90.2连云港经济广播电台"]
# res = Prompt_LLM_OpenAI_Tools(a)
# print(res.model_dump_json(indent=2))
