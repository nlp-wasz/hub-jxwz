# FastApi 部署模块
import json
import time
import traceback
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse

from BaseClass import Request, Response
from Training import train_func, eval_func, pre_func
from Prompt_LLM import (
    Prompt_LLM_OpenAI, Prompt_LLM_LangChain,
    Prompt_LLM_OpenAI_Tools,
    Promt_LLM_ChatOpenAI_Tools,
    Promt_LLM_OpenAI_PydanticTools,
    Promt_LLM_Agent
)

api = FastAPI(
    title='意图识别和实体识别--结合任务',
    description="根据用户输入的文本，判断用户的意图，以及识别文本中的实体并标注实体类型。",
    version="1.0.0",
    docs_url=None
)


# fastapi页面swagger方式默认加载境外资源，这里修改为国内资源
@api.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API 文档",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


# 第一种方式：自定义BERT模型（完成意图识别和实体识别任务）
# 训练模型
@api.post("/bert_model_training", description="自定义BERT模型——Training")
def bert_model_training(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用评估方法
    try:
        train_res = train_func()

        response.response_res = train_res
        response.response_message = "Training Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 第一种方式：自定义BERT模型（完成意图识别和实体识别任务）
# 评估模型
@api.post("/bert_model_eval", description="自定义BERT模型——Eval")
def bert_model_eval(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        eval_res = eval_func()

        response.response_res = eval_res
        response.response_message = "Eval Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 第一种方式：自定义BERT模型（完成意图识别和实体识别任务）
# 预测文本
@api.post("/bert_model_predict", description="自定义BERT模型——Predict")
def bert_model_predict(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        # "你去玩手机了吧用英语怎么说","分手时，背一首诗吧？","本期七星彩的中奖号码是多少？","何以解忧的下一句是什么？","请帮我调频90.2连云港经济广播电台"
        pre_res = pre_func(request_content)
        # print(json.dumps(pre_res, indent=4, ensure_ascii=False))

        response.response_res = pre_res
        response.response_message = "Predict Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 第二种方式：通过 prompt + LLM（完成意图识别和实体识别任务）
# 1.原始OpenAI
@api.post("/openai", description="prompt + LLM（1.原始OpenAI）")
def openai(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Prompt_LLM_OpenAI(request_content)

        print(res.choices[0].message.content)
        response.response_res = res.choices[0].message.content
        response.response_message = "prompt + LLM（1.原始OpenAI） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 2.LangChain 框架实现（ChatOpenAI 或者 init_chat_model  +  ChatPromptTemplate）
@api.post("/langchain", description="prompt + LLM（2.LangChain 框架）")
def langchain(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Prompt_LLM_LangChain(request_content)

        response.response_res = res.content
        response.response_message = "prompt + LLM（2.LangChain 框架） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 3.使用 LLM + tools 方式完成任务
# 3.1 自定义tools（手写 参数信息）
# ①原始 OpenAI
@api.post("/openai_tools", description="prompt + tools + LLM（3.1 自定义tools + 原始OpenAI）")
def openai_tools(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content
    print(request_content)

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Prompt_LLM_OpenAI_Tools(request_content)

        response.response_res = res.choices[0].message.tool_calls[0].function.arguments
        response.response_message = "prompt + tools + LLM（3.1 自定义tools + 原始OpenAI） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 3.1 自定义tools（手写 参数信息）
# ②LangChain 框架的 ChatOpenAI 或者 init_chat_model()
@api.post("/langchain_tools", description="prompt + tools + LLM（3.1 自定义tools + LangChain框架）")
def langchain_tools(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Promt_LLM_ChatOpenAI_Tools(request_content)

        # response.response_res = res.tool_calls[0]["args"]
        response.response_res = res.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        response.response_message = "prompt + tools + LLM（3.1 自定义tools + LangChain框架） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 3.2 pydantic + tools（自动解析 pydantic 类中的属性信息）
@api.post("/pydantic_tools", description="prompt + tools + LLM（3.2 pydantic + tools自动解析）")
def pydantic_tools(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Promt_LLM_OpenAI_PydanticTools(request_content)

        response.response_res = res.choices[0].message.tool_calls[0].function.arguments
        response.response_message = "prompt + tools + LLM（3.2 pydantic + tools自动解析） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


# 3.3 自定义Agent，完成 pydantic + tools + llm 流程（或者 普通的tools + llm）
@api.post("/agent", description="prompt + tools + LLM（3.3 自定义Agent）")
def agent(request: Request) -> Response:
    # 获取请求信息
    request_id = request.request_id
    request_content = request.request_content

    # 响应结果
    response = Response(
        request_id=request_id,
        request_content=request_content,
        response_time="",
        response_res="",
        response_message=""
    )

    start_time = time.time()
    # 调用训练方法
    try:
        res = Promt_LLM_Agent(request_content)

        response.response_res = res.choices[0].message.tool_calls[0].function.arguments
        response.response_message = "prompt + tools + LLM（3.3 自定义Agent） Success"
    except Exception as e:
        response.response_message = traceback.format_exc()

    end_time = time.time()
    execute_time = f"{end_time - start_time:.4f}s" if end_time - start_time > 1 else f"{(end_time - start_time) * 1000}ms"

    response.response_time = execute_time

    return response


if __name__ == '__main__':
    # 自定义启动 FastAPI
    uvicorn.run(api, host="127.0.0.1", port=8000, workers=1)
