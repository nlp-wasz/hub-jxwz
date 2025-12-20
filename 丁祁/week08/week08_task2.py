
from fastapi import FastAPI
from pydantic import BaseModel
from week08_task1_prompt import class_base_on_input
from week08_task1_tools import class_base_on_input_tools

# 创建 FastAPI 应用实例
app = FastAPI(
    title="意图查询API",
    description="意图识别 + 领域识别 + 实体识别。",
    version="1.0.0"
)


# 使用 Pydantic 定义学生数据模型，用于请求体和响应体
class Query(BaseModel):
    """
    用户查询数据模型
    """
    query_str: str


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/class_base_on_input/", status_code=201, summary="查询 api 获取分类数据")
def create_student(query: Query):
    result= class_base_on_input(query.query_str)
    return result

@app.post("/class_base_on_input_tools/", status_code=201, summary="查询 api 获取分类数据 by tools")
def create_student(query: Query):
    result= class_base_on_input_tools(query.query_str)
    return result

# uvicorn week08_task2:app --reload
# curl -i 127.0.0.1:8000/
# curl -X POST "http://127.0.0.1:8000/class_base_on_input/" -H "Content-Type: application/json" -d '{"query_str":"帮忙打开一下酷狗音乐播放音乐行不"}'
# curl -X POST "http://127.0.0.1:8000/class_base_on_input_tools/" -H "Content-Type: application/json" -d '{"query_str":"帮忙打开一下酷狗音乐播放音乐行不"}'


# curl -X POST "http://127.0.0.1:8000/items/" \
#      -H "Content-Type: application/json" \
#      -d '{"name":"Book","price":19.9,"description":"A good read"}'
