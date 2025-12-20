# 主文件 fastapi 入口（包含 知识库的crud  文档的crud，rag多轮问答等接口）
import datetime
import os
import pathlib
import time
import traceback
import uuid
from typing import Annotated, Optional

import uvicorn

from fastapi import FastAPI, Form, File, UploadFile, BackgroundTasks
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse

# 请求 和 响应 格式
from datatype import (
    KnowledgeRequest, KnowledgeResponse,
    DocumentRequest, DocumentResponse,
    RagRequest, RagResponse
)
from rag import Rag

# 数据库 映射实体类对象 和 session 会话对象
from db_api import KnowledgeDatabase, KnowledgeDocument, Session
from sqlalchemy import and_

api = FastAPI(
    title="政企项目",
    description="RAG多轮问答（bm25全文检索，bert语义检索 + 融合排序rrf算法 + 重排序（rerank模型） + 调用llm）",
    version="0.0.1",
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


# 一：知识库操作（curd）
# 1.新增 知识库信息
@api.post("/knowledge_add", description="新增知识库信息")
def knowledge_add(request: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()

    # 获取 知识库 标题和类型
    knowledge_title = request.knowledge_title
    knowledge_category = request.knowledge_category

    # 响应 格式
    knowledgeResponse = KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        knowledge_title=knowledge_title,
        knowledge_category=knowledge_category,
        knowledge_create=None,
        knowledge_update=None,
        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    # 创建 KnowledgeDatabase 对象
    knowledge_databse = KnowledgeDatabase(
        knowledge_title=knowledge_title,
        knowledge_category=knowledge_category
    )

    # 通过 with 自动管理 Session 的关闭
    with Session() as session:
        try:
            # 判断 知识库中是否存在 一样的数据
            delete = session.query(KnowledgeDatabase) \
                .filter(and_(KnowledgeDatabase.knowledge_title == knowledge_title,
                             KnowledgeDatabase.knowledge_category == knowledge_category)) \
                .delete()
            session.commit()
            if delete > 0:
                print(f"从KnowledgeDatabase库删除{delete}条数据")

            # 添加数据
            session.add(knowledge_databse)
            session.commit()

            # 添加完查询
            add_info = session.query(KnowledgeDatabase) \
                .filter(and_(KnowledgeDatabase.knowledge_title == knowledge_title,
                             KnowledgeDatabase.knowledge_category == knowledge_category)) \
                .first()

            if add_info:
                print(f"往KnowledgeDatabase库添加 标题为【{add_info.knowledge_title}】的数据")
                # 整理 response
                knowledgeResponse.knowledge_id = add_info.knowledge_id
                knowledgeResponse.knowledge_create = add_info.create_dt
                knowledgeResponse.knowledge_update = add_info.update_dt
                knowledgeResponse.response_code = "200"
                knowledgeResponse.response_mess = "新增成功"
                knowledgeResponse.response_status = "completed"

        except Exception:
            error_info = traceback.format_exc()
            # 整理 response
            knowledgeResponse.response_code = "500"
            knowledgeResponse.response_mess = f"新增失败【{error_info}】"
            knowledgeResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    knowledgeResponse.response_time = hs

    return knowledgeResponse


# 2.删除知识库信息
@api.post("/knowledge_delete", description="删除知识库信息（根据id）")
def knowledge_delete(knowledge_id: int) -> KnowledgeResponse:
    start_time = time.time()

    # 响应 格式
    knowledgeResponse = KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        knowledge_title="",
        knowledge_category="",
        knowledge_create=None,
        knowledge_update=None,
        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    # 根据 knowledge_id 删除知识库信息（先查询，用于记录）
    with Session() as session:
        try:
            query_res = session.query(KnowledgeDatabase) \
                .filter(KnowledgeDatabase.knowledge_id == knowledge_id) \
                .one()

            # 删除
            delete_res = session.query(KnowledgeDatabase) \
                .filter(KnowledgeDatabase.knowledge_id == knowledge_id) \
                .delete()
            session.commit()
            if delete_res > 0:
                print(f"从KnowledgeDatabase库删除{delete_res}条数据")

            if query_res:
                # 整理 response
                knowledgeResponse.knowledge_title = query_res.knowledge_title
                knowledgeResponse.knowledge_category = query_res.knowledge_category
                knowledgeResponse.knowledge_create = query_res.create_dt
                knowledgeResponse.knowledge_update = query_res.update_dt
                knowledgeResponse.response_code = "200"
                knowledgeResponse.response_mess = "删除成功"
                knowledgeResponse.response_status = "completed"
        except Exception:
            error_info = traceback.format_exc()
            knowledgeResponse.response_code = "500"
            knowledgeResponse.response_mess = f"删除失败【{error_info}】"
            knowledgeResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    knowledgeResponse.response_time = hs

    return knowledgeResponse


# 3.查询知识库信息
@api.post("/knowledge_select", description="查询知识库信息（根据id）")
def knowledge_delete(knowledge_id: int) -> KnowledgeResponse:
    start_time = time.time()

    # 响应 格式
    knowledgeResponse = KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        knowledge_title="",
        knowledge_category="",
        knowledge_create=None,
        knowledge_update=None,
        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    # 根据 knowledge_id 查询知识库信息
    with Session() as session:
        try:
            query_res = session.query(KnowledgeDatabase) \
                .filter(KnowledgeDatabase.knowledge_id == knowledge_id) \
                .one()
            if query_res:
                # 整理 response
                knowledgeResponse.knowledge_title = query_res.knowledge_title
                knowledgeResponse.knowledge_category = query_res.knowledge_category
                knowledgeResponse.knowledge_create = query_res.create_dt
                knowledgeResponse.knowledge_update = query_res.update_dt
                knowledgeResponse.response_code = "200"
                knowledgeResponse.response_mess = "查询成功"
                knowledgeResponse.response_status = "completed"
        except Exception:
            error_info = traceback.format_exc()
            knowledgeResponse.response_code = "500"
            knowledgeResponse.response_mess = f"查询失败【{error_info}】"
            knowledgeResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    knowledgeResponse.response_time = hs

    return knowledgeResponse


# 4.修改知识库信息
@api.post("/knowledge_update", description="修改知识库信息")
def knowledge_delete(knowledge_id: int, request: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()

    # 获取 知识库 修改后的 标题和类型
    knowledge_title = request.knowledge_title
    knowledge_category = request.knowledge_category

    # 响应 格式
    knowledgeResponse = KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        knowledge_title=knowledge_title,
        knowledge_category=knowledge_category,
        knowledge_create=None,
        knowledge_update=None,
        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    # 根据 knowledge_id 先查询知识库信息，再进行修改
    with Session() as session:
        try:
            query_res = session.query(KnowledgeDatabase) \
                .filter(KnowledgeDatabase.knowledge_id == knowledge_id) \
                .one()

            # 更新
            if query_res:
                query_res.knowledge_title = knowledge_title
                query_res.knowledge_category = knowledge_category
                session.commit()
                # 整理 response
                knowledgeResponse.knowledge_create = query_res.create_dt
                knowledgeResponse.knowledge_update = query_res.update_dt
                knowledgeResponse.response_code = "200"
                knowledgeResponse.response_mess = "修改成功"
                knowledgeResponse.response_status = "completed"

        except Exception:
            error_info = traceback.format_exc()
            knowledgeResponse.response_code = "500"
            knowledgeResponse.response_mess = f"修改失败【{error_info}】"
            knowledgeResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    knowledgeResponse.response_time = hs

    return knowledgeResponse


# 二：知识库文档操作（crud，同时也需要操作 Elasticsearch 文件数据库）
# 1.新增 知识库文档信息
@api.post("/document_add", description="新增文档")
def document_add(
        document_title: Annotated[str, Form(...)],
        document_category: Annotated[str, Form(...)],
        knowledge_id: Annotated[int, Form(...)],
        file: Annotated[UploadFile, File(...)]
) -> DocumentResponse:
    start_time = time.time()

    documentResponse = DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=0,
        document_title=document_title,
        document_category=document_category,
        knowledge_id=knowledge_id,
        file_path="",
        document_create=None,
        document_update=None,

        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    try:
        # 先删除原有文档，再新增
        with Session() as session:
            # 根据 document_title  document_category  knowledge_id 删除文档
            delete = session.query(KnowledgeDocument) \
                .filter(and_(KnowledgeDocument.document_title == document_title,
                             KnowledgeDocument.document_category == document_category,
                             KnowledgeDocument.knowledge_id == knowledge_id)) \
                .delete()
            session.commit()
            if delete > 0:
                print(f"从KnowledgeDocument库删除{delete}条数据")

            # 新增
            knowledge_document = KnowledgeDocument(
                document_title=document_title,
                document_category=document_category,
                knowledge_id=knowledge_id,
                file_path="",
                file_type=file.content_type
            )
            session.add(knowledge_document)
            session.commit()

            # 添加完查询
            add_info = session.query(KnowledgeDocument) \
                .filter(and_(KnowledgeDocument.document_title == document_title,
                             KnowledgeDocument.document_category == document_category,
                             KnowledgeDocument.knowledge_id == knowledge_id)) \
                .first()

            # 获取文档ID，根据文档ID 构建文件存储地址 file_path
            # 将上传的文件 加载到 当前目录下的 data文件夹
            file_path = f"./data/{add_info.document_id}_{file.filename}"
            # 如果不存在 data 目录，自动创建
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.file.read())

            # 更新 file_path
            add_info.file_path = file_path
            session.commit()

            # 整理 响应体
            documentResponse.document_id = add_info.document_id
            documentResponse.file_path = add_info.file_path
            documentResponse.document_create = add_info.create_dt
            documentResponse.document_update = add_info.update_dt

            documentResponse.response_code = "200"
            documentResponse.response_mess = f"文档新增成功"
            documentResponse.response_status = "completed"

        # 将新增的文档信息 添加到ES文档数据库中（调用 提取文档内容 方法）
        rag = Rag()
        rag.extract_document(add_info)



    except Exception:
        error_info = traceback.format_exc()
        documentResponse.response_code = "500"
        documentResponse.response_mess = f"文档新增失败【{error_info}】"
        documentResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    documentResponse.response_time = hs

    return documentResponse


# 2.删除 知识库文档信息
@api.post("/document_delete", description="删除文档信息（根据文档ID）")
def document_delete(document_id: Annotated[int, Form(...)]) -> DocumentResponse:
    start_time = time.time()

    documentResponse = DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        document_title="",
        document_category="",
        knowledge_id=0,
        file_path="",
        document_create=None,
        document_update=None,

        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    try:
        # 根据 文档ID 删除 文档信息
        with Session() as session:
            # 先查询（用于记录）
            delete_info = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).one()

            # 删除
            delete = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).delete()
            session.commit()
            if delete > 0:
                print(f"从KnowledgeDocument库删除{delete}条数据")

            # 删除 存储的文档
            file_path = pathlib.Path(delete_info.file_path)
            if file_path.is_file():
                print(file_path)
                file_path.unlink()

            # 整理 response
            documentResponse.document_title = delete_info.document_title
            documentResponse.document_category = delete_info.document_category
            documentResponse.knowledge_id = delete_info.knowledge_id
            documentResponse.file_path = delete_info.file_path
            documentResponse.document_create = delete_info.create_dt
            documentResponse.document_update = delete_info.update_dt

            documentResponse.response_code = "200"
            documentResponse.response_mess = f"文档删除成功"
            documentResponse.response_status = "completed"

        # 删除文档时  删除es文档数据中 对应的信息
        rag = Rag()
        rag.document_delete(delete_info)


    except Exception:
        error_info = traceback.format_exc()
        documentResponse.response_code = "500"
        documentResponse.response_mess = f"文档删除失败【{error_info}】"
        documentResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    documentResponse.response_time = hs

    return documentResponse


# 3.查询 知识库文档信息
@api.post("/document_select", description="查询文档信息（根据文档ID）")
def document_select(document_id: Annotated[int, Form(...)]) -> DocumentResponse:
    start_time = time.time()

    documentResponse = DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        document_title="",
        document_category="",
        knowledge_id=0,
        file_path="",
        document_create=None,
        document_update=None,

        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    try:
        with Session() as session:
            # 查询
            delete_info = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).one()

            # 整理 response
            documentResponse.document_title = delete_info.document_title
            documentResponse.document_category = delete_info.document_category
            documentResponse.knowledge_id = delete_info.knowledge_id
            documentResponse.file_path = delete_info.file_path
            documentResponse.document_create = delete_info.create_dt
            documentResponse.document_update = delete_info.update_dt

            documentResponse.response_code = "200"
            documentResponse.response_mess = f"文档查询成功"
            documentResponse.response_status = "completed"

    except Exception:
        error_info = traceback.format_exc()
        documentResponse.response_code = "500"
        documentResponse.response_mess = f"文档查询失败【{error_info}】"
        documentResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    documentResponse.response_time = hs

    return documentResponse


# 4.更新 知识库文档信息
@api.post("/document_update", description="更新文档信息")
def document_update(
        document_id: Annotated[int, Form(...)],
        document_title: Annotated[str, Form(...)],
        document_category: Annotated[str, Form(...)],
        update_file: Annotated[UploadFile, File(...)]
) -> DocumentResponse:
    start_time = time.time()

    documentResponse = DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        document_title=document_title,
        document_category=document_category,
        knowledge_id=0,
        file_path="",
        document_create=None,
        document_update=None,

        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    try:
        with Session() as session:
            # 查询
            update_info = session.query(KnowledgeDocument) \
                .filter(KnowledgeDocument.document_id == document_id) \
                .one()

            # 将 原来的文件删除
            file_path = pathlib.Path(update_info.file_path)
            if file_path.is_file():
                file_path.unlink()

            # 更新 文档信息
            update_path = f"./data/{document_id}_{update_file.filename}"
            update_info.document_title = document_title
            update_info.document_category = document_category
            update_info.file_path = update_path
            session.commit()

            # 保存 新的文档信息
            with open(update_path, "wb") as f:
                f.write(update_file.file.read())

            # 整理 response
            documentResponse.knowledge_id = update_info.knowledge_id
            documentResponse.file_path = update_info.file_path
            documentResponse.document_create = update_info.create_dt
            documentResponse.document_update = update_info.update_dt

            documentResponse.response_code = "200"
            documentResponse.response_mess = f"文档更新成功"
            documentResponse.response_status = "completed"

        # 更新文档时  更新es文档数据中 对应的信息
        rag = Rag()
        rag.document_update(update_info)

    except Exception:
        error_info = traceback.format_exc()
        documentResponse.response_code = "500"
        documentResponse.response_mess = f"文档更新失败【{error_info}】"
        documentResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    documentResponse.response_time = hs

    return documentResponse


# 三：RAG 问答
@api.post("/rag_chat", description="RAG 问答")
def rag_chat(ragRequest: RagRequest) -> RagResponse:
    start_time = time.time()

    question, knowledge_id = ragRequest.question, ragRequest.knowledge_id

    ragResponse = RagResponse(
        request_id=str(uuid.uuid4()),
        question=question,
        knowledge_id=knowledge_id,
        document_chunk_content="",
        llm_response="",

        response_code="",
        response_mess="",
        response_time="",
        response_status=""
    )

    try:
        # question 用户提出的问题
        rag = Rag()

        # 调用 rag 流程方法
        llm_response, document_chunk_content = rag.rag_chat(question, knowledge_id)

        ragResponse.document_chunk_content = document_chunk_content
        ragResponse.llm_response = llm_response
        ragResponse.response_code = "200"
        ragResponse.response_mess = f"回答完毕"
        ragResponse.response_status = "completed"
    except Exception:
        error_info = traceback.format_exc()
        ragResponse.response_code = "500"
        ragResponse.response_mess = f"回答超时【{error_info}】"
        ragResponse.response_status = "failed"

    end_time = time.time()
    hs = end_time - start_time
    hs = f"{hs * 1000:.4f}ms" if hs * 1000 >= 1 else f"{hs:.4f}s"
    # 耗时
    ragResponse.response_time = hs

    return ragResponse


if __name__ == '__main__':
    uvicorn.run(api, host="127.0.0.1", port=8000, workers=1)
