import datetime
from typing import Optional

from pydantic import BaseModel, Field


# 请求 和 响应 格式定义
# 1.知识库请求和响应格式
class KnowledgeRequest(BaseModel):
    knowledge_title: Optional[str] = Field(..., description="知识库标题")
    knowledge_category: Optional[str] = Field(..., description="知识库类型")


class KnowledgeResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    knowledge_id: int = Field(..., description="知识库id")
    knowledge_title: Optional[str] = Field(..., description="知识库标题")
    knowledge_category: Optional[str] = Field(..., description="知识库类型")
    knowledge_create: Optional[datetime.datetime] = Field(..., description="知识库创建时间")
    knowledge_update: Optional[datetime.datetime] = Field(..., description="知识库更新时间")

    response_code: str = Field(..., description="响应状态")
    response_mess: str = Field(..., description="响应信息")
    response_time: str = Field(..., description="响应时间")
    response_status: str = Field(..., description="处理状态（completed，padding，failed）")


# 2. 知识库文档 请求和响应格式
class DocumentRequest(BaseModel):
    document_title: Optional[str] = Field(..., description="知识库文档标题")
    document_category: Optional[str] = Field(..., description="知识库文档类型")
    knowledge_id: Optional[int] = Field(..., description="知识库文档所有知识库ID")
    # file: UploadFile = Field(..., description="文件信息")


class DocumentResponse(BaseModel):
    request_id: str = Field(..., description="请求ID")
    document_id: int = Field(..., description="文档ID")
    document_title: Optional[str] = Field(..., description="文档标题")
    document_category: Optional[str] = Field(..., description="文档类型")
    knowledge_id: Optional[int] = Field(..., description="所属知识库ID")
    file_path: Optional[str] = Field(..., description="文档path")
    document_create: Optional[datetime.datetime] = Field(..., description="文档创建时间")
    document_update: Optional[datetime.datetime] = Field(..., description="文档更新时间")

    response_code: str = Field(..., description="响应状态")
    response_mess: str = Field(..., description="响应信息")
    response_time: str = Field(..., description="响应时间")
    response_status: str = Field(..., description="处理状态（completed，padding，failed）")


# 3.RAG问答 请求和响应 格式
class RagRequest(BaseModel):
    question: str = Field(..., description="问题")
    knowledge_id: int = Field(..., description="知识库ID")


class RagResponse(BaseModel):
    request_id: str = Field(..., description="请求ID")
    question: str = Field(..., description="问题")
    knowledge_id: int = Field(..., description="知识库ID")
    document_chunk_content: str = Field(..., description="问题对应的chunk内容")
    llm_response: str = Field(..., description="大模型回答")

    response_code: str = Field(..., description="响应状态")
    response_mess: str = Field(..., description="响应信息")
    response_time: str = Field(..., description="响应时间")
    response_status: str = Field(..., description="处理状态（completed，padding，failed）")
