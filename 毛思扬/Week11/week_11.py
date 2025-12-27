from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

app = FastAPI(title="Multimodal RAG API", description="多模态RAG系统接口")


# 数据模型定义
class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    id: str
    filename: str
    file_type: str  # text, image, pdf等
    created_at: datetime
    tags: List[str] = []
    size: int


class TextDocumentCreate(BaseModel):
    """创建文本文档请求模型"""
    content: str
    filename: str
    tags: List[str] = []


class TextDocumentResponse(BaseModel):
    """文本文档响应模型"""
    document_id: str
    content: str
    metadata: DocumentMetadata


class ImageDocumentResponse(BaseModel):
    """图像文档响应模型"""
    document_id: str
    image_url: str
    metadata: DocumentMetadata


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query_text: Optional[str] = None
    query_image: Optional[str] = None  # base64编码的图像或图像URL
    top_k: int = 5
    filters: Dict[str, Any] = {}


class SearchResultItem(BaseModel):
    """搜索结果项"""
    document_id: str
    score: float
    content_preview: str
    document_type: str


class SearchResultResponse(BaseModel):
    """搜索结果响应"""
    results: List[SearchResultItem]
    total: int


class QARequest(BaseModel):
    """问答请求模型"""
    question_text: Optional[str] = None
    question_image: Optional[str] = None  # base64编码的图像或图像URL
    context_documents: List[str] = []  # 相关文档ID列表
    top_k: int = 3


class QAResponse(BaseModel):
    """问答响应模型"""
    answer: str
    sources: List[str]  # 引用的文档ID
    confidence: float


# 数据管理接口
@app.post("/documents/text", response_model=TextDocumentResponse, summary="上传文本文档")
async def upload_text_document(document: TextDocumentCreate):
    """
    上传文本文档到系统

    Args:
        document: 文本文档对象，包含内容、文件名和标签

    Returns:
        TextDocumentResponse: 包含文档ID和元数据的响应
    """
    # 实际实现中这里会保存文档到数据库或向量存储
    doc_id = str(uuid.uuid4())
    metadata = DocumentMetadata(
        id=doc_id,
        filename=document.filename,
        file_type="text",
        created_at=datetime.now(),
        tags=document.tags,
        size=len(document.content)
    )

    return TextDocumentResponse(
        document_id=doc_id,
        content=document.content,
        metadata=metadata
    )


@app.post("/documents/image", response_model=ImageDocumentResponse, summary="上传图像文档")
async def upload_image_document(
        file: UploadFile = File(...),
        filename: str = Form(None),
        tags: str = Form("")
):
    """
    上传图像文档到系统

    Args:
        file: 图像文件
        filename: 文件名（可选）
        tags: 标签，逗号分隔的字符串

    Returns:
        ImageDocumentResponse: 包含文档ID和元数据的响应
    """
    # 实际实现中这里会保存图像文件并提取特征
    doc_id = str(uuid.uuid4())
    tag_list = tags.split(",") if tags else []

    metadata = DocumentMetadata(
        id=doc_id,
        filename=filename or file.filename,
        file_type="image",
        created_at=datetime.now(),
        tags=tag_list,
        size=0  # 实际大小需要从文件获取
    )

    # 模拟图像URL
    image_url = f"/images/{doc_id}"

    return ImageDocumentResponse(
        document_id=doc_id,
        image_url=image_url,
        metadata=metadata
    )


@app.get("/documents/{document_id}", summary="获取文档详情")
async def get_document(document_id: str):
    """
    根据文档ID获取文档详情

    Args:
        document_id: 文档唯一标识符

    Returns:
        dict: 文档详细信息
    """
    # 这里应该是从数据库获取实际文档数据的逻辑
    return {
        "document_id": document_id,
        "content": "Sample document content",
        "metadata": {
            "id": document_id,
            "filename": "sample.txt",
            "file_type": "text",
            "created_at": datetime.now().isoformat(),
            "tags": ["sample", "test"]
        }
    }


@app.delete("/documents/{document_id}", summary="删除文档")
async def delete_document(document_id: str):
    """
    根据文档ID删除文档

    Args:
        document_id: 文档唯一标识符

    Returns:
        dict: 删除操作结果
    """
    # 实际实现中这里会从存储中删除文档
    return {"message": f"Document {document_id} deleted successfully"}


# 多模态检索接口
@app.post("/search", response_model=SearchResultResponse, summary="多模态检索")
async def multimodal_search(request: SearchRequest):
    """
    支持文本和图像查询的多模态检索

    Args:
        request: 搜索请求，可以包含文本查询、图像查询等参数

    Returns:
        SearchResultResponse: 搜索结果列表
    """
    # 实际实现中这里会执行多模态相似度检索
    results = [
        SearchResultItem(
            document_id=str(uuid.uuid4()),
            score=0.95 - i * 0.1,
            content_preview=f"Search result preview {i + 1}",
            document_type="text" if i % 2 == 0 else "image"
        )
        for i in range(min(request.top_k, 5))
    ]

    return SearchResultResponse(
        results=results,
        total=len(results)
    )


# 多模态问答接口
@app.post("/qa", response_model=QAResponse, summary="多模态问答")
async def multimodal_qa(request: QARequest):
    """
    基于多模态内容的问答接口

    Args:
        request: 问答请求，可以包含文本问题、图像问题等

    Returns:
        QAResponse: 问答结果，包括答案、来源和置信度
    """
    # 实际实现中这里是多模态RAG的核心逻辑
    answer = "这是一个示例答案。在实际实现中，这将是基于检索到的相关文档生成的答案。"
    sources = request.context_documents if request.context_documents else [str(uuid.uuid4()) for _ in range(2)]

    return QAResponse(
        answer=answer,
        sources=sources,
        confidence=0.85
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
