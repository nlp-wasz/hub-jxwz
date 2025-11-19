from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class DocumentStatus(str, Enum):
    """文档状态枚举"""
    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 处理失败


class ModalityType(str, Enum):
    """模态类型枚举"""
    TEXT = "text"  # 文本
    IMAGE = "image"  # 图像
    TABLE = "table"  # 表格
    CHART = "chart"  # 图表


class PermissionLevel(str, Enum):
    """权限级别枚举"""
    READ = "read"  # 只读
    WRITE = "write"  # 读写
    ADMIN = "admin"  # 管理员


# ==================== 数据管理接口 ====================

class UserInfo(BaseModel):
    """用户信息"""
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: Optional[str] = Field(None, description="邮箱")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class KnowledgeBase(BaseModel):
    """知识库信息"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    owner_id: str = Field(..., description="所有者ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    document_count: int = Field(default=0, description="文档数量")


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str = Field(..., description="文档ID")
    kb_id: str = Field(..., description="所属知识库ID")
    filename: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小(字节)")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="处理状态")
    uploaded_by: str = Field(..., description="上传者ID")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")
    processed_at: Optional[datetime] = Field(None, description="处理完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")


class ChunkInfo(BaseModel):
    """文档块信息"""
    chunk_id: str = Field(..., description="块ID")
    doc_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="块内容")
    modality: ModalityType = Field(..., description="模态类型")
    page_number: Optional[int] = Field(None, description="页码")
    position: Optional[Dict[str, Any]] = Field(None, description="位置信息")
    embedding_id: Optional[str] = Field(None, description="向量ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


# ==================== 数据管理接口请求/响应 ====================

class UploadDocumentRequest(BaseModel):
    """上传文档请求"""
    kb_id: str = Field(..., description="目标知识库ID")
    file_data: bytes = Field(..., description="文件二进制数据")
    filename: str = Field(..., description="文件名")
    description: Optional[str] = Field(None, description="文档描述")


class UploadDocumentResponse(BaseModel):
    """上传文档响应"""
    success: bool = Field(..., description="是否成功")
    doc_id: Optional[str] = Field(None, description="文档ID")
    message: str = Field(..., description="响应消息")


class DeleteDocumentRequest(BaseModel):
    """删除文档请求"""
    doc_id: str = Field(..., description="文档ID")


class DeleteDocumentResponse(BaseModel):
    """删除文档响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")


class CreateKnowledgeBaseResponse(BaseModel):
    """创建知识库响应"""
    success: bool = Field(..., description="是否成功")
    kb_id: Optional[str] = Field(None, description="知识库ID")
    message: str = Field(..., description="响应消息")


class ListDocumentsRequest(BaseModel):
    """列出文档请求"""
    kb_id: str = Field(..., description="知识库ID")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")
    status_filter: Optional[DocumentStatus] = Field(None, description="状态过滤")


class ListDocumentsResponse(BaseModel):
    """列出文档响应"""
    documents: List[DocumentInfo] = Field(..., description="文档列表")
    total_count: int = Field(..., description="总文档数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")

# ==================== 多模态检索接口 ====================

class RetrievalModality(str, Enum):
    """检索模态枚举"""
    TEXT_ONLY = "text_only"  # 仅文本检索
    IMAGE_ONLY = "image_only"  # 仅图像检索
    MULTI_MODAL = "multi_modal"  # 多模态检索


class SearchQuery(BaseModel):
    """搜索查询"""
    text: Optional[str] = Field(None, description="文本查询")
    image_data: Optional[bytes] = Field(None, description="图像二进制数据")
    retrieval_modality: RetrievalModality = Field(default=RetrievalModality.TEXT_ONLY, description="检索模态")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")


class SearchResult(BaseModel):
    """搜索结果项"""
    chunk_id: str = Field(..., description="块ID")
    doc_id: str = Field(..., description="文档ID")
    content: str = Field(..., description="内容")
    modality: ModalityType = Field(..., description="模态类型")
    page_number: Optional[int] = Field(None, description="页码")
    similarity_score: float = Field(..., description="相似度分数")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class SearchRequest(BaseModel):
    """搜索请求"""
    kb_id: str = Field(..., description="知识库ID")
    query: SearchQuery = Field(..., description="搜索查询")


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool = Field(..., description="是否成功")
    results: List[SearchResult] = Field(..., description="搜索结果")
    total_count: int = Field(..., description="总结果数")
    query_time_ms: int = Field(..., description="查询耗时(毫秒)")


# ==================== 多模态问答接口 ====================

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色")
    content: str = Field(..., description="内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class SourceReference(BaseModel):
    """来源引用"""
    doc_id: str = Field(..., description="文档ID")
    doc_name: str = Field(..., description="文档名称")
    page_number: Optional[int] = Field(None, description="页码")
    chunk_ids: List[str] = Field(..., description="引用的块ID列表")
    relevance_score: float = Field(..., description="相关性分数")


class ChatRequest(BaseModel):
    """聊天请求"""
    kb_id: str = Field(..., description="知识库ID")
    question: str = Field(..., description="问题")
    image_data: Optional[bytes] = Field(None, description="问题中的图像数据")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    history: Optional[List[ChatMessage]] = Field(default=[], description="历史消息")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="检索配置")


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="答案")
    sources: List[SourceReference] = Field(..., description="来源引用")
    conversation_id: str = Field(..., description="对话ID")
    response_time_ms: int = Field(..., description="响应时间(毫秒)")
    confidence: float = Field(..., description="答案置信度")


# ==================== 文档解析接口 ====================

class ParseDocumentRequest(BaseModel):
    """解析文档请求"""
    doc_id: str = Field(..., description="文档ID")
    parse_config: Optional[Dict[str, Any]] = Field(None, description="解析配置")


class ParseDocumentResponse(BaseModel):
    """解析文档响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    chunk_count: Optional[int] = Field(None, description="生成的块数量")


# ==================== 健康检查接口 ====================

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    version: str = Field(..., description="系统版本")
    components: Dict[str, str] = Field(..., description="各组件状态")