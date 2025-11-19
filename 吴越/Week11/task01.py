from pydantic import BaseModel,Field
from typing import Union,List,Any,Tuple,Dict
from datetime import datetime
from typing_extensions import Annotated

from fastapi import FastAPI,File,UploadFile,Form


class KnowledgeRequest(BaseModel):
    knowledge_id:int=Field(description="知识库ID")
    category:str=Field(description="知识库类别")
    title:str=Field(description="知识库标题")


class KnowledgeRseponse(BaseModel):
    request_id:str=Field(description="请求ID")
    knowledge_id:int=Field(description="知识库id")
    category:str=Field(description="知识库类别")
    title:str=Field(description="知识库标题")
    response_code:int=Field(description='响应代码，用于表示成功或错误状态')
    response_msg:str=Field(description='响应信息，详细描述响应状态或错误信息')
    process_status:str=Field(description="处理状态，例如‘completed'、'pending'或'failed'")
    process_time:float=Field(description="处理请求耗时（秒）")


class DocumentRequest(BaseModel):

    knowledge_id:int=Field(description="知识库ID")
    title:str=Field(description="文档标题")
    category:str=Field(description="文档类别")
    document_id:int=Field(description="文档ID")

class DocumentResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    document_id:int=Field(description="文档ID")
    category:str=Field(description="文档类型")
    knowledge_id:int=Field(description="知识库类别")
    file_type:str=Field(description="数据类型")
    response_code: int = Field(description='响应代码，用于表示成功或错误状态')
    response_msg: str = Field(description='响应信息，详细描述响应状态或错误信息')
    process_status: str = Field(description="处理状态，例如‘completed'、'pending'或'failed'")
    process_time: float = Field(description="处理请求耗时（秒）")


class MultiRAGrequest(BaseModel):
    knowledge_id:int=Field(description="知识库ID")
    message:List[Dict]=Field(description="问题内容")

class MultiRAGresponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]=Field(description="模型返回信息")
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")



app=FastAPI()

@app.get("/health")
async def health_check():
    """
    后端健康检查接口
    返回服务状态信息（程序的启动时间，是否连接到关系型数据库，是否连接到milvus）
    """
    return{
        "status":"healthy",
        "service":"Product Search and Matching Service"
    }

@app.get('/get_knowledgeDatabase',description='知识库查询',response_model=KnowledgeRseponse)
def get_knowledgeDatabase(req:KnowledgeRequest):

    #根据请求的的知识库ID查看知识库是否存在

    return KnowledgeRseponse(
        response_code=200,
        response_msg="知识库查询成功！",
    )

@app.post('/add_knowledgeDatabase',description="知识库添加",response_model=KnowledgeRseponse)
def add_knowledgeDatabase(req:KnowledgeRequest):

    #在知识库中添加一条记录

    return KnowledgeRseponse(
        response_code=200,
        response_msg="知识库添加成功"
    )

@app.delete('/delete_knowledgeDatabase',description="知识库删除",response_model=KnowledgeRseponse)
def delete_knowledgeDatabase(req:KnowledgeRequest):

    #根据知识库的ID删除知识库

    return  KnowledgeRseponse(
        response_code=200,
        response_msg="知识库删除成功！"
    )

@app.get('/get_knowledgeDocument',description="文档查询",response_model=DocumentResponse)
def get_knowledgeDocument(req:DocumentRequest):

    #根据文档ID查看文档是否存在
    return DocumentResponse(
        response_code=200,
        response_msg="文档查询成功！"
    )

@app.post('/add_knowledgeDocument',description='文档添加',response_model=DocumentResponse)
def add_knowledgeDocument(req:DocumentRequest):
    #根据知识库ID判断知识库是否存在，将文档数据信息保存到关系型数据库，并解析上传的文档到向量数据库，并保存文件到本地

    return DocumentResponse(
        response_code=200,
        response_msg="文档添加成功"
    )

@app.delete('/delete_knowledgeDocument',description='文档删除',response_model=DocumentResponse)
def delete_knowledgeDocument(req:DocumentRequest):
    #根据文档ID删除关系型数据库的文档以及向量数据库的文档
    return DocumentResponse(
        response_code=200,
        response_msg="文档删除成功"
    )


@app.post('/chat',description='多模态问答',response_model=MultiRAGresponse)
def chat(req:MultiRAGrequest):
    #对用户的提问进行编码
    #将检索的结果进行多路召回，重排序
    #获取原文内容和图片信息，构建上下文信息，调用多模态模型生成回答
    return MultiRAGresponse(
        response_code=200,
        response_msg='图文问答'
    )








