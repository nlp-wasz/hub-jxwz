## 接口定义

### 模型

```text
class BasicResponse(BaseModel):
    """
    基础响应模型
    """
    status: int
    message: str
    data: Optional[Union[dict, list]] = None

class SearchRequest(BaseModel):
    """
    搜索请求模型
    """
    search_type: str = "text2image"
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = 10
```

### 数据管理的接口

#### 上传文档
```text
@app.post("/file/upload", response_model=BasicResponse)
async def upload_file(file: UploadFile = File(...), db=Depends(get_db)):
    """
    上传文档
    
    Args:
        file: 文档文件（支持 "pdf", "txt", "docx", "pptx", "md", "png", "jpg", "jpeg"）
        db: 数据库会话
    
    Returns:
        上传的文档信息
    """
```

#### 获取文档列表
```text
@app.get("/file/list", response_model=BasicResponse)
async def file_list(db=Depends(get_db)):
    """
    获取所有文档列表
    
    Args:
        order: 排序逻辑（插入、更新）
        db: 数据库会话
    
    Returns:
        包含所有文档的列表
    """
```

#### 删除文档
```text
@app.delete("/file/{file_id}", response_model=BasicResponse)
async def delete_file(file_id: int, db=Depends(get_db))
    """
    删除file_id的文档文件
    
    Args:
        file_id: 文档文件id
        db: 数据库会话
    
    Returns:
        删除成功的消息
    
    Raises:
        HTTPException: 当文件不存在时返回404错误
    """
```

### 多模态检索接口
```text
@app.post("/file/search", response_model=BasicResponse)
async def file_search(search_request: SearchRequest, db=Depends(get_db)):
    """
    搜索接口
    支持四种搜索模式：
    1. text2text: 文本搜索文本
    2. text2image: 文本搜索图片
    3. image2text: 图片搜索文本
    4. image2image: 图片搜索图片
    
    Args:
        search_request: 搜索请求参数，包含搜索类型、查询内容和返回数量
        db: 数据库会话
    
    Returns:
        搜索结果列表，按相似度排序
    
    Raises:
        HTTPException: 当请求参数无效时返回400错误
    """
```

### 多模态问答接口
```text
@app.post("/chat", response_model=BasicResponse)
async def chat(query_text: Optional[str], query_image: Optional[str], file_ids: Optional[List[int]], db=Depends(get_db)):
    """
    多模态问答接口
    
    支持基于文本、图片的问答，可结合指定文档内容进行回答，
    
    Args:
        query_text: 文本查询内容（与query_image至少存在一个）
        query_image: 图片查询内容（Base64编码字符串，与query_text至少存在一个）
        file_ids: 指定参考文档ID列表
        db: 数据库会话
    
    Returns:
        返回模型的回答
        
    Raises:
        HTTPException: 当查询参数无效（如文本和图片都为空）时返回400错误
    """
```