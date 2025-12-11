# Markdown RAG问答系统

这是一个基于Streamlit的Markdown文档RAG问答系统，使用阿里云Qwen-Max大模型和Milvus向量数据库。

## 功能特性

- 上传Markdown文件并自动解析
- 使用BGE模型进行文本向量化
- 存储到Milvus向量数据库
- 基于RAG的问答功能
- Streamlit用户界面

## 环境要求

- Python 3.10+
- Milvus数据库
- 阿里云DashScope API密钥

## 安装步骤

1. 安装依赖:
```bash
pip install -r requirements_rag.txt
```

2. 设置环境变量:
```bash
export DASHSCOPE_API_KEY="your_api_key"
export MILVUS_URI="your_milvus_uri"  # 例如: http://localhost:19530
export MILVUS_TOKEN="your_token"     # 如果需要认证
```

3. 运行应用:
```bash
streamlit run rag_app.py
```

## Docker部署

构建镜像:
```bash
docker build -t markdown-rag-app .
```

运行容器:
```bash
docker run -p 8501:8501 \
  -e DASHSCOPE_API_KEY="your_api_key" \
  -e MILVUS_URI="your_milvus_uri" \
  markdown-rag-app
```

## 使用说明

1. 在左侧边栏上传Markdown文件
2. 系统会自动解析并存储到向量数据库
3. 在主界面输入问题获取基于文档的回答
4. 可以查看参考的文档片段

## 技术栈

- **前端**: Streamlit
- **向量数据库**: Milvus
- **嵌入模型**: BGE (bge-small-zh-v1.5)
- **大模型**: 阿里云Qwen-Max
- **框架**: LangChain
