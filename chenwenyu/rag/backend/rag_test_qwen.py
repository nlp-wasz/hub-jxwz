# rag_test_qwen.py
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Literal,List
from pydantic import BaseModel, Field
import json
from tavily import TavilyClient
import os
from dashscope.aigc import Generation

# 加载环境变量
os.environ["USER_AGENT"] = "rag_test"
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable not set.")
tavily_client = TavilyClient(api_key=tavily_api_key)
print("Tavily client initialized.")

### Build Index
# Set local embedding model
embd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings":True},    # 归一化向量
    model_kwargs={"device": "cpu"},  # Use CPU for local inference
)

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"Split into {len(doc_splits)} document chunks.")

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-test-collection",
    embedding=embd,
    persist_directory="./my_chroma_db"  # 明确指定路径
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3} )   #检索时返回3个最相关结果
print(f"Vectorstore created with {len(doc_splits)} document chunks.")

#自动保存

# 验证加载成功
print("集合数量:", vectorstore._collection.count())

### 通义千问LLM封装
class QwenLLM:
    def __init__(self, model_name="qwen-turbo", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
    
    def invoke(self, prompt: str) -> str:
        """Ivoke QWen API"""
        prompt = f"Please respond in English only. {prompt}"
        try:
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                top_p=0.8,
                result_format='text',
                stream=False  # 明确关闭流式输出
            )
            
            # 检查是否是生成器
            if hasattr(response, '__iter__') and not hasattr(response, 'status_code'):
                # 如果是生成器，获取第一个（也是唯一一个）响应
                for resp in response:
                    # 安全地访问属性
                    status_code = getattr(resp, 'status_code', None)
                    if status_code == 200:
                        output = getattr(resp, 'output', None)
                        if output and hasattr(output, 'text'):
                            return output.text
                        else:
                            return f"响应格式异常: {str(output)}"
                    else:
                        code = getattr(resp, 'code', '未知')
                        message = getattr(resp, 'message', '未知错误')
                        return f"API错误: {code} - {message}"
                return "生成器未返回任何响应"
            else:
                # 直接是响应对象 - 安全地访问属性
                status_code = getattr(response, 'status_code', None)
                if status_code == 200:
                    output = getattr(response, 'output', None)
                    if output and hasattr(output, 'text'):
                        return output.text
                    else:
                        return f"响应格式异常: {str(output)}"
                else:
                    code = getattr(response, 'code', '未知')
                    message = getattr(response, 'message', '未知错误')
                    return f"API错误: {code} - {message}"
                
        except Exception as e:
            return f"调用通义千问API时出错: {str(e)}"

# 更简洁的版本，避免类型检查问题
class QwenLLMSimple:
    def __init__(self, model_name="qwen-turbo", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
    
    def invoke(self, prompt: str) -> str:
        """简化版API调用 - 完全安全的属性访问"""
        try:
            result = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                top_p=0.8,
                result_format='text',
                stream=False
            )
            
            # 统一处理生成器和直接响应
            if hasattr(result, '__iter__'):
                # 转换为列表并取第一个元素
                responses = list(result)
                if responses:
                    response_obj = responses[0]
                else:
                    return "API返回空响应"
            else:
                response_obj = result
            
            # 完全安全的属性访问方式
            # 1. 首先检查是否有output属性
            if hasattr(response_obj, 'output'):
                output = response_obj.output    # type: ignore
                # 2. 检查output是否有text属性
                if hasattr(output, 'text'):
                    return output.text
                else:
                    # 如果output没有text属性，安全地转换为字符串
                    try:
                        output_str = str(output)
                        return f"输出内容: {output_str}"
                    except:
                        return "无法转换输出内容为字符串"
            else:
                # 如果没有output属性，安全地转换整个响应对象为字符串
                try:
                    response_str = str(response_obj)
                    return f"完整响应: {response_str}"
                except:
                    return "无法转换响应对象为字符串"
                
        except Exception as e:
            return f"调用API时出错: {str(e)}"


# 使用类型注解来避免静态检查错误
class QwenLLMTypeSafe:
    def __init__(self, model_name="qwen-turbo", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
    
    def invoke(self, prompt: str) -> str:
        """类型安全的API调用"""
        from typing import Generator, Union
        from dashscope.api_entities.dashscope_response import GenerationResponse
        
        try:
            result: Union[GenerationResponse, Generator[GenerationResponse, None, None]] = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                top_p=0.8,
                result_format='text',
                stream=False
            )
            
            # 处理可能的生成器
            if isinstance(result, Generator):
                response_obj: GenerationResponse = next(result)
                # 确保生成器被消耗完
                for _ in result:
                    pass
            else:
                response_obj: GenerationResponse = result
            
            # 现在response_obj肯定是GenerationResponse类型
            if response_obj.status_code == 200:
                if hasattr(response_obj.output, 'text'):
                    return response_obj.output.text
                else:
                    return f"响应格式异常: {str(response_obj.output)}"
            else:
                return f"API错误: {response_obj.code} - {response_obj.message}"
                
        except StopIteration:
            return "生成器为空"
        except Exception as e:
            return f"调用API时出错: {str(e)}"

# 初始化通义千问LLM
llm = QwenLLM(model_name="qwen-max", temperature=0.1)
print("通义千问LLM初始化完成。")

### Router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore."
    )
# 自定义结构化输出提示
structured_prompt = """You are an expert at routing user questions. 
The vectorstore contains documents about agents, prompt engineering, and adversarial attacks.

For questions on these topics, respond with JSON containing: {{"datasource": "vectorstore"}}
For all other questions, respond with JSON containing: {{"datasource": "web_search"}}

Respond ONLY with valid JSON, no other text.

Question: {question}
Response:"""

vectorstore_template = """Answer the question based on the following context information. Please provide accurate and detailed answers.

Context information:
{context}

Question: {question}

Please answer in English:"""

web_search_template = """Answer the question based on the following web search results. Please provide comprehensive and accurate answers.

Web search results:
{search_results}

Question: {question}

Please answer in English:"""

def structured_route_question(question: str) -> str:
    prompt = structured_prompt.format(question=question)
    
    try:
        response = llm.invoke(prompt).strip()
        
        # 尝试解析JSON响应
        try:
            result = json.loads(response)
            if result.get("datasource") == "vectorstore":
                return "vectorstore"
            else:
                return "web_search"
        except json.JSONDecodeError:
            # 如果JSON解析失败，回退到文本分析
            return parse_text_response(response, question)
            
    except Exception as e:
        print(f"LLM路由失败: {e}")
        return rule_based_router(question)

def parse_text_response(response: str, question: str) -> str:
    """解析文本响应"""
    response_lower = response.lower()
    
    if "vectorstore" in response_lower:
        return "vectorstore"
    elif "web_search" in response_lower or "web search" in response_lower:
        return "web_search"
    else:
        return rule_based_router(question)

def rule_based_router(question: str) -> str:
    """备用规则路由"""
    question_lower = question.lower()
    expert_topics = ["agent", "prompt engineering", "adversarial attack", "llm", "language model"]
    
    for topic in expert_topics:
        if topic in question_lower:
            return "vectorstore"
    return "web_search"

print("Test question router:")
questions = [
    "What is a language model?",
    "How do I create an agent with LangChain?",
    "Explain prompt engineering techniques.",
    "Who is the 47th president of the United States of America?"]

for q in questions:
    route = structured_route_question(q)
    print(f"Question: {q}\nRouted to: {route}\n")
print("Question router test completed.")

### Web Search
def web_search(query: str, max_results: int = 3) -> str:
    """使用Tavily进行网络搜索"""
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        search_results = response.get('results', [])
        
        if not search_results:
            return "没有找到相关的网络搜索结果。"
        
        # 格式化搜索结果
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            content = result.get('content', '无内容')
            # 限制内容长度
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted_results.append(
                f"来源[{i}]: {result.get('title', '无标题')}\n"
                f"内容: {content}\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        print(f"网络搜索失败: {e}")
        return f"网络搜索时出现错误: {str(e)}"


### RAG链
def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_from_vectorstore(question: str) -> List:
    """从向量库检索文档"""
    try:
        docs = retriever.invoke(question)
        return docs
    except Exception as e:
        print(f"检索失败: {e}")
        return []

### 主函数
def rag_system(question: str) -> str:
    """完整的RAG系统"""
    print(f"处理问题: {question}")
    
    try:
        # 路由决策
        datasource = structured_route_question(question)
        print(f"路由到: {datasource}")
        
        if datasource == "vectorstore":
            print("使用向量库检索...")
            
            # 检索相关文档
            relevant_docs = retrieve_from_vectorstore(question)
            if not relevant_docs:
                print("本地知识库中没有相关信息，切换到网络搜索")
                datasource = "web_search"
            else:
                context = format_docs(relevant_docs)
                # 限制上下文长度
                if len(context) > 1500:
                    context = context[:1500] + "..."
                prompt = vectorstore_template.format(context=context, question=question)
        
        if datasource == "web_search":
            print("使用网络搜索...")
            
            # 进行网络搜索
            search_results = web_search(question)
            prompt = web_search_template.format(search_results=search_results, question=question)
        
        # 生成回答
        print("生成回答中...")
        response = llm.invoke(prompt)
        return response
        
    except Exception as e:
        print(f"系统错误: {e}")
        return f"处理问题时出现系统错误: {str(e)}"

def check_api_connection():
    """Check QWen API connection"""
    try:
        print("Check QWen API Connection...")
        test_prompt = "你好，请回复'连接成功'"
        response = llm.invoke(test_prompt)
        if "successful" in response or "你好" in response:
            print("✓ QWen API Connection Successful")
            return True
        else:
            print(f"API响应异常: {response}")
            return False
    except Exception as e:
        print(f"✗ QWen API Connection Failure: {e}")
        print("Please Check:")
        print("1. Set DASHSCOPE_API_KEY environment variable correctly")
        print("2. Network connectivity")
        print("3. API Key usage limits")
        return False

### 测试函数
def test_rag_system():
    """测试RAG系统"""
    test_questions = [
        "What is a language model?",  # 应该使用Web Search
        "How do I create an agent with LangChain?",  # 应该使用向量库
        "Explain prompt engineering techniques.",  # 应该使用向量库
        "Who is the 47th president of the United States?",  # 应该使用网络搜索
        "2025-2026年大模型的发展趋势是什么?",  # 应该使用网络搜索
    ]
    
    for question in test_questions:
        print("=" * 50)
        print(f"问题: {question}")
        answer = rag_system(question)
        print(f"回答: {answer}")
        print("=" * 50)
        print("\n")

### FastAPI 后端接口
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI(title="RAG System API")

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],   # localhost:5173是Vite默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, origin: Optional[str] = Header(None)):
    """处理用户问题"""
    try:
        print(f"Received question: {request.question}")
        answer = rag_system(request.question)
        return QuestionResponse(answer=answer)
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "RAG System API"}

@app.get("/")
async def root():
    return {"message": "RAG System API is running", "docs": "http://localhost:8000/docs"}

# 添加OPTIONS请求处理
@app.options("/api/ask")
async def options_ask():
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.options("/api/health")  
async def options_health():
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.middleware("http")
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n=== Incoming Request ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers:")
    for header, value in request.headers.items():
        print(f"  {header}: {value}")
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Response status: {response.status_code}")
        print(f"Process time: {process_time:.2f}s")
        print("=======================\n")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        print(f"Request error: {str(e)}")
        print(f"Process time: {process_time:.2f}s")
        print("=======================\n")
        raise

if __name__ == "__main__":
    # 检查API连接
    if not check_api_connection():
        exit(1)
    
    print("=" * 50)
    print("RAG System API Server")
    print("=" * 50)
    print("Frontend: http://localhost:3000")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/api/health")
    print("=" * 50)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)