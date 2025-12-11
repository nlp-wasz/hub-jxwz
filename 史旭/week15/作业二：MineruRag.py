# 对解析的 pdf 文档构建Rag问答（读取md文件，划分chunk，文本信息进行编码，同时保存对应的图片信息）
import asyncio
import os.path

import markdown, glob, re, pathlib
import torch
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled, AsyncOpenAI, \
    OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from PIL import Image
from io import BytesIO
import base64

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("../../../models/BAAI/bge-small-zh-v1.5", trust_remote_code=True)
embedding_model.to(device)


# 获取图片信息
def get_image_url(chunk_content):
    chunk_images = re.findall(r'!\[.*?\]\((.*?)\)', chunk_content)
    # 转换为 绝对路径
    chunk_images_url = [
        pathlib.Path(os.path.abspath(__file__)).parent / "Week15-文档解析与DeepResearch" / "auto" / chunk_image for
        chunk_image in chunk_images]

    return chunk_images_url


# 1.加载.md文件，获取所有信息
def markdown_load(md_path):
    # 读取 md 文件
    with open(md_path, 'r', encoding="utf-8") as f:
        md_read = f.read()

    # 使用 MarkdownTextSplitter 根据标题划分chunk
    md_split = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )

    md_split_chunks = md_split.split_text(text=md_read)

    # 根据 md_split_chunks 构建临时数据库
    md_chunks_info = []
    md_chunks_content = []
    for chunk in md_split_chunks:
        # 获取 标题 和 完整内容
        chunk_title1 = chunk.metadata.get("Header 1", "")
        chunk_title2 = chunk.metadata.get("Header 2", "")
        chunk_content = chunk.page_content

        # 从 chunk_content 中获取图片信息
        chunk_images = get_image_url(chunk_content)

        # 添加到临时数据库
        md_chunks_info.append({
            "chunk_title": chunk_title1 if chunk_title1 else chunk_title2,
            "chunk_content": chunk_content,
            "chunk_images": chunk_images
        })
        md_chunks_content.append(chunk_content)

    return md_chunks_info, md_chunks_content


# 2.使用Qwen Embedding 编码模型，对chunk文本进行编码
def qwen_embedding_text(md_chunks_content):
    # 模型编码
    md_chunks_content_embedding = embedding_model.encode_document(md_chunks_content, show_progress_bar=True,
                                                                  convert_to_tensor=True)

    return md_chunks_content_embedding


# 3.问答（RAG流程） 问题编码 -》 检索TopK -》 Agent问答
async def qa_agent(md_chunks_info, md_chunks_content_embedding, query: str):
    # 对问题进行编码
    query_embedding = embedding_model.encode_query(query, show_progress_bar=True,
                                                   convert_to_tensor=True)

    similar = embedding_model.similarity(query_embedding, md_chunks_content_embedding)

    # 获取 top3
    top3_source, top3_index = torch.topk(similar, 3)

    # 根据 top3 索引下标，获取 md_chunks_info 详细信息（构建Agent）
    openai_client = AsyncOpenAI(
        api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # openai
    sys_prompt = """你是一个专业的文档问答助手，能够结合提供的文本段落和相关图片进行准确回答。
    请根据以下【检索到的上下文】（包含文字和图片）回答用户的问题。
    - 如果图片对理解问题有帮助，请结合图片内容解释；
    - 如果没有图片或图片无关，仅基于文本回答；
    - 回答要简洁、准确、中文。
    """

    openai_prompt = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"#用户提问：\n{query}，请基于以下检索到的上下文回答："},
            ]
        }]
    # 循环添加 message 信息
    images_file_path = []
    for i, idx in enumerate(top3_index[0], start=1):
        md_chunk_info = md_chunks_info[idx.item()]
        openai_prompt[1]["content"].append({
            "type": "text",
            "text": f"#知识库{i}标题：\n{md_chunk_info['chunk_title']}，\n**内容:**\n{md_chunk_info['chunk_content']}\n\n"
        })

        # 记录图片信息
        images_file_path.extend([img_Path.resolve() for img_Path in md_chunk_info["chunk_images"]])

    # 追加 图片信息
    for image_file_path in images_file_path[:3]:
        image_open = Image.open(image_file_path).convert("RGB")
        image_bytes = BytesIO()
        image_open.save(image_bytes, format='JPEG')
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        openai_prompt[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    res = await openai_client.chat.completions.create(
        model="qwen-vl-plus",
        messages=openai_prompt,
        stream=True
    )
    async for chunk in res:
        print(chunk.choices[0].delta.content, end="")


if __name__ == '__main__':
    # 获取 md 文件路径
    gloob_path = glob.glob("./Week15-文档解析与DeepResearch/auto/*.md")

    # 1.加载.md文件，获取所有信息
    md_chunks_info, md_chunks_content = markdown_load(gloob_path[0])

    # 2.使用Qwen Embedding 编码模型，对chunk文本进行编码
    md_chunks_content_embedding = qwen_embedding_text(md_chunks_content)

    # 3.问答（RAG流程）
    query = "简述PaddleOCR"
    asyncio.run(qa_agent(md_chunks_info, md_chunks_content_embedding, query))
