# 使用RAG检索符合问题的 TOPK，构建提示词工程，LLM回答
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("../../../models/Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True,
                                      device=device)


# 1.读取 pdf_content_latex.csv 文件（获取pdf内容 和 LaTeX公式）
def read_csv():
    pdf_content_latex = pd.read_csv("./pdf_content_latex.csv", header=0, encoding='utf-8')

    return pdf_content_latex


# Qwen3-Embedding 文本编码
def pdf_content_embedding_func(pdf_content_latex):
    pdf_content = pdf_content_latex["page_content"].tolist()

    # 文本编码
    pdf_content_embedding = embedding_model.encode_document(pdf_content, convert_to_tensor=True)

    return pdf_content_embedding


# LLM 问答
def llm_qa(question):
    # 对问题进行编码
    question_embedding = embedding_model.encode_query(question, convert_to_tensor=True)

    # 获取 pdf文档内容 编码信息
    pdf_content_latex = read_csv()
    pdf_content_embedding = pdf_content_embedding_func(pdf_content_latex)

    # RAG 检索和排序
    embedding_res = embedding_model.similarity(question_embedding, pdf_content_embedding)

    # 获取 TOP_3
    top_3_score, top_3_index = torch.topk(embedding_res, 3)

    # 根据 top_3_index 构建prompt提示词工程
    prompt = f"""
    你是一个各种行业的建模专家，擅长结合科学文献和数学公式进行定量分析。

    用户的问题是：
    「{question}」

    以下是与问题最相关的三段知识库内容（按相关性排序）：

    """

    for i, idx in enumerate(top_3_index[0]):
        idx = idx.item()

        # 获取 pdf内容 和 对应的LaTeX公式
        page_content = pdf_content_latex.iloc[idx]["page_content"]
        latex_content = pdf_content_latex.iloc[idx]["latex_content"]

        prompt += f"top{i + 1}知识库内容：\n{page_content}\n   对应的LaTeX公式信息：{latex_content} \n\n"

    prompt += """
    请完成以下任务：
    1. 仔细分析上述参考资料，找出与问题直接相关的数学模型或公式；
    2. 若存在多个公式，请选择最匹配当前问题情境的一个；
    3. 使用该公式，代入用户提供的参数（如内禀增长率、环境承载力、当前种群数量、食物供应量等）进行计算；
    4. 展示完整的推理过程（包括公式引用、变量代入、单位说明）；
    5. 最终给出明确的数值结果和简要解释。

    注意：
    - 如果参考资料中没有足够信息，请明确说明“无法根据现有资料计算”；
    - 所有数学表达式必须使用 LaTeX 格式（如 $x = y$ 或 $$z = \\frac{{a}}{{b}}$$）；
    - 请启用深度推理，逐步思考，避免跳步。
    """

    # 调用 LLM
    llm = OpenAI(
        api_key='sk-04ab3d7290e243dda1badc5a1d5ac858',  # 秘钥
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'  # 模型地址
    )

    llm_res = llm.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=True,
        timeout=120,
        extra_body={
            "enable_thinking": True
        }

    )

    for chunk in llm_res:
        yield chunk.choices[0].delta.content

# if __name__ == "__main__":
#     for chunk in llm_qa(
#             "在水产养殖管理中，已知内禀增长率和环境承载力，当当前种群数量为200、食物供应量为30时，下一个时间步的种群数量是多少？"):
#         print(chunk, end="")
