# PDF 文档解析，获取PDF文档内容，并将其转换为LaTeX公式
import pdfplumber, glob, fitz, pandas as pd, tqdm
from pathlib import Path
from openai import OpenAI


# 1.加载PDF，获取其中内容 并解析为LaTeX文档
# 之后将文档内容和LaTeX 保存为csv文档信息
def pdf_analysis():
    # 打开文档
    path = Path("../documents")

    pdf_content = []
    # 遍历所有文件并解析
    for p in path.iterdir():
        # 获取文档内容
        p_content = ""

        if p.name.endswith(".pdf"):
            p_page = fitz.open(p)

            for page in p_page:
                p_content += page.get_text("text")  # 或 "blocks", "dict" 获取结构
        else:
            with open(p, "r", encoding="utf-8") as f:
                p_content += f.read().strip()

        pdf_content.append(p_content)

    return pdf_content


# 2.生成LaTeX公式
def generator_latex(pdf_content):
    pdf_content_latex = []

    # 加载 OpenAI
    llm = OpenAI(
        api_key='sk-04ab3d7290e243dda1badc5a1d5ac858',  # 秘钥
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'  # 模型地址
    )

    prompt = """
    你是一名专业的科学排版专家，擅长将含数学表达式的自然语言文本转换为标准 LaTeX 格式。

    请完成以下任务：
    1. 识别原文中所有数学公式、符号、表达式或科学记法；
    2. 将它们转换为正确的 LaTeX 代码；
    3. 不要添加任何解释、注释或额外内容，仅输出转换后的LaTeX公式内容。

    原文如下：
    {page_content}
    """

    for page_content in tqdm.tqdm(iterable=pdf_content, desc="正在解析LaTeX公式", total=len(pdf_content)):
        llm_res = llm.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "user", "content": prompt.format(page_content=page_content)}
            ],
            temperature=0.1,
            stream=True,
            timeout=120,
            extra_body={
                "enable_thinking": True
            }
        )

        latex_content = ""
        for chunk in llm_res:
            latex_content += chunk.choices[0].delta.content

            print(chunk.choices[0].delta.content, end="")

        pdf_content_latex.append({
            "page_content": page_content,
            "latex_content": latex_content
        })

    return pdf_content_latex


# 3.保存 pdf解析内容 和 对应的LaTeX
def save_csv(pdf_content_latex):
    df = pd.DataFrame(pdf_content_latex)
    df.to_csv("./pdf_content_latex.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    pdf_content = pdf_analysis()

    pdf_content_latex = generator_latex(pdf_content)

    save_csv(pdf_content_latex)
