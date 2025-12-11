import fitz  # PyMuPDF
import os

# PDF文件路径
pdf_dir = "/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week14/作业/pdf"

# 存储所有PDF内容
pdf_contents = {}

# 读取所有PDF文件
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        try:
            # 打开PDF文件
            doc = fitz.open(file_path)
            text = ""
            # 提取每一页的文本
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            pdf_contents[filename] = text
            doc.close()
        except Exception as e:
            print(f"无法读取文件 {filename}: {e}")

# 打印每个PDF的部分内容以了解内容
for filename, content in pdf_contents.items():
    print(f"\n=== {filename} ===")
    # 只打印前500个字符以避免输出过长
    print(content[:500] + "..." if len(content) > 500 else content)