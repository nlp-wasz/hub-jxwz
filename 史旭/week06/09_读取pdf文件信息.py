import pdfplumber

# 通过 pdfplumber 库，操作pdf文件
# 1.open() 读取pdf文件
pdf = pdfplumber.open("./data/汽车知识手册.pdf")
print(f"pdf对象信息：{pdf}")

# 2.pages属性：获取 pdf 每一页对象信息
pages = pdf.pages
print(f"每一页page对象信息：{pages}")

# 3.pages[0]：通过索引或切片，获取第一页 对象信息
page_one = pages[0]
print(f"第一页page对象信息：{page_one}")
print(f"第一页page对象类型：{type(page_one)}")

# 4.extract_text()：提取文本内容
text = page_one.extract_text()
print("\n第一页文本内容：")
print(text)

# 5.extract_table()：提取表格内容
table = page_one.extract_tables()
print("\n第一页表格内容：")
print(table)
