import os
import dashscope
from pathlib import Path
import fitz  # PyMuPDF

# https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2860683
api_key = "sk-f8c04a4037fe4d6fbaffb6787e49a989"

pdf_path = r"D:\BaiduSyncdisk\badou全\第14周：Reasoning模型应用\Week14\Week14\07-文档公式解析与智能问答\documents\demo.pdf"

# Convert PDF to image
doc = fitz.open(pdf_path)
page = doc.load_page(0)  # load the first page
pix = page.get_pixmap()
image_path = pdf_path.replace(".pdf", ".png")
pix.save(image_path)
doc.close()

file_path = image_path
# file_url = Path(file_path).as_uri()

messages = [{
    "role": "user",
    "content": [
        {
            "image": file_path,
            "min_pixels": 32 * 32 * 3,
            "max_pixels": 32 * 32 * 8192,
            "enable_rotate": False
        }
    ]
}]

response = dashscope.MultiModalConversation.call(
    api_key=api_key,
    model='qwen-vl-ocr-latest',
    messages=messages,
    ocr_options={"task": "advanced_recognition"}
)
print(response)
print("\n\n\n高精度识别：")
if response.status_code == 200:
    print(response.output.choices[0].message.content[0]["text"])
else:
    print(f"Error: {response.code} - {response.message}")


response = dashscope.MultiModalConversation.call(
    api_key=api_key,
    model='qwen-vl-ocr-latest',
    messages=messages,
    ocr_options={"task": "document_parsing"}
)
print("\n\n\n文档解析：")
if response.status_code == 200:
    print(response.output.choices[0].message.content[0]["text"])
else:
    print(f"Error: {response.code} - {response.message}")

# Clean up temporary image
if os.path.exists(image_path):
    os.remove(image_path)


