import base64
import requests
from openai import OpenAI
import pandas as pd

# 数据集下载路径 https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class/files
df = pd.read_csv("./train.csv")
images_path, classes = df['image:FILE'], df['category'] # 0 cat 1 dog

images = []
names = ["cat" if i == 0 else "dog" for i in classes]
for image_path in images_path:
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_str = base64.b64encode(image_data).decode("utf-8")
        image_category = image_path.split('.')[-1]
        data_url = f"data:image/{image_category};base64,{base64_str}"
        images.append(data_url)


# 调用模型
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="any"  # Ollama 任意 key 即可
)
sum_count = len(names)
success_count = 0.0
for image_data, name in zip(images, names):
    response = client.chat.completions.create(
        model="qwen3-vl:8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": "该图片中的动物是什么？ 只输出cat or dog，不要输出其他内容。"}
                ]
            }
        ],
        temperature=0.7,
        max_tokens=512
    )
    print(response.choices[0].message.content)
    if name in response.choices[0].message.content:
        success_count += 1

print(f"Accuracy: {success_count / sum_count}*100")