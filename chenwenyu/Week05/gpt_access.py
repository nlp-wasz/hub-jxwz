import os
from zhipuai import ZhipuAI

zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=zhipuai_api_key)

resp = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "system", "content": "You are an helpful mathematics assistant. "},
        {"role": "user", "content": "Minkovski Inequity"}
    ],
    stream=False,
    temperature=0.1,
    max_tokens=400
)
print(f"Model:{resp.model}")    #type:ignore
print(f"Content:{resp.choices[0].message.content}") #type:ignore
print(f"Total Tokens: {resp.usage.total_tokens}")   #type:ignore