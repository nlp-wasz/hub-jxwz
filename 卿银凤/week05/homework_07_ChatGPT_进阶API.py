import openai
import json

client = openai.OpenAI(
    api_key="sk-WkCbMVOViwqUVVdD97E9E88612A14071A40213E24c2989Ab",
    base_url="https://openkey.cloud/v1"
)



response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content":
"""
请帮我进行文本分类，判断下面的文本是正向情感还是负面情感。请直接输出类别，不要有其他输出，可选类别：正/负

我今天很开心。
"""}],
    stream=False,
    logprobs = True,
    top_logprobs = 5
)
print(response)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "今天是9月11日，帮我查一下成都昨天有哪些演唱会？"},
    ],
    functions=[
        {
            "name": "get_movie",
            "description": "查询城市在昨天的演唱会",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市"
                    },
                    "date_str": {
                        "type": "string",
                        "description": "日期"
                    },
                },
                "required": ["city", "date_str"]
            }
        }
    ]
)

print(response.model_dump_json(indent=4))

def get_movie(city, date_str):
    print(f"{city}  {date_str} 有几场演唱会")


if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    function_args_str = response.choices[0].message.function_call.arguments

    # 将 JSON 字符串解析为 Python 字典
    function_args = json.loads(function_args_str)

    print(f"\n模型请求调用函数: {function_name}")
    print(f"参数: {function_args}\n")

    # --- 推荐的、更安全的方法 ---
    # 使用函数映射来调用本地函数，这比 eval 更安全
    function_map = {
        "get_movie": get_movie
    }

    if function_name in function_map:
        local_function = function_map[function_name]
        result = local_function(**function_args)
        print(result)
