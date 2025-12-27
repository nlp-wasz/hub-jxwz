import openai

client = openai.OpenAI(
    api_key = "sk-1***",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model= "qwen-vl-plus",
    messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://ts1.tc.mm.bing.net/th/id/R-C.59ab522dec1ff1768ae9b6fabbaee692?rik=lg4oEOsjzR%2fhsA&riu=http%3a%2f%2fpic1.bbzhi.com%2fdongwubizhi%2fmaorongrongxiaogougouxiezhenbizhi%2fanimal_white_baby_dogs_2_1920x1200_5739_10.jpg&ehk=1UgNhoOWgvsHCeT3SmAqG5s%2fMqpe4Ue4WNbaS8bpG%2fU%3d&risl=&pid=ImgRaw&r=0"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请识别这张图片中的动物是狗还是猫？只回答'狗'或'猫'，不要解释。"
                        }
                    ]
                }
            ],
    max_tokens= 10
)
print(completion.model_dump_json())
