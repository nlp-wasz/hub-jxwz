import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional
from typing_extensions import Literal

# 创建抽取智能体
class ExtractionAgent:
    def __init__(self, api_key, model_name: str):
        self.client=OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            )
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        # 增强提示词
        enhanced_prompt = f"""
        请仔细分析以下文本，识别领域、意图和所有相关的实体信息。
        特别注意提取：
        - 应用名称（如：微信、支付宝、携程、网易云音乐、等）
        - 歌曲名称、歌手名称
        - 城市名称、地点信息
        - 菜名(如:爆椒牛柳、烩菜心、回锅肉等）实体
        - 时间信息（如：三分钟后、明天、下午三点等）
        - 实体若有多个则都提取出来
        文本：{user_prompt}
        """

        messages = [
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "parameters": response_model.model_json_schema()
                    },
                }
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            print(f"调试 - 完整返回: {response}")
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                print(f"调试 - 原始返回: {arguments}")
                
                # 手动处理 JSON 数据
                data = json.loads(arguments)
                
                # 确保所有字段都存在
                for field_name in response_model.model_fields:
                    if field_name not in data:
                        field_info = response_model.model_fields[field_name]
                        # 检查是否为列表类型
                        if (hasattr(field_info.annotation, '__origin__') and 
                            field_info.annotation.__origin__ is list):
                            data[field_name] = []
                        else:
                            data[field_name] = None
                    # 确保列表字段不为 None
                    elif data[field_name] is None:
                        field_info = response_model.model_fields[field_name]
                        if (hasattr(field_info.annotation, '__origin__') and 
                            field_info.annotation.__origin__ is list):
                            data[field_name] = []
                
                return response_model.model_validate(data)
            else:
                print('没有找到工具调用')
                return None    
        except json.JSONDecodeError as e:
            print(f'JSON 解析错误: {e}')
            print(f'原始参数: {arguments}')
            return None
        except Exception as e:
            print(f'其他错误: {e}')
            return None


class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus','navigation', 'cooking', 'other'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY', 'PLAY', 'NAVIGATION','ORDER','OTHER'] = Field(description="意图")
    #实体字段
    Src: Optional[str] = Field(description="出发地")
    Des: Optional[str] = Field(description="目的地")
    App: Optional[str] = Field(description="应用名称列表,如:微信、支付宝、携程、网易云音乐、B站等")
    Song: Optional[str] = Field(description="歌曲名称列表")
    Artist: Optional[str] = Field(description="歌手名称列表")
    City: Optional[str] = Field(description="城市名称列表")
    Location: Optional[str] = Field(description="地点信息")
    DishName: Optional[str] = Field(description="菜名列表")
    Datetime: Optional[str] = Field(description="时间信息，如:三分钟后、明天、早上、下午三点等")

# Sentence generator
def sentences_generator_expr(filename):
    """使用生成器表达式"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            return iter(lines)
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return iter([])

# Entry
if __name__ == "__main__":  
    api_key = os.getenv('ZHIPUAI_API_KEY')

    sentences_gen = sentences_generator_expr('sentences.txt')

    for sentence in sentences_gen:
        print(f"分析句子: {sentence}")
        result = ExtractionAgent(api_key, model_name="glm-4").call(sentence, IntentDomainNerTask)
        if result:
            print(f"分析结果:")
            print(f"  领域: {result.domain}")
            print(f"  意图: {result.intent}")
            
            # 使用字典推导式过滤空值
            non_empty_fields = {
                field: value for field, value in result.model_dump().items() 
                if field not in ['domain', 'intent'] and value is not None
            }
            
            for field_name, field_value in non_empty_fields.items():
                field_info = result.model_fields[field_name]
                description = field_info.description or field_name
                print(f"  {description}: {field_value}")
                
            # 如果没有其他非空字段
            if not non_empty_fields:
                print("  无其他实体信息")
        else:
            print("分析失败")
        print("-" * 50)