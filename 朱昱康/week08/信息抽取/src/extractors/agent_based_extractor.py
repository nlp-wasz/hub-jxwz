import openai
import json
import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from config.settings import settings

class AgentBasedExtractor:
    """
    基于智能体（Agent）的文本信息抽取器
    使用函数调用（Function Calling）从文本中抽取领域类别、意图类型和实体标签
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化抽取器
        
        Args:
            api_key: API密钥，如果不提供则从配置中获取
        """
        # 从配置获取API密钥，如果没有提供
        self.api_key = api_key or settings.openai_api_key
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=settings.openai_base_url,
        )
        
        # 加载领域、意图和实体类别
        self.domains = self._load_categories("domains.txt")
        self.intents = self._load_categories("intents.txt")
        self.entities = self._load_categories("entities.txt")
        
        # 创建动态数据模型
        self.ExtractionModel = self._create_dynamic_model()
    
    def _load_categories(self, filename: str) -> List[str]:
        """
        从文件中加载类别列表
        
        Args:
            filename: 文件名
            
        Returns:
            类别列表
        """
        file_path = os.path.join(settings.data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"警告: 文件 {file_path} 未找到")
            return []
    
    def _create_dynamic_model(self):
        """
        创建动态数据模型，根据加载的领域、意图和实体类别
        
        Returns:
            动态创建的Pydantic模型类
        """
        # 创建领域字段的字面量类型
        domain_literals = Literal[tuple(self.domains)]
        
        # 创建意图字段的字面量类型
        intent_literals = Literal[tuple(self.intents)]
        
        # 动态创建字段
        fields = {
            "domain": (domain_literals, Field(description="领域类别")),
            "intent": (intent_literals, Field(description="意图类别")),
            "__annotations__": {}
        }
        
        # 为每个实体类别创建可选字段
        for entity in self.entities:
            fields[entity] = (Optional[str], Field(description=f"实体: {entity}"))
        
        # 创建模型类
        model_name = "IntentDomainNerTask"
        model = type(model_name, (BaseModel,), fields)
        model.__doc__ = "对文本抽取领域类别、意图类型、实体标签"
        
        return model
    
    def extract(self, text: str) -> Optional[BaseModel]:
        """
        从文本中抽取信息
        
        Args:
            text: 输入文本
            
        Returns:
            抽取结果模型实例
        """
        try:
            # 构建用户提示词
            user_prompt = f"请从以下文本中抽取领域类别、意图类型和实体标签：\n\n{text}"
            
            # 构建工具（函数）定义
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": self.ExtractionModel.model_json_schema()['title'],
                        "description": self.ExtractionModel.model_json_schema().get('description', ''),
                        "parameters": {
                            "type": "object",
                            "properties": self.ExtractionModel.model_json_schema().get('properties', {}),
                        },
                    }
                }
            ]
            
            # 调用模型
            response = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "user", "content": user_prompt}],
                tools=tools,
                tool_choice="auto",
                temperature=0.1,  # 降低温度以获得更一致的结果
            )
            
            # 解析响应
            try:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                result = self.ExtractionModel.model_validate_json(arguments)
                return result
            except (IndexError, AttributeError, KeyError) as e:
                print(f"警告: 无法解析工具调用结果: {e}")
                print(f"响应内容: {response.choices[0].message}")
                return None
                
        except Exception as e:
            print(f"抽取过程中发生错误: {e}")
            return None
    
    def extract_to_dict(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中抽取信息并转换为字典格式
        
        Args:
            text: 输入文本
            
        Returns:
            抽取结果字典
        """
        result = self.extract(text)
        if result:
            return result.model_dump()
        return None


# 示例用法
if __name__ == "__main__":
    extractor = AgentBasedExtractor()
    
    # 单个文本抽取示例
    text = "帮我查询下从北京到天津到武汉的汽车票"
    result = extractor.extract(text)
    print("单个文本抽取结果:")
    print(result.model_dump_json(indent=2, ensure_ascii=False))
    
    # 转换为字典格式
    result_dict = extractor.extract_to_dict(text)
    print("\n转换为字典格式:")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))

