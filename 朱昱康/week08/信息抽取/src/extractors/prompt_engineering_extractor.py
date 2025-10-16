import openai
import json
import os
from typing import Dict, Any, Optional, List
from config.settings import settings

class PromptEngineeringExtractor:
    """
    基于提示词工程的文本信息抽取器
    从文本中抽取领域类别、意图类型和实体标签
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
        
        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()
    
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
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词
        
        Returns:
            系统提示词字符串
        """
        domains_str = " / ".join(self.domains)
        intents_str = " / ".join(self.intents)
        entities_str = " / ".join(self.entities)
        
        return f"""你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：{domains_str}
- 待选的意图类别：{intents_str}
- 待选的实体标签：{entities_str}

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

```json
{{
    "domain": "",
    "intent": "",
    "slots": {{
      "实体标签": "实体名词",
    }}
}}
```
"""
    
    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中抽取信息
        
        Args:
            text: 输入文本
            
        Returns:
            抽取结果字典，包含domain、intent和slots
        """
        try:
            completion = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,  # 降低温度以获得更一致的结果
            )
            
            result_text = completion.choices[0].message.content
            
            # 尝试解析JSON结果
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON结果: {result_text}")
                return None
                
        except Exception as e:
            print(f"抽取过程中发生错误: {e}")
            return None
    
# 示例用法
if __name__ == "__main__":
    extractor = PromptEngineeringExtractor()
    
    # 单个文本抽取示例
    text = "糖醋鲤鱼怎么做啊？你只负责吃，c则c。"
    result = extractor.extract(text)
    print("单个文本抽取结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))