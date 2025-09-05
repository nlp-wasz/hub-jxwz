"""
外卖评价BERT分类模型
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Union, List
import os

class WaimaiBertClassifier:
    def __init__(self, model_path="./models/waimai-bert"):
        """
        初始化外卖评价BERT分类器
        
        Args:
            model_path: 微调后的BERT模型路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 50
        
        # 标签映射
        self.id_to_label = {0: "负面", 1: "正面"}
        self.label_to_id = {"负面": 0, "正面": 1}
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path, 
                num_labels=2
            )
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            print(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请先运行训练脚本生成微调后的模型")
            raise e
    
    def predict_single(self, text: str) -> dict:
        """
        对单条文本进行情感分类预测
        
        Args:
            text: 待分类的文本
            
        Returns:
            dict: 包含预测结果和置信度的字典
        """
        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        return {
            "text": text,
            "label": self.id_to_label[predicted_id],
            "label_id": predicted_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                "负面": round(probabilities[0][0].item(), 4),
                "正面": round(probabilities[0][1].item(), 4)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """
        对多条文本进行批量预测
        
        Args:
            texts: 待分类的文本列表
            
        Returns:
            List[dict]: 预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def predict(self, text_input: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        通用预测接口
        
        Args:
            text_input: 单条文本或文本列表
            
        Returns:
            预测结果
        """
        if isinstance(text_input, str):
            return self.predict_single(text_input)
        elif isinstance(text_input, list):
            return self.predict_batch(text_input)
        else:
            raise ValueError("输入必须是字符串或字符串列表")

# 全局模型实例（用于FastAPI服务）
_classifier = None

def get_classifier():
    """获取分类器实例（单例模式）"""
    global _classifier
    if _classifier is None:
        _classifier = WaimaiBertClassifier()
    return _classifier

def predict_sentiment(text: Union[str, List[str]]) -> Union[dict, List[dict]]:
    """
    外卖评价情感分类预测接口
    
    Args:
        text: 待分类的文本或文本列表
        
    Returns:
        预测结果
    """
    classifier = get_classifier()
    return classifier.predict(text)

# 测试代码
if __name__ == "__main__":
    # 测试用例
    test_texts = [
        "菜品很好吃，送餐也很快，五星好评！",
        "味道一般，而且送餐太慢了，差评",
        "性价比不错，下次还会点",
        "菜品质量不行，不推荐"
    ]
    
    try:
        classifier = WaimaiBertClassifier()
        
        # 单条预测测试
        print("=== 单条预测测试 ===")
        result = classifier.predict("菜品很好吃，服务态度也不错！")
        print(result)
        
        # 批量预测测试
        print("\n=== 批量预测测试 ===")
        results = classifier.predict(test_texts)
        for result in results:
            print(f"文本: {result['text']}")
            print(f"预测: {result['label']} (置信度: {result['confidence']})")
            print("-" * 50)
            
    except Exception as e:
        print(f"测试失败: {e}")
        print("请先运行训练脚本生成微调后的模型")
