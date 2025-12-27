import os
import joblib
import logging
import torch
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertPredictor:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 检查模型文件
        model_files = os.listdir(model_dir)
        logger.info(f"模型目录文件: {model_files}")
            
        try:    
            # 加载模型
            logger.info("开始加载模型...")
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
            logger.info("模型加载成功")
                
            # 加载分词器
            logger.info("开始加载分词器...")
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            logger.info("分词器加载成功")
                
            self.model.to(self.device)  #type:ignore
            self.model.eval()
                
            # 加载标签编码器
            label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                self.id_to_label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
                logger.info(f"标签编码器加载成功，类别: {list(self.label_encoder.classes_)}")
            else:
                logger.warning("Label encoder not found")
                self.label_encoder = None
                self.id_to_label = None
                
                logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Load Model error: {e}")
            raise e

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """预测批量文本"""
        try:
            if not texts:
                return []
            
            # 分词处理
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                confidences, predicted_classes = torch.max(predictions, dim=1)
            
            # 转换为CPU numpy数组
            predicted_classes = predicted_classes.cpu().numpy()
            confidences = confidences.cpu().numpy()
            all_probabilities = predictions.cpu().numpy()
            
            results = []
            for i, (text, cls, conf) in enumerate(zip(texts, predicted_classes, confidences)):
                # 转换标签ID为实际标签
                if self.id_to_label:
                    predicted_label = self.id_to_label.get(int(cls), f"class_{cls}")
                else:
                    predicted_label = f"class_{cls}"
                
                results.append({
                    "text": text,
                    "predicted_label": predicted_label,
                    "predicted_class": int(cls),
                    "confidence": float(conf),
                })
            #print("In the predictor, results:",results)
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """预测单个文本"""
        results = self.predict([text])
        return results[0] if results else {}

# 全局预测器实例
predictor = None

def load_predictor(model_dir: str):
    """加载预测器"""
    global predictor
    predictor = BertPredictor(model_dir)
    return predictor