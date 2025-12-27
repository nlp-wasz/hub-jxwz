import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os


class TextClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2, dropout=0.1):
        super(TextClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        output = self.dropout(pooled_output)
        return self.classifier(output)


class TextClassificationModel:
    def __init__(self, model_path="./assets/models/google-bert/bert-base-chinese"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查微调后的模型是否存在
        fine_tuned_path = "./models/fine_tuned_bert_simple"
        import os
        
        if os.path.exists(fine_tuned_path) and os.path.exists(os.path.join(fine_tuned_path, "config.json")):
            print("发现微调后的模型，优先使用微调模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            self.bert_model = AutoModel.from_pretrained(fine_tuned_path)
            self.model_path = fine_tuned_path
        else:
            print("使用预训练模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.bert_model = AutoModel.from_pretrained(model_path)
            self.model_path = model_path
        
        # 创建分类器
        self.classifier = TextClassifier(self.bert_model)
        self.classifier.to(self.device)
        
        # 如果使用的是微调模型，加载完整的微调模型（包括BERT和分类器）
        if "fine_tuned_bert_simple" in self.model_path:
            print("使用微调后的完整模型")
        
        # 设置评估模式
        self.classifier.eval()
        
    def predict(self, text):
        """对输入文本进行分类预测"""
        # 编码输入文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.classifier(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }