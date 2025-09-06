import torch
from transformers import BertTokenizer, BertForSequenceClassification

import joblib


class BertTextClassifier:
    def __init__(self, model_path, label_encoder_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # 编码文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测结果
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

        # 转换回原始标签
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "all_predictions": {
                label: round(predictions[0][i].item(), 4)
                for i, label in enumerate(self.label_encoder.classes_)
            }
        }


