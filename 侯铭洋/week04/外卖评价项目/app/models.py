import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于缓存加载的模型和分词器
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id2label = {0: "负面", 1: "正面"}  # 假设你的模型是这样映射的


def load_model(model_path: str):
    """
    加载微调后的BERT模型和分词器

    Args:
        model_path: 模型保存的路径
    """
    global model, tokenizer

    try:
        logger.info(f"正在从 {model_path} 加载模型...")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        # 将模型移动到合适的设备 (GPU/CPU)
        model.to(device)
        model.eval()  # 设置为评估模式

        logger.info("模型加载成功!")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e


def predict_sentiment(text: str, max_length: int = 64):
    """
    对输入的文本进行情感分析预测

    Args:
        text: 要分析的文本
        max_length: 最大序列长度

    Returns:
        包含预测结果的字典
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise ValueError("模型未加载，请先调用 load_model()")

    try:
        # 对输入文本进行编码
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        # 将输入数据移动到与模型相同的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 模型预测 (禁用梯度计算以提升性能)
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取预测结果
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = np.argmax(probabilities).item()
        confidence = probabilities[predicted_class_id]

        # 构建结果字典
        result = {
            "sentiment": predicted_class_id,
            "confidence": float(confidence),
            "sentiment_label": id2label.get(predicted_class_id, "未知"),
            "probabilities": {
                id2label.get(i, f"类别{i}"): float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

        return result

    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}")
        raise e