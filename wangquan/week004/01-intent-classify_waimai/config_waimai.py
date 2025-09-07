REGEX_RULE = {
    "0": ["好", "不错", "快"], # 句子是不是包含特定的单词，做出分类
    "1": ["差", "难吃", "慢"]
}




CATEGORY_NAME = [
    '0', '1'
]

TFIDF_MODEL_PKL_PATH = "assets/weights/tfidf_ml_waimai.pkl"

BERT_MODEL_PKL_PATH = "assets/weights/bert_waimai.pt"
BERT_MODEL_PERTRAINED_PATH = "assets/models/bert-base-chinese/"

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1" # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "qwen2.5:0.5b"
