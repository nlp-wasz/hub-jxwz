REGEX_RULE = {
    "FilmTele-Play": ["播放", "电视剧"], # 句子是不是包含特定的单词，做出分类
    "HomeAppliance-Control": ["空调", "广播"]
}




CATEGORY_NAME = [
    '差评', '好评'
]


BERT_MODEL_PKL_PATH = "weights/bert.pt"
BERT_MODEL_PERTRAINED_PATH = "../../../models/google-bert/bert-base-chinese"

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1" # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "qwen2.5:0.5b"
