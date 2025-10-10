REGEX_RULE = {
    "好评": ['很快', '辛苦', '小哥', '师傅', '包装', '超级', '骑士', '喜欢', '好评', '谢谢', '卷饼', '好喝', '煎饼', '满意', '送货', '准时', '挺好吃', '口味', '实惠', '服务态度', '肘子', '挺快', '很棒', '感谢', '棒棒'],
    "差评": ['难吃', '太慢', '两个', '打电话', '电话', '分钟', '米饭', '差评', '订单', '商家', '不好', '一份', '一个半', '订餐', '这次', '送达', '实在', '牛肉', '居然', '店里', '送过来', '一个多', '这家', '套餐', '备注']
}



CATEGORY_NAME = ['差评', '好评']

TFIDF_MODEL_PKL_PATH = "assets/weights/tfidf_ml.pkl"

BERT_MODEL_PKL_PATH = "assets/weights/bert.pt"
BERT_MODEL_PERTRAINED_PATH = "assets/models/google-bert/bert-base-chinese"

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1" # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "qwen2.5:0.5b"
