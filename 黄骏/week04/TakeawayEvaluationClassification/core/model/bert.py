from typing import Union, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from config import CATEGORY_NAME, BERT_MODEL_PKL_PATH, BERT_MODEL_PERTRAINED_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)

# 直接加载保存的模型
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=2)
model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH, weights_only=True))
model.to(device)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

async def classify_for_bert(request_evaluation: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    使用BERT进行评价文本分类
    :param request_evaluation: 待分类的评价文本
    :return: 分类结果
    """
    classify_result: Union[str, List[str]] = None

    if isinstance(request_evaluation, str):
        request_evaluation = [request_evaluation]
    elif isinstance(request_evaluation, list):
        pass
    else:
        raise Exception("格式不支持")

    test_encoding = tokenizer(list(request_evaluation), truncation=True, padding=True, max_length=30)
    test_dataset = NewsDataset(test_encoding, [0] * len(request_evaluation))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())

    classify_result = [CATEGORY_NAME[x] for x in pred]

    return classify_result