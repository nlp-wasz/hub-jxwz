# 通过 BERT 与训练模型，实现命名实体识别（NER）
# 步骤划分：1.数据读取  2.构建 Dataset，分词预处理  3.构建模型参数，训练模型  4.模型预测
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq
from tqdm import tqdm

# 1.读取数据
# 定义 实体分类标签
entity_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
labels_to_idx = {label: idx for idx, label in enumerate(entity_labels)}
idx_to_labels = {idx: label for idx, label in enumerate(entity_labels)}

# 读取 训练集文本和对应的实体标签
train_texts = open("../data/msra/train/sentences.txt", "r", encoding="utf-8").readlines()
train_labels = open("../data/msra/train/tags.txt", "r", encoding="utf-8").readlines()

# 读取 测试集文本和对应的实体标签
eval_texts = open("../data/msra/val/sentences.txt", "r", encoding="utf-8").readlines()
eval_labels = open("../data/msra/val/tags.txt", "r", encoding="utf-8").readlines()

# 训练集和测试集 读取到的文本和标签 存在许多空格和其它字符，需要处理掉
train_texts = [list(train_text.replace(" ", "").replace("\n", "")) for train_text in train_texts]
train_labels = [train_label.replace("\n", "").split(" ") for train_label in train_labels]

eval_texts = [list(eval_text.replace(" ", "").replace("\n", "")) for eval_text in eval_texts]
eval_labels = [eval_label.replace("\n", "").split(" ") for eval_label in eval_labels]

# 2.构建Dataset数据集 以及分词处理逻辑（token分词后 会生成特殊token，原始label无法对应，需要对齐）
train_dataset = Dataset.from_dict({
    "texts": train_texts[:1000],
    "labels": train_labels[:1000],
})
eval_dataset = Dataset.from_dict({
    "texts": eval_texts[:1000],
    "labels": eval_labels[:1000],
})

# 定义 分词处理逻辑
tokenizer = BertTokenizerFast.from_pretrained("../../models/google-bert/bert-base-chinese")


def get_tokenizer_result(dataset):
    # 从 datset 中获取 texts 和 labels
    texts = dataset["texts"]
    labels = dataset["labels"]

    # 对 texts 进行分词处理
    texts_token = tokenizer(texts, padding=False, truncation=True, max_length=512, is_split_into_words=True)

    # texts_token -> input_ids  attention_mask  token_type_ids
    texts_labels = []
    for i in range(len(texts)):
        # word_ids() 获取每个样本中 token对应的原始样本中的位置索引
        # 如一些分词器加入的特殊token，不存在与原始token中，在word_ids中值为None，还有一些共同指向同一个索引
        # 这些都需要特殊处理，将其对应的标签修改为-100，bert模型会在计算损失时自动屏蔽这些未知的token（自动忽略）
        word_ids = texts_token.word_ids(i)
        label = []
        curr_word_id = None
        for word_id in word_ids:
            # 如果 为None，代表是分词器加入的特殊字符（如CLS，SEP等）
            if word_id is None:
                label.append(-100)
            elif word_id != curr_word_id:
                label.append(labels_to_idx[labels[i][word_id]])
            else:
                label.append(-100)

            curr_word_id = word_id

        # 保存 对齐后的 labels
        texts_labels.append(label)

    texts_token["labels"] = texts_labels

    return texts_token


# 对 训练集和测试集 dataset 进行分词预处理（自定义）
train_dataset = train_dataset.map(get_tokenizer_result, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(get_tokenizer_result, batched=True, remove_columns=eval_dataset.column_names)

# 3.构建模型参数，训练模型
bert_ner_model = BertForTokenClassification.from_pretrained(
    "../../models/google-bert/bert-base-chinese",
    num_labels=len(entity_labels),
    id2label=idx_to_labels,
    label2id=labels_to_idx)

training_arguments = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    output_dir="./data/ner-train",
    logging_dir="./data/logs",
    logging_steps=50,

    save_strategy="epoch",
    eval_strategy="epoch",
    do_eval=True,
    report_to="none",
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=bert_ner_model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8  # GPU填充对齐
    )
)

print("模型训练开始")
trainer.train()
print("模型训练完毕")

print("模型评估开始")
eval_results = trainer.evaluate()
print(eval_results)
print("模型评估完毕")

# 4.预测
test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]

for sentence in tqdm(test_sentences, desc="NER 预测"):
    sentence = list(sentence)

    # 对 sentence 分词处理
    sentence_token = tokenizer(sentence, padding=False, truncation=True, max_length=512, is_split_into_words=True,
                               return_tensors="pt").to("cuda")

    # 通过模型 预测
    with torch.no_grad():
        sentence_logits = bert_ner_model(**sentence_token).logits

        # sentence_res 是对所有token 预测的标签idx -> (1, token_size)
        sentence_res = sentence_logits.argmax(dim=-1).view(-1)

    # 获取 预测token对应的标签（需要把 特殊token 去除掉）
    # 获取 分词后的样本token 对应原始样本索引
    word_ids = sentence_token.word_ids(0)
    pre_label2idx = []
    curr_word_id = None

    # 只有当 word_id 不为None（不是特殊token），并且多个相同的word_id 只取一个
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id != curr_word_id:
            pre_label2idx.append(sentence_res[i].item())

    # 判断 pre_label 和 sentence 长度是否一致
    if len(pre_label2idx) > len(sentence):
        pre_label2idx = pre_label2idx[:len(sentence)]
    elif len(pre_label2idx) < len(sentence):
        pre_label2idx += [0] * (len(sentence) - len(pre_label2idx))

    # 将 pre_label 转换为文字标签
    pre_idx2label = [idx_to_labels[label] for label in pre_label2idx]

    # 命名实体识别（自定义逻辑）
    sentence_ner = []
    curr_node = None
    curr_type = None
    for index, label in enumerate(pre_idx2label):
        if label.startswith("B-"):
            if curr_node:
                sentence_ner.append((curr_node, curr_type))
            curr_node = sentence[index]
            curr_type = label[2:]
        elif label.startswith("I-") and curr_node and curr_type == label[2:]:
            curr_node += sentence[index]
        else:
            if curr_node:
                sentence_ner.append((curr_node, curr_type))
            curr_node = None
            curr_type = None

    if curr_node:
        sentence_ner.append((curr_node, curr_type))

    # 打印 识别到的实体
    if sentence_ner:
        print(f"{sentence}")
        for entity, label in sentence_ner:
            print(f"{label}: {entity} \n")
    else:
        print(f"{sentence} \n 未识别到实体！")
