import codecs
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings
from peft import LoraConfig, TaskType, get_peft_model

warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('./msra/train/sentences.txt').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('./msra/train/tags.txt').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = codecs.open('./msra/val/sentences.txt').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('./msra/val/tags.txt').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[label2id[x] for x in tag] for tag in val_tags]

# 转换为 HuggingFace Dataset
def prepare_dataset(texts, tags):
    tokens = [list(text) for text in texts]
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })
    return dataset

train_dataset = prepare_dataset(train_lines, train_tags)
eval_dataset = prepare_dataset(val_lines, val_tags)

# 初始化 tokenizer 和模型
model_path = "../models/Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=len(tag_type),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 数据处理函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        max_length=128,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize 数据集
train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
eval_tokenized = eval_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=eval_dataset.column_names)

# 训练参数
training_args = TrainingArguments(
    output_dir='./output_qwen_lora_ner',
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    report_to="none",
    gradient_checkpointing=True,
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model()
tokenizer.save_pretrained('./output_qwen_lora_ner')


# 加载微调后的模型
model = AutoModelForTokenClassification.from_pretrained('./output_qwen_lora_ner')
tokenizer = AutoTokenizer.from_pretrained('./output_qwen_lora_ner')

# todo 暂未研究清楚，还需要继续调试

sentence = "今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。"
tokens = list(sentence)
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
predicted_labels = [id2label[p.item()] for p in predictions[0]]
print(list(zip(tokens, predicted_labels)))