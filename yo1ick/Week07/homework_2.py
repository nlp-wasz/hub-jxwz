import torch
from transformers import (
    Qwen2Tokenizer,
    Qwen2ForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 标签定义
tag_types = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_types)}
label2id = {label: i for i, label in enumerate(tag_types)}

# 加载MSRA数据
def load_msra_data(sentences_file, tags_file, max_samples=1000):
    # 读取句子
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip().replace(' ', '') for line in f.readlines()[:max_samples]]
    
    # 读取标签
    with open(tags_file, 'r', encoding='utf-8') as f:
        tags_lines = [line.strip().split() for line in f.readlines()[:max_samples]]
    
    # 确保标签数量与句子字符数一致
    processed_tags = []
    for sentence, tags in zip(sentences, tags_lines):
        if len(tags) != len(sentence):
            # 如果标签数量不一致，使用'O'填充
            tags = tags[:len(sentence)] + ['O'] * (len(sentence) - len(tags))
        processed_tags.append([label2id[tag] for tag in tags])
    
    return sentences, processed_tags

# 加载数据
train_sentences, train_labels = load_msra_data('./msra/train/sentences.txt', './msra/train/tags.txt', 500)
val_sentences, val_labels = load_msra_data('./msra/val/sentences.txt', './msra/val/tags.txt', 100)

# 初始化tokenizer和模型
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
# 添加pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Qwen2ForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tag_types),
    id2label=id2label,
    label2id=label2id
)

# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 简化的数据预处理 - 不使用offset_mapping
def tokenize_function(examples):
    # 将字符列表转换为字符串
    texts = [''.join(tokens) for tokens in examples["tokens"]]
    
    # Tokenize
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=256,
    )
    
    # 简单的标签对齐 - 假设第一个token对应第一个字符，以此类推
    # 这是一个简化的方法，实际应用中需要更精确的对齐
    labels = []
    for i, label_ids in enumerate(examples["labels"]):
        input_ids = tokenized_inputs["input_ids"][i]
        label_ids_expanded = []
        
        # 为每个token分配标签
        char_idx = 0
        for j, token_id in enumerate(input_ids):
            token = tokenizer.decode([token_id])
            # 特殊token
            if token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                label_ids_expanded.append(-100)
            # 正常token
            elif char_idx < len(label_ids):
                label_ids_expanded.append(label_ids[char_idx])
                # 如果token包含多个字符，前进相应的位置
                if len(token.strip()) > 1 and not token.startswith('#'):
                    char_idx += len(token.strip())
                else:
                    char_idx += 1
            else:
                label_ids_expanded.append(-100)
        
        labels.append(label_ids_expanded)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 准备数据集
train_dataset = Dataset.from_dict({
    "tokens": [list(sent) for sent in train_sentences],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "tokens": [list(sent) for sent in val_sentences],
    "labels": val_labels
})

# 应用tokenize
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=8,
    remove_columns=train_dataset.column_names,
)

tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=8,
    remove_columns=val_dataset.column_names,
)

# 训练参数 - 使用正确的参数名
training_args = TrainingArguments(
    output_dir="./qwen-ner-lora",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,  # 减少epochs以加快训练
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",  # 修正参数名
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
)

# 数据收集器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

# 评估函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 移除忽略的索引（-100）
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions.append(id2label[p])
                true_labels.append(id2label[l])
    
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {"accuracy": accuracy}

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练
print("开始训练Qwen-LoRA NER模型...")
trainer.train()

# 保存模型
trainer.save_model()
tokenizer.save_pretrained("./qwen-ner-lora")

print("训练完成！")

# 简化的预测函数
def predict_ner(text):
    model.eval()
    
    # Tokenize输入
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_ids = predictions[0].cpu().numpy()
    
    # 将预测的ID转换为标签
    predicted_labels = []
    for pred_id in predicted_ids:
        if pred_id != -100:
            predicted_labels.append(id2label.get(pred_id, 'O'))
    
    # 简单的实体提取
    entities = []
    current_entity = ""
    current_type = ""
    
    # 由于标签和字符对齐复杂，这里简单显示预测结果
    print(f"文本: {text}")
    print(f"预测标签: {predicted_labels[:len(text)]}")
    
    # 尝试简单的实体提取（可能需要调整）
    for i, char in enumerate(text):
        if i < len(predicted_labels):
            label = predicted_labels[i]
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = char
                current_type = label[2:]
            elif label.startswith('I-') and current_entity and current_type == label[2:]:
                current_entity += char
            else:
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""
                if label.startswith('B-'):
                    current_entity = char
                    current_type = label[2:]
    
    if current_entity:
        entities.append((current_entity, current_type))
    
    return entities

# 测试预测
test_texts = [
    "今天我约了王浩在恭王府吃饭",
    "人工智能是未来的希望",
    "明天我们一起在海淀吃个饭吧"
]

print("\n测试预测结果:")
for text in test_texts:
    entities = predict_ner(text)
    print(f"识别到的实体: {entities}")
    print()
