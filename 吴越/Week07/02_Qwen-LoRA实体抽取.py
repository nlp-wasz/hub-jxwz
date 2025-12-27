import pandas as pd
import codecs
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch

# # 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# # 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('./msra/train/sentences.txt',encoding='utf-8').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('./msra/train/tags.txt',encoding='utf-8').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = codecs.open('./msra/val/sentences.txt',encoding='utf-8').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('./msra/val/tags.txt',encoding='utf-8').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[label2id[x] for x in tag] for tag in val_tags]



# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model

def prepare_dataset(texts, tags):
    # 将文本拆分为字符列表
    tokens = [list(text) for text in texts]

    # 创建数据集
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })


    return dataset

# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    
    text = example['text']
    tags = example['tags']

    # 将标签转换为文本形式
    tag_texts = [id2label[tag] for tag in tags]
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在进行实体识别任务，请输入文本中每个字符的实体类型。实体类型包括：O（非实体）、B-ORG（组织机构开始）、I-ORG（组织机构内部）、B-PER（人名开始）、I-PER（人名内部）、B-LOC（地名开始）、I-LOC（地名内部）<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{tag_texts}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断超过最大长度的序列
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen1.5",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "现在进行实体识别任务，请为输入文本中的每个字符标注实体类型。实体类型包括：O（非实体）、B-ORG（组织机构开始）、I-ORG（组织机构内部）、B-PER（人名开始）、I-PER（人名内部）、B-LOC（地名开始）、I-LOC（地名内部）。"},
        {"role": "user", "content": text}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 处理响应，提取标签
    
    predicted_tags = response.strip().split()

    # 确保标签数量与文本字符数量一致
    if len(predicted_tags) > len(text):
         predicted_tags = predicted_tags[:len(text)]
    elif len(predicted_tags) < len(text):
        predicted_tags.extend(['O'] * (len(text) - len(predicted_tags)))

    # 提取实体
    entities = []
    current_entity = ""
    current_type = ""

    for char, tag in zip(text, predicted_tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = char
                current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
                current_entity += char
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""
            if tag.startswith('B-'):
                current_entity = char
                current_type = tag[2:]

    if current_entity:
        entities.append((current_entity, current_type))

    return entities, predicted_tags


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="预测意图"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_dataset = prepare_dataset(train_lines, train_tags)
    eval_dataset = prepare_dataset(val_lines, val_tags)

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    train_tokenized = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(process_func_with_tokenizer, remove_columns=eval_dataset.column_names)

    # 5. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 6. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 7. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_Qwen")

# 测试预测
test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]


# 如果MPS预测失败，使用CPU预测
for sentence in test_sentences:
    try:
        entities = predict_intent(sentence)
        print(f"句子: {sentence}")
        if entities:
            for entity, entity_type in entities:
                print(f"  {entity_type}: {entity}")
        else:
            print("  未识别到实体")
        print()
    except Exception as e:
        print(f"处理句子时出错: {sentence}")
        print(f"错误信息: {e}")
        print()