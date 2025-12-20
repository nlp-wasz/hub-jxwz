#coding=utf-8
import pandas as pd
from datasets import Dataset
import codecs
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
import json

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

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('../msra/train/sentences.txt').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('../msra/train/tags.txt').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
# train_tags = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = codecs.open('../msra/val/sentences.txt').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('../msra/val/tags.txt').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
# val_tags = [[label2id[x] for x in tag] for tag in val_tags]


def prepare_dataset(texts:list,tags:list):
    train_data = []
    for text,tag in zip(texts,tags):
        data = {}
        data["instruction"] = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点;人名;组织.以 json 格式输出,如{'组织':['name1','name2'],'人物':['person1','person2'],'地点': ['loc1','loc22']}."""

        data["input"] = text
        start = 0
        last = 0
        last_ner = '0'
        ORG = []
        PER = []
        LOC = []
        entity = ""
        for i,ner in enumerate(tag):
            if ner == 'O':
                if last_ner != '0':
                    if last_ner == 'B-ORG':
                        ORG.append(entity)
                    elif last_ner == 'B-PER':
                        PER.append(entity)
                    elif last_ner == 'B-LOC':
                        LOC.append(entity)
                    else:
                        print("do not support this type: ", last_ner)
                    entity = ""
            elif ner == 'B-ORG':
                entity = entity + text[i]
                last_ner = ner
            elif ner == 'I-ORG':
                entity = entity + text[i]
            elif ner == 'B-PER':
                entity = entity + text[i]
                last_ner = ner
            elif ner == 'I-PER':
                entity = entity + text[i]
            elif ner == 'B-LOC':
                entity = entity + text[i]
                last_ner = ner
            elif ner == 'I-LOC':
                entity = entity + text[i]
            else:
                print("do not support this type: ", ner)

        data["output"] = json.dumps({
            "组织": [x for x in set(ORG) if x != ''],
            "人物": [x for x in set(PER) if x != ''],
            "地点": [x for x in set(LOC) if x != '']
        },ensure_ascii=False)
        # print(data["instruction"])
        train_data.append(data)
    pdata = pd.DataFrame(train_data)
    ds = Dataset.from_pandas(pdata)
    return ds



def process_func(example, tokenizer, max_length=1000):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    # print(instruction_text)
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

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

    # 对什么模型，以什么方式进行微调
    config = LoraConfig(
        # 任务类型，自回归语言建模
        task_type=TaskType.CAUSAL_LM,

        # 对什么层的默写模块进行高效微调
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
        {"role": "system", "content": "你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点;人名;组织.以 json 格式输出,如{'组织':['name1','name2'],'人物':['person1','person2'],'地点': ['loc1','loc22']}."},
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

    return response.strip()

def train():
    print("加载数据...")
    train_dataset = prepare_dataset(train_lines, train_tags)
    val_dataset = prepare_dataset(val_lines, val_tags)

    print("加载模型...")
    tokenizer, model = initialize_model_and_tokenizer('../../Qwen/Qwen3-0.6B')

    print("准备数据...")
    train_tokenized = train_dataset.map(lambda example: process_func(example, tokenizer),remove_columns=train_dataset.column_names)
    eval_tokenized = val_dataset.map(lambda example: process_func(example, tokenizer),remove_columns=val_dataset.column_names)


    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

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

def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../../Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen1.5/checkpoint-205/")
    model.cpu()

    # 测试预测
    test_text = "中国银行的王强问从山东去北京如何走？"
    result = predict_intent(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"预测实体: {result}")


if __name__ == '__main__':
    test_single_example()
