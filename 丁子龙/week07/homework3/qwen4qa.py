import json
import torch
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import numpy as np
from typing import TypedDict, List
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



def prepare_dataset(jsonData:dict):
    dataset = []
    for paragraph in jsonData['data']:
        context = paragraph['paragraphs'][0]['context']
        qas = paragraph['paragraphs'][0]['qas']
        for qa in qas:
            question = "问题：" + qa['question']
            answers = qa['answers']
            question_answer = ""
            for i, answer in enumerate(answers):
                question_answer = question_answer + "回答" + str(i+1) + ":"  + answer['text'] + '\n'

            dataset.append({
                'instruction': """你是一个知识问答模型，可以从下面文本中回答用户的提问，如果文本中没有答案则返回 【没有答案】""" ,
                'input': context + question,
                'output': question_answer
            })
    pdata = pd.DataFrame(dataset)
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
        output_dir="./outputqa_Qwen1.5",
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
        {"role": "system", "content": "你是一个知识问答模型，可以从下面文本中回答用户的提问，如果文本中没有答案则返回 【没有答案】"},
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





def train():
    print("加载数据...")
    # 加载数据
    train = json.load(open('../cmrc2018_public/train.json'))
    dev = json.load(open('../cmrc2018_public/dev.json'))

    train_dataset = prepare_dataset(train)
    val_dataset = prepare_dataset(dev)

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
    model.load_adapter("./outputqa_Qwen1.5/checkpoint-2110/")
    model.cpu()

    # 测试预测
    dev = json.load(open('../cmrc2018_public/trial.json'))
    val_dataset = prepare_dataset(dev)
    for example in val_dataset:
        test_text = example["input"]
        question = test_text.split("问题：")[1]
        answer = example["output"]
        result = predict_intent(model, tokenizer, test_text)
        print(f"输入: {test_text}")
        print(f"问题: {question}")
        print(f"预测结果: {result}")
        print(f"真实结果: {answer}")


if __name__ == '__main__':
    # train()
    test_single_example()