import pandas as pd
import codecs
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


def convert_ner_to_instruction_format(texts, tags, split_name="train"):
    """将NER格式数据转换为指令格式"""
    converted_data = []

    for i, (text, tag_list) in enumerate(zip(texts, tags)):
        # 将标签ID转换为标签名称
        tag_names = [id2label[tag_id] for tag_id in tag_list]

        # 创建指令数据
        instruction = f"请对以下文本进行命名实体识别，找出其中的人名(PER)、组织机构(ORG)和地理位置(LOC)实体：{text}"

        # 创建输出格式：实体类型: 实体内容
        entities = []
        current_entity = ""
        current_type = ""

        for char, tag in zip(text, tag_names):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(f"{current_type}: {current_entity}")
                current_entity = char
                current_type = tag[2:]
            elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
                current_entity += char
            else:
                if current_entity:
                    entities.append(f"{current_type}: {current_entity}")
                current_entity = ""
                current_type = ""
                if tag.startswith('B-'):
                    current_entity = char
                    current_type = tag[2:]

        if current_entity:
            entities.append(f"{current_type}: {current_entity}")

        # 如果没有找到实体，添加特殊标记
        if not entities:
            entities.append("未识别到实体")

        output = "；".join(entities)

        converted_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    return converted_data


def load_and_preprocess_data(data_dir='./msra', force_convert=False):
    """加载和预处理数据"""
    converted_data_path = os.path.join(data_dir, 'converted_ner_data.json')

    # 如果已存在转换后的数据且不强制转换，则直接加载
    if os.path.exists(converted_data_path) and not force_convert:
        print("加载已转换的数据...")
        with open(converted_data_path, 'r', encoding='utf-8') as f:
            converted_data = json.load(f)

        # 转换为DataFrame
        train_df = pd.DataFrame(converted_data['train'])
        val_df = pd.DataFrame(converted_data['val'])

        return train_df, val_df

    print("转换数据格式...")
    # 加载训练数据
    train_lines = codecs.open(os.path.join(data_dir, 'train/sentences.txt'), encoding='utf-8').readlines()[:1000]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open(os.path.join(data_dir, 'train/tags.txt')).readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]
    train_tags = [[label2id[x] for x in tag] for tag in train_tags]

    # 加载验证数据
    val_lines = codecs.open(os.path.join(data_dir, 'val/sentences.txt'), encoding='utf-8').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open(os.path.join(data_dir, 'val/tags.txt'), encoding='utf-8').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]
    val_tags = [[label2id[x] for x in tag] for tag in val_tags]

    # 转换数据格式
    train_converted = convert_ner_to_instruction_format(train_lines, train_tags, "train")
    val_converted = convert_ner_to_instruction_format(val_lines, val_tags, "val")

    # 保存转换后的数据
    converted_data = {
        'train': train_converted,
        'val': val_converted
    }

    with open(converted_data_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    # 转换为DataFrame
    train_df = pd.DataFrame(train_converted)
    val_df = pd.DataFrame(val_converted)

    return train_df, val_df


def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 - 修复torch_dtype警告
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,  # 使用dtype替代torch_dtype
        trust_remote_code=True
    )

    return tokenizer, model


def process_func(example, tokenizer, max_length=512):
    """处理单个样本的函数"""
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n你是一个命名实体识别专家，负责从文本中识别出人名(PER)、组织机构(ORG)和地理位置(LOC)实体。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

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


def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./output_ner_model"):
    """训练模型"""
    # 设置训练参数 - 禁用梯度检查点
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        eval_steps=50,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False,  # 禁用梯度检查点
        report_to="none",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # 显式启用FP16训练
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer


def predict_ner(model, tokenizer, text, device='cuda'):
    """预测单个文本的命名实体"""
    messages = [
        {"role": "system",
         "content": "你是一个命名实体识别专家，负责从文本中识别出人名(PER)、组织机构(ORG)和地理位置(LOC)实体。"},
        {"role": "user",
         "content": f"请对以下文本进行命名实体识别，找出其中的人名(PER)、组织机构(ORG)和地理位置(LOC)实体：{text}"}
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
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


def evaluate_model(model, tokenizer, test_texts, test_tags, device='cuda'):
    """评估模型性能"""
    pred_labels = []
    true_labels = []

    for text, tags in tqdm(zip(test_texts, test_tags), desc="评估模型", total=len(test_texts)):
        try:
            # 预测
            pred_output = predict_ner(model, tokenizer, text, device)

            # 解析预测结果
            pred_entities = parse_ner_output(pred_output)

            # 将预测结果转换为BIO标签格式
            pred_bio = convert_entities_to_bio(text, pred_entities)

            # 确保长度一致
            min_len = min(len(pred_bio), len(tags))
            pred_bio = pred_bio[:min_len]
            tags = tags[:min_len]
            # 将数字标签转换为字符串标签
            tags_str = [id2label[tag] for tag in tags]
            pred_labels.extend(pred_bio)
            true_labels.extend(tags_str)

        except Exception as e:
            print(f"处理文本 '{text}' 时出错: {e}")

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)

    return accuracy, report


def parse_ner_output(output):
    """解析模型输出的NER结果"""
    entities = []

    # 分割不同的实体
    entity_parts = output.split('；')

    for part in entity_parts:
        if ':' in part:
            entity_type, entity_text = part.split(':', 1)
            entity_type = entity_type.strip()
            entity_text = entity_text.strip()

            entities.append((entity_type, entity_text))

    return entities


def convert_entities_to_bio(text, entities):
    """将实体列表转换为BIO标签序列"""
    bio_labels = ['O'] * len(text)

    for entity_type, entity_text in entities:
        # 在文本中查找实体
        start_idx = text.find(entity_text)

        if start_idx != -1:
            # 标记B-标签
            bio_labels[start_idx] = f'B-{entity_type}'

            # 标记I-标签
            for i in range(start_idx + 1, start_idx + len(entity_text)):
                if i < len(bio_labels):
                    bio_labels[i] = f'I-{entity_type}'

    return bio_labels


def main(mode="train", model_path=None):
    """主函数"""
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_df, val_df = load_and_preprocess_data()

    if mode == "train":
        # 初始化模型和tokenizer
        model_path = model_path or "../asset/models/Qwen/Qwen3-0.6B/"
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 设置LoRA
        model = setup_lora(model)
        model.to(device)

        # 确保模型处于训练模式
        model.train()

        # 处理数据
        process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

        # 创建数据集
        train_ds = Dataset.from_pandas(train_df)
        eval_ds = Dataset.from_pandas(val_df)

        train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
        eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

        # 训练模型
        trainer = train_model(model, tokenizer, train_tokenized, eval_tokenized)

        # 评估模型
        print("评估模型...")

        # 加载原始验证数据用于评估
        val_lines = codecs.open('./msra/val/sentences.txt', encoding='utf-8').readlines()[:100]
        val_lines = [x.replace(' ', '').strip() for x in val_lines]

        val_tags = codecs.open('./msra/val/tags.txt', encoding='utf-8').readlines()[:100]
        val_tags = [x.strip().split(' ') for x in val_tags]
        val_tags = [[label2id[x] for x in tag] for tag in val_tags]

        accuracy, report = evaluate_model(model, tokenizer, val_lines, val_tags, device)
        print(f"准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report([id2label[x] for x in val_tags[0]], [id2label[x] for x in val_tags[0]]))

    elif mode == "inference":
        if not model_path:
            raise ValueError("推理模式需要指定模型路径")

        # 加载微调后的模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.to(device)

        # 设置为评估模式
        model.eval()

        # 测试句子
        test_sentences = [
            '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
            '人工智能是未来的希望，也是中国和美国的冲突点。',
            '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
            '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
            '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
            '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
        ]

        # 进行预测
        for sentence in test_sentences:
            try:
                result = predict_ner(model, tokenizer, sentence, device)
                print(f"句子: {sentence}")
                print(f"识别结果: {result}")
                print()
            except Exception as e:
                print(f"处理句子时出错: {sentence}")
                print(f"错误信息: {e}")
                print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NER using LoRA fine-tuning')
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'],
                        help='运行模式: train (训练) 或 inference (推理)')
    parser.add_argument('--model_path', type=str, default="./output_ner_model",
                        help='模型路径 (对于推理模式是必需的)')

    args = parser.parse_args()

    main(mode=args.mode, model_path=args.model_path)