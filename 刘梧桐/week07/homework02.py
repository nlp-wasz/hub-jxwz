import pandas as pd
import codecs
import numpy as np
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
import re

# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型（与BERT实体识别一致）
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


class QwenNERProcessor:
    """实体识别数据处理器"""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_msra_data(self, sentences_path, tags_path, max_samples=None):
        """加载MSRA数据集"""
        # 加载训练数据
        train_lines = codecs.open(sentences_path, 'r', encoding='utf-8').readlines()
        if max_samples:
            train_lines = train_lines[:max_samples]
        train_lines = [x.replace(' ', '').strip() for x in train_lines]

        train_tags = codecs.open(tags_path).readlines()
        if max_samples:
            train_tags = train_tags[:max_samples]
        train_tags = [x.strip().split(' ') for x in train_tags]
        train_tags = [[label2id[x] for x in tag] for tag in train_tags]

        return train_lines, train_tags

    def convert_to_instruction_format(self, text, tags):
        """将实体识别数据转换为指令格式"""
        # 将标签ID转换为标签名称
        tag_labels = [id2label[tag_id] for tag_id in tags]

        # 提取实体
        entities = self.extract_entities(text, tag_labels)

        # 构建指令
        instruction = f"请从以下文本中识别出所有的实体，包括人名(PER)、地名(LOC)和组织名(ORG)。文本：{text}"

        # 构建输出
        if entities:
            output_parts = []
            for entity, entity_type in entities:
                output_parts.append(f"{entity}({entity_type})")
            output = "识别到的实体有：" + "，".join(output_parts)
        else:
            output = "未识别到实体"

        return instruction, output

    def extract_entities(self, text, tags):
        """从标签序列中提取实体"""
        entities = []
        current_entity = ""
        current_type = ""

        for char, tag in zip(text, tags):
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

        return entities

    def prepare_dataset(self, sentences_path, tags_path, max_samples=1000):
        """准备训练数据集"""
        texts, tags = self.load_msra_data(sentences_path, tags_path, max_samples)

        instructions = []
        outputs = []

        for text, tag in zip(texts, tags):
            instruction, output = self.convert_to_instruction_format(text, tag)
            instructions.append(instruction)
            outputs.append(output)

        # 创建数据集
        dataset_dict = {
            "instruction": instructions,
            "output": outputs,
            "input": [""] * len(instructions)  # 空输入
        }

        return Dataset.from_dict(dataset_dict)


def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分（使用ChatML格式）
    instruction_text = f"<|im_start|>system\n你是一个实体识别专家，需要从文本中识别出人名、地名和组织名。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
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
    """设置LoRA配置"""
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


def predict_entities(model, tokenizer, text, device='cuda'):
    """预测文本中的实体"""
    messages = [
        {"role": "system", "content": "你是一个实体识别专家，需要从文本中识别出人名、地名和组织名。"},
        {"role": "user", "content": f"请从以下文本中识别出所有的实体，包括人名(PER)、地名(LOC)和组织名(ORG)。文本：{text}"}
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
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 解析响应中的实体
    entities = parse_entities_from_response(response)

    return entities, response


def parse_entities_from_response(response):
    """从模型响应中解析实体"""
    entities = []

    # 匹配模式：实体(类型)
    pattern = r'([^，。！？；]+)\(([^)]+)\)'
    matches = re.findall(pattern, response)

    for match in matches:
        entity, entity_type = match
        # 标准化实体类型
        if '人' in entity_type or entity_type.upper() == 'PER':
            entity_type = 'PER'
        elif '地' in entity_type or '位置' in entity_type or entity_type.upper() == 'LOC':
            entity_type = 'LOC'
        elif '组织' in entity_type or entity_type.upper() == 'ORG':
            entity_type = 'ORG'

        entities.append((entity.strip(), entity_type))

    return entities


def main():
    """主函数"""
    # 1. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"  # 根据实际情况修改路径

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 使用更安全的方式加载模型，避免meta device问题
    print("加载模型...")
    if torch.cuda.is_available():
        # 如果有GPU，直接加载到GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        # 如果只有CPU，使用低精度加载
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=torch.float32,
            trust_remote_code=True
        )

    # 2. 准备数据
    print("准备数据...")
    processor = QwenNERProcessor(tokenizer)

    # 加载训练数据
    train_dataset = processor.prepare_dataset(
        './msra/train/sentences.txt',
        './msra/train/tags.txt',
        max_samples=500  # 减少样本数量以便快速训练
    )

    # 加载验证数据
    eval_dataset = processor.prepare_dataset(
        './msra/val/sentences.txt',
        './msra/val/tags.txt',
        max_samples=100
    )

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    tokenized_train = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(process_func_with_tokenizer, remove_columns=eval_dataset.column_names)

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./qwen-ner-model",
        per_device_train_batch_size=2,  # 减小batch size
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        logging_steps=50,
        eval_steps=50,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=False,  # 暂时禁用，避免复杂问题
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=3,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,  # 避免列移除问题
    )

    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    # 7. 开始训练
    print("开始训练...")
    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-ner-model")

    return model, tokenizer


def test_model(model, tokenizer):
    """测试训练好的模型"""
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
    ]

    print("开始测试模型...")
    for sentence in test_sentences:
        try:
            entities, response = predict_entities(model, tokenizer, sentence, device)
            print(f"句子: {sentence}")
            print(f"模型响应: {response}")
            if entities:
                for entity, entity_type in entities:
                    print(f"  识别到实体: {entity} ({entity_type})")
            else:
                print("  未识别到实体")
            print("-" * 50)
        except Exception as e:
            print(f"处理句子时出错: {sentence}")
            print(f"错误信息: {e}")
            print()


if __name__ == "__main__":
    try:
        # 训练模型
        model, tokenizer = main()

        # 测试模型
        test_model(model, tokenizer)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()
