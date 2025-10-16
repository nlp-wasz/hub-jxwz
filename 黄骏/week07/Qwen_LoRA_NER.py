import codecs
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# 用于参数高效微调
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据加载和预处理
def load_and_preprocess_data(max_samples=1000):
    """加载和预处理NER数据"""
    # 加载训练数据
    train_lines = codecs.open('./msra/train/sentences.txt').readlines()[:max_samples]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt').readlines()[:max_samples]
    train_tags = [x.strip().split(' ') for x in train_tags]

    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt').readlines()[:max_samples // 10]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt').readlines()[:max_samples // 10]
    val_tags = [x.strip().split(' ') for x in val_tags]

    # 转换为生成式NER训练数据
    train_data = prepare_generative_ner_data(train_lines, train_tags)
    val_data = prepare_generative_ner_data(val_lines, val_tags)

    # 转换为DataFrame格式
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # 转换为Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


def prepare_generative_ner_data(texts, tag_lists):
    """将传统NER数据转换为生成式格式"""
    data = []
    for text, tags in zip(texts, tag_lists):
        # 提取实体及其类型
        entities = extract_entities_from_bio(list(text), tags)

        # 构建实体标注字符串，格式：实体1[类型1]|实体2[类型2]|...
        entity_annotations = ""
        for entity, entity_type in entities:
            entity_annotations += f"{entity}[{entity_type}]|"

        # 去掉最后一个分隔符
        if entity_annotations:
            entity_annotations = entity_annotations[:-1]

        # 如果没有实体，设置为"无实体"
        if not entity_annotations:
            entity_annotations = "无实体"

        # 添加到数据集
        data.append({
            "instruction": f"请识别以下文本中的实体（人名用PER标注，地点用LOC标注，组织机构用ORG标注）：{text}",
            "input": "",
            "output": entity_annotations
        })

    return data


def extract_entities_from_bio(tokens, bio_tags):
    """从BIO标注中提取实体"""
    entities = []
    current_entity = ""
    current_type = ""

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token
            current_type = tag[2:]  # 去掉"B-"前缀
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            current_entity += token
        elif tag == 'O':
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            if tag.startswith('B-'):
                current_entity = token
                current_type = tag[2:]
            else:
                current_entity = ""
                current_type = ""

    if current_entity:
        entities.append((current_entity, current_type))

    return entities


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


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n请识别文本中的实体（人名、地点、组织）<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
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
        # 对模型的特定层进行高效微调
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,  # LoRA的秩
        lora_alpha=32,  # LoRA的缩放参数
        lora_dropout=0.1  # LoRA的dropout率
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen_NER",
        per_device_train_batch_size=4,  # 减小批量以避免内存溢出
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=100,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_entities(model, tokenizer, text, device='cpu'):
    """预测文本中的实体"""
    messages = [
        {"role": "system",
         "content": "请识别文本中的实体（人名用PER标注，地点用LOC标注，组织机构用ORG标注）。每个实体只需标注一次，不要重复。输出格式为：实体1[类型1]|实体2[类型2]|...，不要添加引号或其他符号。"},
        {"role": "user", "content": f"请识别以下文本中的实体：{text}"}
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
            max_new_tokens=128,
            do_sample=True,
            temperature=0.05,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return parse_entity_output(response.strip())


def parse_entity_output(output):
    """解析模型输出的实体标注，并去重"""
    if output == "无实体":
        return []

    entities = []
    seen = set()  # 用于去重

    # 分割实体标注
    entity_annotations = output.split("|")

    for annotation in entity_annotations:
        try:
            # 匹配格式：实体[类型]
            if '[' in annotation and ']' in annotation:
                entity = annotation.split('[')[0].strip()
                # 去除引号
                entity = entity.replace('"', '').replace('"', '').replace('"', '')
                entity_type = annotation.split('[')[1].split(']')[0].strip()

                # 去重
                entity_key = f"{entity}_{entity_type}"
                if entity_key not in seen:
                    entities.append((entity, entity_type))
                    seen.add(entity_key)
        except:
            continue

    return entities


# 评估函数
def evaluate_predictions(true_entities, pred_entities):
    """评估预测结果"""
    true_flat = []
    pred_flat = []

    for true, pred in zip(true_entities, pred_entities):
        # 构建真实标签集
        true_set = {f"{entity}_{type}" for entity, type in true}
        # 构建预测标签集
        pred_set = {f"{entity}_{type}" for entity, type in pred}

        # 添加到平铺列表中进行评估
        for item in true_set:
            true_flat.append(item)
            pred_flat.append(item if item in pred_set else "O")

        for item in pred_set:
            if item not in true_set:
                true_flat.append("O")
                pred_flat.append(item)

    # 计算指标
    report = classification_report(true_flat, pred_flat, output_dict=True)
    return report


# 批量预测
def batch_predict(model, tokenizer, test_texts, test_tags=None, device='cuda'):
    """批量预测测试集的实体"""
    pred_entities = []
    true_entities = []

    for i, text in enumerate(tqdm(test_texts, desc="预测实体")):
        try:
            pred_entity = predict_entities(model, tokenizer, text, device)
            pred_entities.append(pred_entity)

            # 如果有真实标签，提取真实实体
            if test_tags is not None:
                true_entity = extract_entities_from_bio(list(text), test_tags[i])
                true_entities.append(true_entity)

        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_entities.append([])
            if test_tags is not None:
                true_entities.append([])

    # 如果有真实标签，评估结果
    if test_tags is not None:
        report = evaluate_predictions(true_entities, pred_entities)
        print("\n实体识别评估报告:")
        print(f"精确率: {report['weighted avg']['precision']:.4f}")
        print(f"召回率: {report['weighted avg']['recall']:.4f}")
        print(f"F1分数: {report['weighted avg']['f1-score']:.4f}")

    return pred_entities, true_entities if test_tags else None


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds, val_ds = load_and_preprocess_data(max_samples=1000)  # 可以根据需要调整样本数量

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../../../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(process_func_with_tokenizer, remove_columns=val_ds.column_names)

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 6. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 7. 保存模型
    print("保存模型...")
    trainer.save_model("./output_Qwen_NER_final")

    return trainer


# 测试函数
def test_model():
    """测试已训练的模型"""
    # 加载模型和tokenizer
    model_path = "../../../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载LoRA权重
    model.load_adapter("./output_Qwen_NER_final")
    model.to(device)

    # 测试样例
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
    ]

    # 批量预测
    pred_entities, _ = batch_predict(model, tokenizer, test_sentences, device=device)

    # 打印预测结果
    for i, sentence in enumerate(test_sentences):
        print(f"\n句子: {sentence}")
        if pred_entities[i]:
            for entity, entity_type in pred_entities[i]:
                print(f"  {entity_type}: {entity}")
        else:
            print("  未识别到实体")


if __name__ == "__main__":
    # 执行主函数（训练模型）
    trainer = main()

    # 测试模型
    test_model()