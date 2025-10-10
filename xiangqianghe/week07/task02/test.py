import json
import pandas as pd
import torch
import codecs
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import os
import gc
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score
import evaluate

# 设置UTF-8编码环境
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 首先检查GPU状态
print("=== GPU设备检查 ===")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"内存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")
else:
    print("未检测到GPU，将使用CPU运行")

# 下载模型（确保路径正确）
model_dir = snapshot_download(
    'Qwen/Qwen3-0.6B',
    local_dir='./models/Qwen/Qwen3-0.6B',
    revision='master'
)

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


def load_ner_data():
    """加载NER数据集 - 修复编码问题并添加数据验证"""
    try:
        # 加载训练数据 - 使用utf-8编码
        with codecs.open('./msra/train/sentences.txt', 'r', encoding='utf-8-sig') as f:
            train_lines = f.readlines()[:1000]
        train_lines = [x.replace(' ', '').strip() for x in train_lines]

        with codecs.open('./msra/train/tags.txt', 'r', encoding='utf-8-sig') as f:
            train_tags = f.readlines()[:1000]
        train_tags = [x.strip().split(' ') for x in train_tags]
        train_tags = [[label2id.get(x, 0) for x in tag] for tag in train_tags]  # 使用get避免KeyError

        # 加载验证数据 - 使用utf-8编码
        with codecs.open('./msra/val/sentences.txt', 'r', encoding='utf-8-sig') as f:
            val_lines = f.readlines()[:100]
        val_lines = [x.replace(' ', '').strip() for x in val_lines]

        with codecs.open('./msra/val/tags.txt', 'r', encoding='utf-8-sig') as f:
            val_tags = f.readlines()[:100]
        val_tags = [x.strip().split(' ') for x in val_tags]
        val_tags = [[label2id.get(x, 0) for x in tag] for tag in val_tags]

        # 数据验证和清洗
        valid_train_data = []
        valid_train_tags = []
        for i, (line, tags) in enumerate(zip(train_lines, train_tags)):
            if len(line) > 0 and len(tags) == len(list(line)):
                valid_train_data.append(line)
                valid_train_tags.append(tags)

        valid_val_data = []
        valid_val_tags = []
        for i, (line, tags) in enumerate(zip(val_lines, val_tags)):
            if len(line) > 0 and len(tags) == len(list(line)):
                valid_val_data.append(line)
                valid_val_tags.append(tags)

        print(f"训练数据: {len(valid_train_data)} 条, 验证数据: {len(valid_val_data)} 条")
        return valid_train_data, valid_train_tags, valid_val_data, valid_val_tags

    except Exception as e:
        print(f"数据加载失败: {e}")
        # 备用示例数据 - 确保编码正确
        sample_sentences = [
            "北京时间今天上午苹果公司在加州库比蒂诺发布了新款iPhone",
            "马云于1999年在杭州创立了阿里巴巴集团"
        ]
        sample_tags = [
            [0, 0, 0, 0, 0, 1, 2, 0, 5, 6, 6, 0, 0, 0, 0],
            [3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 1, 2, 2, 0]
        ]
        return sample_sentences, sample_tags, sample_sentences, sample_tags


def initialize_model_and_tokenizer(model_path):
    """初始化模型和tokenizer - 修复警告问题"""
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 设置填充token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(tag_type),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        print("模型和分词器初始化成功")
        return tokenizer, model

    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise


def tokenize_and_align_labels(tokenizer, sentences, all_labels):
    """对数据进行tokenize并对齐标签"""
    # 使用tokenizer处理多个句子
    tokenized_inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        is_split_into_words=False
    )

    labels = []
    for i in range(len(sentences)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label = all_labels[i]
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        # 确保长度一致
        current_length = len(tokenized_inputs["input_ids"][i])
        if len(label_ids) > current_length:
            label_ids = label_ids[:current_length]
        elif len(label_ids) < current_length:
            label_ids.extend([-100] * (current_length - len(label_ids)))

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def setup_lora(model):
    """设置LoRA配置"""
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_training_args():
    """配置训练参数 - 添加评估策略"""
    training_args = TrainingArguments(
        output_dir="./models/qwen3-0.6B-ner-output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to=None,
        eval_strategy="epoch",  # 改为每个epoch评估一次
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
    )
    return training_args


def compute_metrics(eval_pred):
    """计算评估指标 - 新增函数"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # 移除padding部分的标签（-100）
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]

    # 计算token级别的准确率
    correct = 0
    total = 0
    for pred_list, label_list in zip(true_predictions, true_labels):
        for p, l in zip(pred_list, label_list):
            if p == l:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    # 使用seqeval计算实体级别的指标
    try:
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
    except:
        precision, recall, f1 = 0, 0, 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def predict_entities(model, tokenizer, text):
    """预测实体 - 修复未识别实体问题"""
    try:
        # Tokenize输入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True  # 获取字符偏移量
        ).to(model.device)

        # 预测
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取预测结果
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 关键修复：正确解码token
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [id2label[pred.item()] for pred in predictions[0]]

        # 提取实体
        entities = []
        current_entity = ""
        current_type = ""

        # 获取字符偏移量
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        for i in range(len(tokens)):
            token = tokens[i]
            label = predicted_labels[i] if i < len(predicted_labels) else 'O'

            # 跳过特殊token
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                if current_entity:
                    entities.append((current_entity, current_type))
                    current_entity = ""
                    current_type = ""
                continue

            # 处理子词标记
            if token.startswith("##"):
                token = token[2:]

            # 获取当前token在原始文本中的位置
            start, end = offset_mapping[i]
            if start == 0 and end == 0:  # 跳过特殊token
                continue

            if label.startswith("B-"):
                # 开始新实体
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = text[start:end]  # 直接从原始文本中提取
                current_type = label[2:]
            elif label.startswith("I-") and current_entity and current_type == label[2:]:
                # 继续当前实体 - 确保类型匹配
                # 直接从原始文本中提取，避免拼接问题
                current_entity = text[offset_mapping[i][0]:offset_mapping[i][1]]
            else:
                # 结束当前实体
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""

        # 添加最后一个实体
        if current_entity:
            entities.append((current_entity, current_type))

        return entities

    except Exception as e:
        print(f"预测实体时出错: {e}")
        return []


def comprehensive_model_evaluation(model, tokenizer, val_sentences, val_tags):
    """综合模型评估 - 新增函数"""
    print("\n" + "=" * 60)
    print("=== 模型综合性能评估 ===")
    print("=" * 60)

    # 1. 使用Trainer进行评估
    print("1. 使用Trainer进行评估...")
    val_encodings = tokenize_and_align_labels(tokenizer, val_sentences, val_tags)
    val_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_encodings["labels"]
    })

    training_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=8,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print("评估结果:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # 2. 详细分类报告
    print("\n2. 生成详细分类报告...")
    all_predictions = []
    all_true_labels = []

    model.eval()
    with torch.no_grad():
        for i, (sentence, true_tags) in enumerate(zip(val_sentences[:50], val_tags[:50])):  # 限制数量避免内存问题
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # 对齐长度
            min_len = min(len(predictions), len(true_tags), len(inputs["input_ids"][0]))
            predictions = predictions[:min_len]
            true_tags = true_tags[:min_len]

            # 过滤padding
            for j in range(min_len):
                if true_tags[j] != -100 and j < len(predictions):
                    all_predictions.append(predictions[j])
                    all_true_labels.append(true_tags[j])

    # 生成分类报告
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(all_true_labels, all_predictions, target_names=target_names, digits=4))

    # 3. 测试样例预测
    print("\n3. 测试样例预测:")
    test_texts = [
        "北京时间今天上午苹果公司在加州库比蒂诺发布了新款iPhone",
        "马云于1999年在杭州创立了阿里巴巴集团",
        "李华明天将在北京清华大学参加会议",
        "河南新闻广播找一下啊是新闻台",
        "播放香港电台的王菲的经典音乐"
    ]

    for i, text in enumerate(test_texts):
        entities = predict_entities(model, tokenizer, text)
        print(f"\n样例 {i + 1}: {text}")
        if entities:
            print("识别实体:")
            for entity, label in entities:
                print(f"  {entity}: {label}")
        else:
            print("未识别到实体")

    # 4. 绘制混淆矩阵
    print("\n4. 生成混淆矩阵...")
    try:
        cm = confusion_matrix(all_true_labels, all_predictions, labels=range(len(tag_type)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tag_type, yticklabels=tag_type)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('NER模型混淆矩阵')
        plt.tight_layout()
        plt.savefig('./models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("混淆矩阵已保存为 './models/confusion_matrix.png'")
    except Exception as e:
        print(f"生成混淆矩阵时出错: {e}")

    print("\n" + "=" * 60)
    print("模型评估完成！")
    print("=" * 60)


def pre_training_validation(model, tokenizer, train_sentences, train_tags):
    """训练前验证 - 新增函数"""
    print("\n" + "=" * 60)
    print("=== 训练前验证 ===")
    print("=" * 60)

    # 1. 验证数据预处理
    print("1. 验证数据预处理...")
    try:
        sample_sentence = train_sentences[0]
        sample_tags = train_tags[0]
        print(f"样本句子: {sample_sentence}")
        print(f"样本标签: {sample_tags}")

        # 处理数据
        tokenized = tokenize_and_align_labels(tokenizer, [sample_sentence], [sample_tags])
        print("数据预处理成功!")
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return False

    # 2. 验证模型预测
    print("\n2. 验证模型预测...")
    try:
        test_texts = [
            "北京时间今天上午苹果公司在加州库比蒂诺发布了新款iPhone",
            "马云于1999年在杭州创立了阿里巴巴集团"
        ]

        for i, text in enumerate(test_texts):
            entities = predict_entities(model, tokenizer, text)
            print(f"\n测试文本 {i + 1}: {text}")
            if entities:
                print("预测实体:")
                for entity, label in entities:
                    print(f"  {entity}: {label}")
            else:
                print("未识别到实体")

        print("模型预测验证成功!")
    except Exception as e:
        print(f"模型预测验证失败: {e}")
        return False

    # 3. 验证LoRA设置
    print("\n3. 验证LoRA设置...")
    try:
        lora_model = setup_lora(model)
        print("LoRA设置成功!")
        print(lora_model)
    except Exception as e:
        print(f"LoRA设置失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("训练前验证通过!")
    print("=" * 60)
    return True


def main():
    """主执行函数 - 包含完整训练和验证流程"""
    try:
        # 1. 加载数据
        print("步骤1: 加载NER数据...")
        train_sentences, train_tags, val_sentences, val_tags = load_ner_data()

        # 2. 初始化模型和tokenizer
        print("步骤2: 初始化模型和tokenizer...")
        model_path = "./models/Qwen/Qwen3-0.6B"
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 新增：训练前验证
        print("\n步骤3: 执行训练前验证...")
        validation_passed = pre_training_validation(model, tokenizer, train_sentences, train_tags)

        if not validation_passed:
            print("训练前验证失败，请检查错误后重新运行程序。")
            return None, None

        # 询问用户是否继续训练
        user_input = input("\n训练前验证通过，是否继续训练模型？(y/n): ")
        if user_input.lower() != 'y':
            print("用户选择不继续训练，程序退出。")
            return model, tokenizer

        # 3. 处理数据
        print("步骤4: 处理训练数据...")
        train_encodings = tokenize_and_align_labels(tokenizer, train_sentences, train_tags)
        val_encodings = tokenize_and_align_labels(tokenizer, val_sentences, val_tags)

        # 转换为Dataset
        train_dataset = Dataset.from_dict({
            "input_ids": train_encodings["input_ids"],
            "attention_mask": train_encodings["attention_mask"],
            "labels": train_encodings["labels"]
        })

        val_dataset = Dataset.from_dict({
            "input_ids": val_encodings["input_ids"],
            "attention_mask": val_encodings["attention_mask"],
            "labels": val_encodings["labels"]
        })

        print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")

        # 4. 设置LoRA
        print("步骤5: 设置LoRA...")
        model = setup_lora(model)

        # 5. 配置训练参数
        print("步骤6: 配置训练参数...")
        training_args = setup_training_args()

        # 6. 创建Trainer并开始训练
        print("步骤7: 开始训练...")
        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 训练模型
        print("开始训练模型...")
        trainer.train()

        # 7. 保存模型
        print("步骤8: 保存模型...")
        os.makedirs("./models/output_Qwen3-0.6B-ner", exist_ok=True)
        trainer.save_model("./models/output_Qwen3-0.6B-ner")
        tokenizer.save_pretrained("./models/output_Qwen3-0.6B-ner")

        print("训练完成!")

        # 8. 综合模型评估
        print("步骤9: 开始模型验证...")
        comprehensive_model_evaluation(model, tokenizer, val_sentences, val_tags)

        return model, tokenizer

    except Exception as e:
        print(f"主函数执行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # 执行主函数
    model, tokenizer = main()