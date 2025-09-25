import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train = json.load(open('./cmrc2018_public/train.json'))
dev = json.load(open('./cmrc2018_public/dev.json'))

# 初始化tokenizer和模型
model_name = "Qwen/Qwen2-7B"  # 使用Qwen模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 输入 11 token -》 13 token -》 13 * 768 （特征） -》 13 * 2 对应回答在原文的位置（通过开头和结尾确定的）

# 准备训练数据
def prepare_dataset(data):
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append({
                'answer_start': [qa['answers'][0]['answer_start']],
                'text': [qa['answers'][0]['text']]
            })

    return paragraphs, questions, answers


# 准备训练和验证数据
train_paragraphs, train_questions, train_answers = prepare_dataset(train)
val_paragraphs, val_questions, val_answers = prepare_dataset(dev)

# 创建数据集字典
train_dataset_dict = {
    'context': train_paragraphs[:1000],
    'question': train_questions[:1000],
    'answers': train_answers[:1000]
}

val_dataset_dict = {
    'context': val_paragraphs[:100],
    'question': val_questions[:100],
    'answers': val_answers[:100]
}

# 转换为Hugging Face Dataset
train_dataset = Dataset.from_dict(train_dataset_dict)
val_dataset = Dataset.from_dict(val_dataset_dict)


# 预处理函数
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于可能有溢出，需要重新映射样本
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取对应的原始样本
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        # 如果没有答案，设置默认值
        if len(answer["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 找到token的起始和结束位置
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 找到context的开始和结束
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # 如果答案完全在context之外，标记为不可回答
        if offset_mapping[i][context_start][0] > end_char or offset_mapping[i][context_end][1] < start_char:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # 否则找到答案的token位置
            idx = context_start
            while idx <= context_end and offset_mapping[i][idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[i][idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


# 应用预处理
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

# 设置LoRA配置
peft_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

# 应用LoRA适配器
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./qa-qwen-lora-model",
    learning_rate=3e-4,  # LoRA通常使用更高的学习率
    per_device_train_batch_size=4,  # 根据显存调整
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./qwen-qa-logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    fp16=True,  # 启用混合精度训练
    # 添加设备相关设置
    no_cuda=(device.type != "cuda"),  # 如果不是CUDA设备，禁用CUDA
)

# 数据收集器
data_collator = DefaultDataCollator()

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
print("开始训练QA模型...")
trainer.train()

# 保存模型
trainer.save_model()
tokenizer.save_pretrained('./qa-qwen-lora-model')

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")


# 预测函数
def predict(context, question):
    # 将模型移动到CPU进行预测
    model.to('cpu')

    # Tokenize输入
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测的起始和结束位置
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 找到最可能的答案跨度
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    # 将token位置转换为字符位置
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer_tokens = all_tokens[start_idx:end_idx + 1]

    # 将token转换回文本
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # 清理答案
    answer = answer.replace(" ", "").replace("##", "")

    # 将模型移回原设备
    model.to(device)

    return answer

# 在验证集上测试几个样本
print("\n在验证集上测试:")
for i in range(min(3, len(val_paragraphs))):
    context = val_paragraphs[i]
    question = val_questions[i]
    expected_answer = val_answers[i]['text'][0]

    predicted_answer = predict(context, question)

    print(f"问题 {i + 1}: {question}")
    print(f"上下文: {context[:100]}...")
    print(f"预期答案: {expected_answer}")
    print(f"预测答案: {predicted_answer}")
    print("-" * 50)
