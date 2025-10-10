import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据加载和预处理
def load_and_preprocess_data(data_file):
    """加载和预处理CMRC2018数据集"""
    data = json.load(open(data_file))

    # 准备训练数据
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append(qa['answers'][0]['text'])

    # 创建DataFrame
    qa_data = pd.DataFrame({
        "context": paragraphs,
        "question": questions,
        "answer": answers
    })

    # 转换为指令格式
    qa_data["instruction"] = qa_data.apply(
        lambda row: f"回答问题，只输出实体名称。文本：{row['context']}问题：{row['question']}",
        axis=1
    )
    qa_data["output"] = qa_data["answer"]
    qa_data["input"] = ""

    # 保留必要的列
    qa_data = qa_data[["instruction", "output", "input"]]

    # 转换为Hugging Face Dataset
    ds = Dataset.from_pandas(qa_data)

    return ds


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
    instruction_text = f"<|im_start|>system\n你是一个专注于实体识别和知识问答的助手，请只输出实体，不要添加额外文字。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 确保输出不包含"The answer is"等多余文本
    clean_output = example['output'].strip()

    # 构建响应部分
    response = tokenizer(f"{clean_output}", add_special_tokens=False)

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

        # 对什么层的一些模块进行高效微调
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
        output_dir="./output_Qwen_NER_QA",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=5e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_answer(model, tokenizer, context, question, device='cpu'):
    """预测问题的答案"""
    # 构建提示
    prompt = f"从以下文本中识别出能够回答问题的实体。\n\n文本：{context}\n\n问题：{question}"

    messages = [
        {"role": "system",
         "content": "你是一个专注于实体识别和知识问答的助手。只输出识别出的实体，不要添加任何额外文字，不要生成新问题。"},
        {"role": "user", "content": prompt}
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
            max_new_tokens=20,  # 进一步减少生成长度
            do_sample=False,  # 使用贪婪解码
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True  # 启用早停以避免生成额外内容
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 后处理清理输出
    response = clean_model_output(raw_response)

    return response


def clean_model_output(text):
    """清理模型输出，移除多余内容"""
    # 检查是否包含"Human:"标记，如果有则截断
    if "Human:" in text:
        text = text.split("Human:")[0].strip()

    # 检查是否包含"The answer is"等前缀
    if "The answer is:" in text or "The answer is" in text:
        for pattern in ["The answer is:", "The answer is"]:
            if pattern in text:
                parts = text.split(pattern)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # 只保留第一个句子或到第一个换行符
                    answer = answer.split("\n")[0].split(".")[0].strip()
                    return answer

    # 如果没有特定模式，返回处理后的文本
    return text.split("\n")[0].strip()


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds = load_and_preprocess_data('./cmrc2018_public/train.json')
    dev_ds = load_and_preprocess_data('./cmrc2018_public/dev.json')

    # 数据切分，避免使用过多数据
    train_ds = Dataset.from_pandas(train_ds.to_pandas().iloc[:500])
    dev_ds = Dataset.from_pandas(dev_ds.to_pandas().iloc[:50])

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../../../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    dev_tokenized = dev_ds.map(process_func_with_tokenizer, remove_columns=dev_ds.column_names)

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
        eval_dataset=dev_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 7. 保存模型
    print("保存模型...")
    trainer.save_model("./output_Qwen_NER_QA_final")

    # 8. 测试模型
    print("\n测试模型...")

    # 获取原始数据用于测试
    test_samples = dev_ds.to_pandas().iloc[:3]

    for i, row in test_samples.iterrows():
        context = row['instruction'].split('文本：')[1].split('\n\n问题：')[0]
        question = row['instruction'].split('问题：')[1]
        expected_answer = row['output']

        predicted_answer = predict_answer(model, tokenizer, context, question, device=device)

        print(f"问题 {i + 1}: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"匹配: {expected_answer in predicted_answer}")
        print()


# 测试已训练模型
def test_trained_model():
    # 加载训练好的模型和tokenizer
    model_path = "../../../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen_NER_QA_final")

    # 测试样例
    context = "中国的首都是北京，上海是中国最大的城市，广州是南方的重要城市。"
    question = "中国的首都是哪里？"

    answer = predict_answer(model, tokenizer, context, question, device=device)
    print(f"问题: {question}")
    print(f"上下文: {context}")
    print(f"预测答案: {answer}")


if __name__ == "__main__":
    # 执行主函数进行训练
    main()

    # 或者测试已训练的模型
    test_trained_model()