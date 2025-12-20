import json
import pandas as pd
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


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理CMRC2018知识问答数据"""

    # 加载训练和验证数据
    train_data = json.load(open('./cmrc2018_public/train.json','r',encoding='utf-8'))
    dev_data = json.load(open('./cmrc2018_public/dev.json','r',encoding='utf-8'))

    # 转换为问答格式
    def convert_to_qa_format(data):
        samples = []
        for paragraph in data['data']:
            context = paragraph['paragraphs'][0]['context']
            for qa in paragraph['paragraphs'][0]['qas']:
                question = qa['question']
                if qa['answers']:
                    answer = qa['answers'][0]['text']
                else:
                    answer = "无答案"

                # 构建指令格式
                instruction = f"基于以下文本回答问题：\n文本：{context}\n问题：{question}"

                samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": answer
                })
        return samples

    train_samples = convert_to_qa_format(train_data)[:1000]  # 限制数据量用于演示
    dev_samples = convert_to_qa_format(dev_data)[:200]

    # 转换为DataFrame
    train_df = pd.DataFrame(train_samples)
    dev_df = pd.DataFrame(dev_samples)

    # 转换为Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    dev_ds = Dataset.from_pandas(dev_df)

    return train_ds, dev_ds


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建完整的对话格式
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，请基于提供的文本准确回答问题。"},
        {"role": "user", "content": example['instruction'] + example['input']},
        {"role": "assistant", "content": example['output']}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    # 构建标签（只计算assistant回复部分的loss）
    # 找到assistant回复开始的位置
    assistant_start = formatted_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_content = formatted_text[assistant_start:]

    # 计算assistant内容的token长度
    assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)

    # 构建labels：非assistant部分设为-100
    labels = [-100] * (len(tokenized["input_ids"]) - len(assistant_tokens)) + assistant_tokens

    # 确保长度一致
    if len(labels) > len(tokenized["input_ids"]):
        labels = labels[:len(tokenized["input_ids"])]
    elif len(labels) < len(tokenized["input_ids"]):
        labels = labels + [-100] * (len(tokenized["input_ids"]) - len(labels))

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
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
        output_dir="./output_qwen_qa",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=50,
        eval_steps=100,
        save_steps=200,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        warmup_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )


# 预测函数
def predict_answer(model, tokenizer, context, question, device='cuda'):
    """预测问题的答案"""
    # 构建输入
    instruction = f"基于以下文本回答问题：\n文本：{context}\n问题：{question}"

    messages = [
        {"role": "system", "content": "你是一个知识问答助手，请基于提供的文本准确回答问题。"},
        {"role": "user", "content": instruction}
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
            max_new_tokens=128,
            do_sample=False,  # 使用贪婪搜索以获得更确定的输出
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 评估函数
def evaluate_model(model, tokenizer, eval_data, device='cuda', num_samples=10):
    """评估模型性能"""
    model.eval()
    results = []

    for i in range(min(num_samples, len(eval_data))):
        sample = eval_data[i]
        context = sample['instruction'].split('文本：')[1].split('\n问题：')[0]
        question = sample['instruction'].split('问题：')[1]
        true_answer = sample['output']

        try:
            pred_answer = predict_answer(model, tokenizer, context, question, device)

            # 简单的匹配度检查
            is_correct = true_answer.lower() in pred_answer.lower() or pred_answer.lower() in true_answer.lower()

            results.append({
                'question': question,
                'true_answer': true_answer,
                'pred_answer': pred_answer,
                'is_correct': is_correct
            })

            print(f"问题 {i + 1}: {question}")
            print(f"真实答案: {true_answer}")
            print(f"预测答案: {pred_answer}")
            print(f"是否正确: {is_correct}")
            print("-" * 50)

        except Exception as e:
            print(f"评估样本 {i + 1} 时出错: {e}")
            continue

    # 计算准确率
    accuracy = sum([r['is_correct'] for r in results]) / len(results) if results else 0
    print(f"\n评估准确率: {accuracy:.2%}")

    return results, accuracy


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds, eval_ds = load_and_preprocess_data()
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"  # 请根据实际路径修改
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    tokenized_train_ds = train_ds.map(
        process_func_with_tokenizer,
        remove_columns=train_ds.column_names
    )

    tokenized_eval_ds = eval_ds.map(
        process_func_with_tokenizer,
        remove_columns=eval_ds.column_names
    )

    # 4. 设置LoRA
    print("设置LoRA...")
    model = setup_lora(model)

    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 6. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    # 开始训练
    trainer.train()

    # 7. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_qwen_qa")

    # 8. 评估模型
    print("评估模型性能...")
    eval_results, accuracy = evaluate_model(model, tokenizer, eval_ds)

    return model, tokenizer, eval_results, accuracy


# 单独测试函数
def test_single_example():
    """测试单个例子"""
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_qwen_qa/")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试例子
    context = "《红楼梦》是中国古典四大名著之一，作者是清代作家曹雪芹。小说以贾、史、王、薛四大家族的兴衰为背景，以贾宝玉、林黛玉、薛宝钗的爱情婚姻故事为主线。"
    question = "《红楼梦》的作者是谁？"

    result = predict_answer(model, tokenizer, context, question)
    print(f"上下文: {context}")
    print(f"问题: {question}")
    print(f"预测答案: {result}")


if __name__ == "__main__":
    # 执行主函数
    model, tokenizer, eval_results, accuracy = main()

    # 单独测试
    print("\n单独测试例子:")
    test_single_example()
