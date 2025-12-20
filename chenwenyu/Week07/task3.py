import json
import codecs

import pandas as pd
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

from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train = json.load(open('../cmrc2018_public/train.json'))
dev = json.load(open('../cmrc2018_public/dev.json'))

def prepare_dataset(data):
    """准备阅读理解数据集"""
    contexts = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            contexts.append(context)
            questions.append(qa['question'])
            # 提取答案文本
            answers.append(qa['answers'][0]['text'] if qa['answers'] else "")

    return contexts, questions, answers

# 准备训练和验证数据
train_contexts, train_questions, train_answers = prepare_dataset(train)
val_contexts, val_questions, val_answers = prepare_dataset(dev)

# 创建数据集字典
train_dataset_dict = {
    'context': train_contexts[:1000],
    'question': train_questions[:1000],
    'answer': train_answers[:1000]
}

val_dataset_dict = {
    'context': val_contexts[:100],
    'question': val_questions[:100],
    'answer': val_answers[:100]
}

# 初始化tokenizer和模型
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

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    return tokenizer, model

# 数据处理函数 - 修改为阅读理解格式
def process_func(example, tokenizer, max_length=512):
    """
    处理阅读理解样本的函数
    格式: 基于上下文回答问题
    """
    # 构建阅读理解提示
    prompt = f"""请基于以下文本回答问题：

文本：{example['context']}

问题：{example['question']}

答案："""
    
    # Tokenize提示部分
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    
    # Tokenize答案部分
    answer_tokens = tokenizer(example['answer'], add_special_tokens=False)
    
    # 组合输入ID和注意力掩码
    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"] + [1]
    
    # 构建标签（提示部分用-100忽略，只计算答案部分的损失）
    labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"] + [tokenizer.pad_token_id]
    
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
        output_dir="./output_reading_comprehension/",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        eval_strategy="steps",
        save_total_limit=2
    )

# 阅读理解预测函数
def predict_answer(model, tokenizer, context, question, device='cuda'):
    """基于上下文预测问题的答案"""
    
    # 构建阅读理解提示
    prompt = f"""请基于以下文本回答问题：

文本：{context}

问题：{question}

答案："""
    
    # Tokenize输入
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # 提取生成的答案（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return answer.strip()

# 批量预测
def batch_predict_qa(model, tokenizer, contexts, questions, device='cuda'):
    """批量预测阅读理解答案"""
    pred_answers = []
    
    for context, question in tqdm(zip(contexts, questions), desc="预测答案", total=len(questions)):
        try:
            pred_answer = predict_answer(model, tokenizer, context, question, device)
            pred_answers.append(pred_answer)
        except Exception as e:
            print(f"预测问题时出错: {e}")
            pred_answers.append("")
    
    return pred_answers

# 评估函数
def evaluate_qa_model(model, tokenizer, test_contexts, test_questions, test_answers, device='cuda'):
    """评估模型性能"""
    print("开始评估模型...")
    
    pred_answers = batch_predict_qa(model, tokenizer, test_contexts, test_questions, device)
    
    # 打印一些示例
    print("\n=== 预测示例 ===")
    for i in range(min(3, len(test_questions))):
        print(f"问题: {test_questions[i]}")
        print(f"真实答案: {test_answers[i]}")
        print(f"预测答案: {pred_answers[i]}")
        print("-" * 50)
    
    return pred_answers

# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    
    train_data = pd.DataFrame(train_dataset_dict)
    train_data = train_data[['context', 'question', 'answer']]
    ds = Dataset.from_pandas(train_data)
    
    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../../../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)
    
    # 4. 划分训练集和验证集
    split_idx = int(0.8 * len(tokenized_ds))
    train_tokenized = tokenized_ds.select(range(split_idx))
    eval_tokenized = tokenized_ds.select(range(split_idx, len(tokenized_ds)))
    
    print(f"训练集大小: {len(train_tokenized)}")
    print(f"验证集大小: {len(eval_tokenized)}")
    
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
            pad_to_multiple_of=8
        ),
    )
    
    trainer.train()
    
    # 保存最终模型
    trainer.save_model()
    print("训练完成，模型已保存！")
    
    return model, tokenizer

# 单独测试函数
def test_single_example(model=None, tokenizer=None):
    """测试单个阅读理解示例"""
    if model is None or tokenizer is None:
        model_path = "../../../models/Qwen/Qwen3-0.6B/"
        tokenizer, model = initialize_model_and_tokenizer(model_path)
        
        # 加载训练好的LoRA权重
        try:
            model.load_adapter("./output_reading_comprehension/checkpoint-xxx/")  # 替换为实际checkpoint路径
        except:
            print("使用基础模型进行测试...")
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试用例
    test_context = "ChatGPT 是由美国人工智能研究公司 OpenAI 开发的一款大型语言模型。它基于 GPT（Generative Pre-trained Transformer）架构，通过海量的文本数据进行训练，能够理解和生成人类语言，实现与用户的自然对话。"
    test_question = "谁开发的ChatGPT？"
    
    print("=== 阅读理解测试 ===")
    print(f"文本: {test_context}")
    print(f"问题: {test_question}")
    
    answer = predict_answer(model, tokenizer, test_context, test_question)
    print(f"答案: {answer}")

if __name__ == "__main__":
    # 执行主函数
    trained_model, trained_tokenizer = main()
    
    # 测试训练好的模型
    test_single_example(trained_model, trained_tokenizer)
    
    # 在验证集上评估
    print("\n在验证集上评估模型...")
    val_contexts = val_dataset_dict['context']
    val_questions = val_dataset_dict['question'] 
    val_answers = val_dataset_dict['answer']
    
    pred_answers = evaluate_qa_model(
        trained_model, trained_tokenizer, 
        val_contexts, val_questions, val_answers
    )
