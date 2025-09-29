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
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
train = json.load(open('./cmrc2018_public/train.json'))
dev = json.load(open('./cmrc2018_public/dev.json'))

# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  
    )

    return tokenizer, model

# 准备训练数据
def prepare_dataset(data):
    contexts = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            contexts.append(context)
            questions.append(qa['question'])
            # 只取第一个答案
            if qa['answers']:
                answers.append(qa['answers'][0]['text'])
            else:
                answers.append("无法回答")

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

# 转换为Hugging Face Dataset
train_dataset = Dataset.from_dict(train_dataset_dict)
val_dataset = Dataset.from_dict(val_dataset_dict)

# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将问题和上下文转换为模型训练格式
    """
    # 获取上下文、问题和答案
    context = example['context']
    question = example['question']
    answer = example['answer']
    
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在进行问答任务，请根据提供的上下文回答问题。如果上下文中没有相关信息，请回答\"无法回答\"。<|im_end|>\n<|im_start|>user\n上下文：{context}\n问题：{question}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    
    # 构建响应部分
    response = tokenizer(answer, add_special_tokens=False)
    
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
        output_dir="./output_Qwen_QA",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  
    )

# 预测函数
def predict_answer(model, tokenizer, context, question, device='cpu'):
    """预测问题的答案"""
    messages = [
        {"role": "system", "content": "现在进行问答任务，请根据提供的上下文回答问题。如果上下文中没有相关信息，请回答\"无法回答\"。"},
        {"role": "user", "content": f"上下文：{context}\n问题：{question}"}
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
            temperature=0.1,  
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()

# 主函数
def main():
    """主执行函数"""
    # 1. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 2. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_train_ds = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    tokenized_eval_ds = val_dataset.map(process_func_with_tokenizer, remove_columns=val_dataset.column_names)
    
    # 3. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)
    
    # 4. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()
    
    # 5. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )
    
    trainer.train()
    
    # 6. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_Qwen_QA")
    
    # 7. 评估模型
    print("评估模型...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")
    
    # 8. 测试预测
    print("\n在验证集上测试:")
    for i in range(min(3, len(val_contexts))):
        context = val_contexts[i]
        question = val_questions[i]
        expected_answer = val_answers[i]
        
        predicted_answer = predict_answer(model, tokenizer, context, question, device=device)
        
        print(f"问题 {i + 1}: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"匹配: {expected_answer == predicted_answer}")
        print()

if __name__ == "__main__":
    main()