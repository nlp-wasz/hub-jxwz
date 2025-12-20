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
import json


def convert_qa_data_to_instruction_format():
    """将问答数据转换为指令微调格式"""
    # 加载训练和验证数据
    train_data = json.load(open('./cmrc2018_public/train.json', 'r', encoding='utf-8'))
    dev_data = json.load(open('./cmrc2018_public/dev.json', 'r', encoding='utf-8'))

    # 准备训练数据
    train_instructions = []
    for paragraph in train_data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            question = qa['question']
            answer = qa['answers'][0]['text']

            instruction = f"根据以下内容回答问题：{context}"
            input_text = f"问题：{question}"
            output = f"答案：{answer}"

            train_instructions.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })

    # 准备验证数据
    val_instructions = []
    for paragraph in dev_data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            question = qa['question']
            answer = qa['answers'][0]['text']

            instruction = f"根据以下内容回答问题：{context}"
            input_text = f"问题：{question}"
            output = f"答案：{answer}"

            val_instructions.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })

    return train_instructions[:1000], val_instructions[:100]  # 限制数据量


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理问答数据"""
    # 转换数据格式
    train_data, val_data = convert_qa_data_to_instruction_format()

    # 合并训练和验证数据用于训练
    all_data = train_data + val_data

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    # 转换为Hugging Face Dataset
    ds = Dataset.from_pandas(df)

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
def process_func(example, tokenizer, max_length=384):
    """
       处理单个样本的函数
       将指令和输出转换为模型训练格式
       """
    # 构建指令部分 - 修改为问答任务
    instruction_text = f"<|im_start|>system\n你是一个问答助手，请根据给定的内容回答问题<|im_end|>\n<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
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
        output_dir="./output_Qwen1.5_qa",
        per_device_train_batch_size=4,  # 减小batch size以适应更长的序列
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=3,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"
    )


# 预测函数
def predict_qa(model, tokenizer, context, question, device='cpu'):
    """预测问答结果"""
    messages = [
        {"role": "system", "content": "你是一个问答助手，请根据给定的内容回答问题"},
        {"role": "user", "content": f"根据以下内容回答问题：{context}\n问题：{question}"}
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
    # 1. 加载数据
    print("加载数据...")
    ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week07/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    print(model)
    # 模型架构
    # Qwen3ForCausalLM(
    #     (model): Qwen3Model(
    #     (embed_tokens): Embedding(151936, 1024)
    # (layers): ModuleList(
    #     (0 - 27): 28
    # x
    # Qwen3DecoderLayer(
    #     (self_attn): Qwen3Attention(
    #     (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
    # (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
    # (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
    # (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
    # (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
    # (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
    # )
    # (mlp): Qwen3MLP(
    #     (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
    # (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
    # (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
    # (act_fn): SiLU()
    # )
    # (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
    # (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
    # )
    # )
    # (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    # (rotary_emb): Qwen3RotaryEmbedding()
    # )
    # (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
    # )

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])
    eval_ds = Dataset.from_pandas(ds.to_pandas()[-200:])

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

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
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    # print("保存模型...")
    # trainer.save_model()
    # tokenizer.save_pretrained("./output_Qwen")


# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week07/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week07/output_Qwen1.5_qa/checkpoint-39")
    model.cpu()

    # 测试预测
    # 测试问答
    test_cases = [
        {
            "context": "阿里巴巴集团成立于1999年，由马云和他的团队在杭州创立。阿里巴巴集团是中国最大的电子商务公司之一。",
            "question": "阿里巴巴集团是哪一年成立的？"
        },
        {
            "context": "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python以其简洁易读的语法而闻名。",
            "question": "Python是由谁创建的？"
        }
    ]

    for case in test_cases:
        result = predict_qa(model, tokenizer, case["context"], case["question"])
        print(f"上下文: {case['context']}")
        print(f"问题: {case['question']}")
        print(f"回答: {result}")
        print("-" * 50)


if __name__ == "__main__":
    # 执行主函数
    # result_df = main()

    # 单独测试
    test_single_example()
