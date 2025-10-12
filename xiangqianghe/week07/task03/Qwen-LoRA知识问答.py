import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import os
import gc

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


def load_and_preprocess_data():
    """加载和预处理知识问答数据"""
    try:
        # 从CSV文件加载数据 - 假设格式为: 问题\t答案
        train_data = pd.read_csv('../qa-dataset.csv', sep='\t', header=None)

        # 重命名列并添加输入列
        train_data.columns = ["input", "output"]
        train_data["instruction"] = "你是一个知识问答助手，请根据知识库回答用户的问题。"

        # 转换为Hugging Face Dataset
        dataset = Dataset.from_pandas(train_data)
        print(f"知识问答数据加载成功，共 {len(dataset)} 条样本")
        return dataset

    except Exception as e:
        print(f"数据加载失败: {e}")
        # 备用示例数据 - 知识问答格式
        sample_data = [
            {
                "instruction": "你是一个知识问答助手，请根据知识库回答用户的问题。",
                "input": "什么是人工智能？",
                "output": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器和软件。这些任务包括学习、推理、问题解决、感知和语言理解。"
            },
            {
                "instruction": "你是一个知识问答助手，请根据知识库回答用户的问题。",
                "input": "机器学习有哪些主要类型？",
                "output": "机器学习主要分为三类：监督学习（使用标记数据训练模型）、无监督学习（发现未标记数据中的模式）和强化学习（通过试错学习最优策略）。"
            },
            {
                "instruction": "你是一个知识问答助手，请根据知识库回答用户的问题。",
                "input": "深度学习与机器学习有什么区别？",
                "output": "深度学习是机器学习的一个子集，使用多层神经网络从数据中学习复杂模式。传统机器学习通常需要特征工程，而深度学习可以自动学习特征。"
            },
            {
                "instruction": "你是一个知识问答助手，请根据知识库回答用户的问题。",
                "input": "什么是Transformer架构？",
                "output": "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理任务。它摒弃了循环神经网络，支持并行计算，大大提高了训练效率。"
            }
        ]
        df = pd.DataFrame(sample_data)
        return Dataset.from_pandas(df)


def initialize_model_and_tokenizer(model_path):
    """初始化模型和tokenizer"""
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )

        # 设置填充token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 根据GPU可用性选择设备
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.bfloat16
            print("使用GPU设备")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("使用CPU设备")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # 启用梯度计算
        model.enable_input_require_grads()

        print("模型和分词器初始化成功")
        return tokenizer, model

    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise


def process_func(example, tokenizer, max_length=512):
    """知识问答数据处理函数"""
    # 构建指令格式 - 适配知识问答任务
    instruction_text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n" \
                       f"<|im_start|>user\n问题：{example['input']}<|im_end|>\n" \
                       f"<|im_start|>assistant\n答案："

    response_text = example['output']

    # tokenize
    instruction = tokenizer(
        instruction_text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=True,
        max_length=max_length - 100  # 预留答案空间
    )
    response = tokenizer(
        response_text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=True,
        max_length=100  # 限制答案长度
    )

    # 拼接输入和输出
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 标签处理（只计算answer部分的loss）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断到最大长度
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.1):
    """设置LoRA配置"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_training_args():
    """配置训练参数 - 动态适应硬件资源"""
    # 根据GPU可用性调整批大小和梯度累积步数
    if torch.cuda.is_available():
        per_device_batch_size = 2
        gradient_accumulation_steps = 4
        fp16 = True
        print("启用FP16混合精度训练")
    else:
        per_device_batch_size = 1
        gradient_accumulation_steps = 8
        fp16 = False
        print("使用FP32精度训练")

    training_args = TrainingArguments(
        output_dir="./models/qwen3-0.6B-qa-output",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=fp16,
        report_to=None,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    return training_args


def predict_answer(model, tokenizer, question, max_new_tokens=256):
    """预测单个问题的答案"""
    # 构建提示词 - 知识问答专用
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，请根据知识库准确、简洁地回答用户的问题。"},
        {"role": "user", "content": f"问题：{question}"}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)

    # 生成输出 - 调整参数以适应知识问答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # 降低随机性，提高准确性
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1  # 减少重复
    )

    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def batch_predict_answers(model, tokenizer, questions, batch_size=4):
    """批量预测问题答案"""
    results = []

    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(questions), batch_size), desc="批量回答问题"):
        batch_questions = questions[i:i + batch_size]
        batch_results = []

        for question in batch_questions:
            try:
                result = predict_answer(model, tokenizer, question)
                batch_results.append(result)
            except Exception as e:
                print(f"回答问题 '{question}' 时出错: {e}")
                batch_results.append("抱歉，我无法回答这个问题。")

        results.extend(batch_results)

    return results


def main():
    """主执行函数"""
    try:
        # 1. 加载数据
        print("步骤1: 加载知识问答数据...")
        ds = load_and_preprocess_data()

        # 检查数据量
        data_size = len(ds)
        print(f"数据集大小: {data_size} 条样本")

        if data_size < 10:
            print("警告: 数据量过少，建议增加训练数据")

        # 2. 初始化模型和tokenizer
        print("步骤2: 初始化模型和tokenizer...")
        model_path = "./models/Qwen/Qwen3-0.6B"  # 与下载路径一致
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 3. 处理数据
        print("步骤3: 处理训练数据...")
        process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
        tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

        # 4. 划分训练集和验证集（动态划分）
        print("步骤4: 划分训练集和验证集...")
        split_ratio = 0.8  # 80%训练，20%验证
        split_index = int(data_size * split_ratio)

        train_indices = list(range(split_index))
        eval_indices = list(range(split_index, data_size))

        train_ds = tokenized_ds.select(train_indices)
        eval_ds = tokenized_ds.select(eval_indices) if eval_indices else None

        print(f"训练集: {len(train_ds)} 条, 验证集: {len(eval_ds) if eval_ds else 0} 条")

        # 5. 设置LoRA
        print("步骤5: 设置LoRA...")
        model = setup_lora(model)

        # 6. 配置训练参数
        print("步骤6: 配置训练参数...")
        training_args = setup_training_args()

        # 7. 创建Trainer并开始训练
        print("步骤7: 开始训练...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_ds else None,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=8
            ),
        )

        # 清除内存缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer.train()

        # 8. 保存模型
        print("步骤8: 保存模型...")
        os.makedirs("./models/output_Qwen3-0.6B-qa", exist_ok=True)
        trainer.save_model("./models/output_Qwen3-0.6B-qa")
        tokenizer.save_pretrained("./models/output_Qwen3-0.6B-qa")

        print("知识问答模型训练完成!")
        return model, tokenizer

    except Exception as e:
        print(f"主函数执行出错: {e}")
        return None, None


def test_single_question():
    """测试单个问题"""
    try:
        model_path = "./models/Qwen/Qwen3-0.6B/"
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 尝试加载训练好的LoRA权重
        try:
            model.load_adapter("./models/output_Qwen3-0.6B-qa")
            print("成功加载训练好的LoRA权重")
        except:
            print("未找到训练好的权重，使用原始模型")

        # 测试预测
        test_question = "什么是机器学习？"
        result = predict_answer(model, tokenizer, test_question)
        print(f"问题: {test_question}")
        print(f"答案: {result}")

    except Exception as e:
        print(f"单例测试出错: {e}")


def test_batch_questions(model, tokenizer):
    """测试批量问题"""
    try:
        test_questions = [
            "人工智能有哪些应用领域？",
            "深度学习的工作原理是什么？",
            "自然语言处理的主要任务有哪些？",
            "计算机视觉和图像识别有什么区别？",
            "什么是神经网络？"
        ]

        print("\n=== 批量问答测试 ===")
        results = batch_predict_answers(model, tokenizer, test_questions, batch_size=2)

        # 打印结果
        for i, (question, answer) in enumerate(zip(test_questions, results)):
            print(f"\n问题 {i + 1}:")
            print(f"问题: {question}")
            print(f"答案: {answer}")

    except Exception as e:
        print(f"批量测试出错: {e}")


def interactive_qa(model, tokenizer):
    """交互式知识问答"""
    print("\n=== 进入交互式知识问答模式 ===")
    print("输入 '退出' 或 'quit' 结束对话")

    while True:
        try:
            question = input("\n请输入您的问题: ").strip()

            if question.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用知识问答系统！")
                break

            if not question:
                print("问题不能为空，请重新输入。")
                continue

            answer = predict_answer(model, tokenizer, question)
            print(f"\n答案: {answer}")

        except KeyboardInterrupt:
            print("\n\n对话结束。")
            break
        except Exception as e:
            print(f"回答问题时出错: {e}")


if __name__ == "__main__":
    # 执行主函数
    trained_model, trained_tokenizer = main()

    if trained_model is not None:
        # 交互式问答
        interactive_qa(trained_model, trained_tokenizer)

        # 批量测试
        test_batch_questions(trained_model, trained_tokenizer)
    else:
        print("训练失败，无法进行测试")