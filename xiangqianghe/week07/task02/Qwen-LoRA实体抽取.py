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
    """加载和预处理数据 - 优化版本"""
    try:
        # 从CSV文件加载数据
        train_data = pd.read_csv('./intent-dataset.csv', sep='\t', header=None)

        # 重命名列并添加输入列
        train_data["input"] = ""
        train_data.columns = ["instruction", "output", "input"]

        # 转换为Hugging Face Dataset
        dataset = Dataset.from_pandas(train_data)
        print(f"数据加载成功，共 {len(dataset)} 条样本")
        return dataset

    except Exception as e:
        print(f"数据加载失败: {e}")
        # 备用示例数据
        sample_data = [
            {
                "instruction": "你是一个文本实体识别专家，需要从给定句子中提取实体。",
                "input": "北京时间今天上午，苹果公司在加州库比蒂诺发布了新款iPhone。",
                "output": '{"entity_text": "北京时间", "entity_label": "TIME"}{"entity_text": "苹果公司", "entity_label": "ORG"}{"entity_text": "加州库比蒂诺", "entity_label": "LOC"}'
            },
            {
                "instruction": "你是一个文本实体识别专家，需要从给定句子中提取实体。",
                "input": "马云于1999年在杭州创立了阿里巴巴集团。",
                "output": '{"entity_text": "马云", "entity_label": "PER"}{"entity_text": "1999年", "entity_label": "TIME"}{"entity_text": "杭州", "entity_label": "LOC"}{"entity_text": "阿里巴巴集团", "entity_label": "ORG"}'
            }
        ]
        df = pd.DataFrame(sample_data)
        return Dataset.from_pandas(df)


def initialize_model_and_tokenizer(model_path):
    """初始化模型和tokenizer - 优化版本"""
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
            torch_dtype = torch.bfloat16  # 使用BF16精度
            print("使用GPU设备，启用BF16精度")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("使用CPU设备，使用FP32精度")

        # 加载模型 - 启用梯度检查点节省内存
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_cache=False,  # 禁用缓存以启用梯度检查点
        )

        # 启用梯度检查点
        model.gradient_checkpointing_enable()
        print("启用梯度检查点，节省显存")

        # 启用梯度计算
        model.enable_input_require_grads()

        print("模型和分词器初始化成功")
        return tokenizer, model

    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise


def process_func(example, tokenizer, max_length=512):
    """数据处理函数 - 优化版本"""
    # 构建指令格式
    instruction_text = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n" \
                       f"<|im_start|>user\n{example['input']}<|im_end|>\n" \
                       f"<|im_start|>assistant\n"

    response_text = example['output']

    # tokenize
    instruction = tokenizer(
        instruction_text,
        add_special_tokens=False,
        return_tensors=None
    )
    response = tokenizer(
        response_text,
        add_special_tokens=False,
        return_tensors=None
    )

    # 拼接输入和输出
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 标签处理（只计算response部分的loss）
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


def setup_lora(model, r=8, lora_alpha=16, lora_dropout=0.05):  # 减少参数加速训练
    """设置LoRA配置 - 优化版本"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 减少目标模块
        inference_mode=False,
        r=r,  # 降低秩
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_training_args():
    """配置训练参数 - 优化版本"""
    # 根据GPU可用性调整批大小和梯度累积步数
    if torch.cuda.is_available():
        per_device_batch_size = 8  # 增加批大小
        gradient_accumulation_steps = 2  # 减少梯度累积步数
        fp16 = False
        bf16 = True  # 优先使用BF16
        optim = "adamw_torch_fused"  # 使用融合优化器
        print("启用BF16混合精度训练并使用融合优化器")
    else:
        per_device_batch_size = 4
        gradient_accumulation_steps = 4
        fp16 = False
        bf16 = False
        optim = "adamw_torch"  # 使用非融合优化器
        print("使用FP32精度训练并使用非融合优化器")

    training_args = TrainingArguments(
        output_dir="./models/qwen3-0.6B-ner-output",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=2,  # 减少训练轮数
        learning_rate=3e-4,  # 提高学习率
        logging_steps=100,  # 减少日志频率
        save_steps=1000,  # 减少保存频率
        save_total_limit=1,  # 减少保存的检查点
        remove_unused_columns=True,  # 删除未使用列节省内存
        warmup_ratio=0.05,  # 减少预热比例
        fp16=fp16,
        bf16=bf16,
        report_to=None,
        eval_strategy="no",  # 禁用评估以加速
        load_best_model_at_end=False,  # 禁用最佳模型加载
        optim=optim,  # 动态选择优化器
        gradient_checkpointing=True,  # 启用梯度检查点
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,  # 使用多进程加载数据
    )
    return training_args


def predict_intent(model, tokenizer, text, max_new_tokens=512):
    """预测单个文本的意图"""
    # 构建提示词
    messages = [
        {"role": "system", "content": "你是一个文本实体识别专家，从文本中提取实体并以JSON格式输出。"},
        {"role": "user", "content": f"请从以下文本中提取实体：{text}"}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)

    # 生成输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def batch_predict_entities(model, tokenizer, test_texts, batch_size=8):  # 增加批大小
    """批量预测实体识别结果"""
    results = []

    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(test_texts), batch_size), desc="批量预测实体"):
        batch_texts = test_texts[i:i + batch_size]
        batch_results = []

        # 批量处理
        try:
            # 构建批量提示词
            messages_batch = []
            for text in batch_texts:
                messages = [
                    {"role": "system", "content": "你是一个文本实体识别专家，从文本中提取实体并以JSON格式输出。"},
                    {"role": "user", "content": f"请从以下文本中提取实体：{text}"}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                messages_batch.append(formatted_text)

            # 批量编码
            model_inputs = tokenizer(
                messages_batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            # 批量生成
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 批量解码
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            batch_results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"批量预测出错: {e}")
            batch_results = ["预测出错"] * len(batch_texts)

        results.extend(batch_results)

    return results


def main():
    """主执行函数 - 优化版本"""
    try:
        # 1. 加载数据
        print("步骤1: 加载数据...")
        ds = load_and_preprocess_data()

        # 检查数据量
        data_size = len(ds)
        print(f"数据集大小: {data_size} 条样本")

        if data_size > 10000:
            # 使用数据子集加速训练
            print(f"数据量较大({data_size})，使用10%的子集加速训练...")
            ds = ds.select(range(int(data_size * 0.1)))
            data_size = len(ds)
            print(f"使用子集大小: {data_size} 条样本")

        # 2. 初始化模型和tokenizer
        print("步骤2: 初始化模型和tokenizer...")
        model_path = "./models/Qwen/Qwen3-0.6B"  # 与下载路径一致
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 3. 处理数据
        print("步骤3: 处理训练数据...")
        process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

        # 使用缓存加速
        cache_file = "./data_cache.arrow"
        if os.path.exists(cache_file):
            print("加载缓存数据...")
            tokenized_ds = Dataset.load_from_disk(cache_file)
        else:
            print("预处理数据并创建缓存...")
            tokenized_ds = ds.map(
                process_func_with_tokenizer,
                remove_columns=ds.column_names,
                num_proc=4  # 使用多进程
            )
            tokenized_ds.save_to_disk(cache_file)
            print("数据缓存已保存")

        # 4. 划分训练集和验证集
        print("步骤4: 划分训练集和验证集...")
        # 简化划分 - 使用全部数据训练
        train_ds = tokenized_ds
        eval_ds = None

        print(f"训练集: {len(train_ds)} 条, 验证集: 0 条 (禁用评估以加速)")

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

        # 训练模型
        trainer.train()

        # 8. 保存模型
        print("步骤8: 保存模型...")
        os.makedirs("./models/output_Qwen3-0.6B-ner", exist_ok=True)
        trainer.save_model("./models/output_Qwen3-0.6B-ner")
        tokenizer.save_pretrained("./models/output_Qwen3-0.6B-ner")

        print("训练完成!")
        return model, tokenizer

    except Exception as e:
        print(f"主函数执行出错: {e}")
        return None, None


def test_single_example():
    """测试单个示例"""
    try:
        model_path = "./models/Qwen/Qwen3-0.6B/"
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        # 尝试加载训练好的LoRA权重
        try:
            model.load_adapter("./models/output_Qwen3-0.6B-ner")
            print("成功加载训练好的LoRA权重")
        except:
            print("未找到训练好的权重，使用原始模型")

        # 测试预测
        test_text = "去北京如何走？"
        result = predict_intent(model, tokenizer, test_text)
        print(f"输入: {test_text}")
        print(f"实体识别结果: {result}")

    except Exception as e:
        print(f"单例测试出错: {e}")


def test_batch_examples(model, tokenizer):
    """测试批量示例"""
    try:
        test_texts = [
            "李华明天将在北京清华大学参加会议。",
            "北京时间今天上午，苹果公司在加州库比蒂诺发布了新款iPhone。",
            "马云于1999年在杭州创立了阿里巴巴集团。",
            "明天下午三点在上海浦东机场有一场重要的会议。",
            "张三和李四计划下周去纽约旅游。"
        ]

        print("\n=== 批量预测测试 ===")
        results = batch_predict_entities(model, tokenizer, test_texts, batch_size=2)

        # 打印结果
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"\n样本 {i + 1}:")
            print(f"输入文本: {text}")
            print(f"实体识别结果: {result}")

    except Exception as e:
        print(f"批量测试出错: {e}")


if __name__ == "__main__":
    # 执行主函数
    model, tokenizer = main()

    if model is not None:
        # 单独测试
        test_single_example()

        # 批量测试
        test_batch_examples(model, tokenizer)
    else:
        print("训练失败，无法进行测试")