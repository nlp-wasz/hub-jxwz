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
import codecs

# 在 Qwen-LoRA微调-知识问答.py 中添加数据处理函数
def convert_ner_data_to_instruction_format():
    """将NER数据转换为指令微调格式"""
    # 加载训练数据
    train_lines = codecs.open('./msra/train/sentences.txt').readlines()[:1000]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]

    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]

    # 定义标签类型
    tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

    # 转换为指令格式
    train_data = []
    for text, tags in zip(train_lines, train_tags):
        instruction = f"请识别以下文本中的实体：{text}"
        # 将标签转换为可读格式
        entities = convert_tags_to_entities(text, tags)
        output = format_entities_output(entities)
        train_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    val_data = []
    for text, tags in zip(val_lines, val_tags):
        instruction = f"请识别以下文本中的实体：{text}"
        entities = convert_tags_to_entities(text, tags)
        output = format_entities_output(entities)
        val_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    return train_data, val_data


def convert_tags_to_entities(text, tags):
    """将BIO标签转换为实体列表"""
    entities = []
    current_entity = ""
    current_type = ""

    for i, (char, tag) in enumerate(zip(text, tags)):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            current_entity += char
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""
            if tag.startswith('B-'):
                current_entity = char
                current_type = tag[2:]

    if current_entity:
        entities.append((current_entity, current_type))

    return entities


def format_entities_output(entities):
    """格式化实体输出"""
    if not entities:
        return "未识别到实体"

    entity_groups = {}
    for entity, entity_type in entities:
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)

    output_parts = []
    for entity_type, entity_list in entity_groups.items():
        output_parts.append(f"{entity_type}: {', '.join(entity_list)}")

    return "; ".join(output_parts)


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理NER数据"""
    # 转换数据格式
    train_data, val_data = convert_ner_data_to_instruction_format()

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
    # 构建指令部分 - 修改为实体识别任务
    instruction_text = f"<|im_start|>system\n你是一个实体识别助手，请识别文本中的实体并按照指定格式输出<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
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
        output_dir="./output_Qwen1.5_entities",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 修改预测函数以适应实体识别任务
def predict_entities(model, tokenizer, text, device='cpu'):
    """预测文本中的实体"""
    messages = [
        {"role": "system", "content": "你是一个实体识别助手，请识别文本中的实体并按照指定格式输出"},
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


# 测试函数
def test_single_example():
    model_path = "/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week07/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week07/output_Qwen1.5_entities/checkpoint-45")
    model.cpu()

    # 测试实体识别
    test_texts = [
        "山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报"
    ]

    for text in test_texts:
        result = predict_entities(model, tokenizer, text)
        print(f"输入: {text}")
        print(f"识别结果: {result}")
        print("-" * 50)

if __name__ == "__main__":
    # 执行主函数
    # result_df = main()

    # 单独测试
    test_single_example()