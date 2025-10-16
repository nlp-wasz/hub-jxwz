# 使用 LoRA 对Qwen等LLM进行微调（步骤和使用BERT预训练模型几乎一致）
# 1.读取数据  2.构建Dataset数据集，分词处理  3.构建LoRA配置对象，对模型进行微调（指定层添加低秩矩阵）
# 4.构建模型训练参数，训练模型  5.预测
import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import pandas as pd
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

# 1.读取数据
data = pd.read_csv("../data/intent-dataset.csv", header=None, sep="\t")
data.columns = ["intent", "output"]

train_data = data[:100]
eval_data = data[-100:]

# 2.构建Dataset数据集，分词处理
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# 分词处理（将 intent 和 output 转换成LLM能够识别的格式，让分词器处理 得到LLM可以训练的inputs）
# use_fast：是否使用升级版Tokenizer（LLM一般都有自己的分词器，不支持Fast）
# trust_remote_code：是否允许执行外部代码（LLM一般自带一些程序代码，需要额外执行）
llm_tokenizer = AutoTokenizer.from_pretrained('../../models/Qwen/Qwen3-0.6B', use_fast=False, trust_remote_code=True)


def get_token_result(dataset):
    # 获取 每个样本的 intent 和 output
    intent = dataset["intent"]
    output = dataset["output"]

    # 根据 加载的 LLM 分词器，将intent转换为能够识别的格式
    message = [
        {"role": "system",
         "content": "你是一个意图识别专家，根据用户的输入识别用户的意图，除了意图之外的其它任何信息都不要输出。"},
        {"role": "user", "content": intent},
        {"role": "assistant"}
    ]

    format_intent = llm_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)

    # 对格式化后的 format_intent 和 output 都进行tokenize分词处理（然后拼接在一起）
    # add_special_tokens：是否天剑特殊标记token（format_intent已经包含特殊标记token，无需添加）
    intent_tokenize = llm_tokenizer(format_intent, add_special_tokens=False)
    output_tokenize = llm_tokenizer(output)

    input_ids = intent_tokenize["input_ids"] + output_tokenize["input_ids"] + [llm_tokenizer.eos_token_id]
    attention_mask = intent_tokenize["attention_mask"] + output_tokenize["attention_mask"] + [
        llm_tokenizer.eos_token_id]

    # 意图部分不参与计算损失值等，只对需要输出的 output 部分进行预测
    labels = [-100] * len(intent_tokenize["input_ids"]) + output_tokenize["input_ids"] + [llm_tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


train_dataset = train_dataset.map(get_token_result, batched=False, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(get_token_result, batched=False, remove_columns=eval_dataset.column_names)

# 3.构建 LoRa 配置对象，对模型进行微调
# device_map：使用的设备  torch_dtype=torch.float16：使用半精度节省内存（训练速度会降低）
llm_model = AutoModelForCausalLM.from_pretrained(
    "../../models/Qwen/Qwen3-0.6B", device_map="auto", torch_dtype=torch.float16)
llm_model.enable_input_require_grads()  # 支持节点计算梯度，与 gradient_checkpointing 参数配合使用

# LoRa 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 微调的模型类型（LLM）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 添加低秩矩阵的层
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
# peft 模型微调
lora_llm_model = get_peft_model(llm_model, lora_config)

# 4.构建模型训练参数，训练模型
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    warmup_ratio=0.1,

    output_dir="../data/lora_qwen_train",
    logging_dir="../data/logs",
    logging_steps=50,

    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    do_eval=True,
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True
)

trainer = Trainer(
    model=lora_llm_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=llm_tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=llm_tokenizer,
        padding=True,
        pad_to_multiple_of=8  # 样本对齐格式（GPU 对齐）
    )
)

print("模型训练开始")
# trainer.train()
print("模型训练完毕")

print("模型评估开始")
# eval_results = trainer.evaluate()
# print(eval_results)
print("模型评估完毕")

# 5.预测
# 加载 LoRA微调后的 模型
lora_tokenizer = AutoTokenizer.from_pretrained('../../models/Qwen/Qwen3-0.6B', use_fast=False,
                                               trust_remote_code=True)
lora_model = AutoModelForCausalLM.from_pretrained('../../models/Qwen/Qwen3-0.6B', device_map="auto",
                                                  torch_dtype=torch.float16)

# 加载 微调后的权重等信息
peft_lora_model = PeftModel.from_pretrained(lora_model, "../data/lora_qwen_train/checkpoint-21")
peft_lora_model.cpu()

# 预测
for pre_data in tqdm.tqdm(data[:3].iterrows(), desc="意图识别"):
    pre_intent = pre_data[1][0]
    pre_output = pre_data[1][1]

    # 构建符合格式的 format_intent
    message = [
        {"role": "system",
         "content": "你是一个意图识别专家，根据用户的输入识别用户的意图，除了意图之外的其它任何信息都不要输出。"},
        {"role": "user", "content": pre_intent},
        {"role": "assistant"}
    ]

    format_intent = lora_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    pre_intent_tokenize = lora_tokenizer(format_intent, add_special_tokens=False, return_tensors="pt")

    # 预测
    with torch.no_grad():
        pre_outputs = peft_lora_model.generate(
            **pre_intent_tokenize,
            max_new_tokens=512,
            do_sample=True,  # 采样
            temperature=0.1,
            pad_token_id=lora_tokenizer.pad_token_id,
            eos_token_id=lora_tokenizer.eos_token_id
        )

    pre_res = lora_tokenizer.batch_decode(pre_outputs[:, pre_intent_tokenize["input_ids"].shape[1]:],
                                          skip_special_tokens=True)

    print(pre_res)
