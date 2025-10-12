# 通过 LoRA 微调Qwen大模型，完成QA知识问答任务
import json
import re

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DefaultDataCollator
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch
from tqdm import tqdm


# 0.加载 Tokenizer 和 Model
def get_tokenizer_model(path):
    qwen_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16)
    # qwen_model.enable_input_require_grads()

    return qwen_tokenizer, qwen_model


# 1.读取数据
def get_content_qas(datas):
    # 循环 获取所有问题
    contents, questions, answers = [], [], []

    for data in datas:
        content = data["paragraphs"][0]["context"]
        for qa in data["paragraphs"][0]["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]

            # 记录
            contents.append(content)
            questions.append(question)
            answers.append(answer)

    return contents[:100], questions[:100], answers[:100]


def read_data(path):
    json_data = json.load(open(path, encoding="utf-8"))
    datas = json_data["data"]

    # 提取 content  question  answer
    contents, questions, answers = get_content_qas(datas)

    return contents, questions, answers


# 2.构建Dataset数据集，分词处理
def get_dataset(contents, questions, answers):
    return Dataset.from_dict({
        "contents": contents,
        "questions": questions,
        "answers": answers,
    })


# 分词处理逻辑
def get_tokenizer_result(dataset, qwen_tokenizer, max_length=256):
    # 获取 每一个样本的 content  question  answer
    content = dataset["contents"][:128]
    question = dataset["questions"]
    answer_text = dataset["answers"]["text"]
    answer_start = dataset["answers"]["answer_start"]

    # 起止索引位置
    position = answer_start + len(answer_text)

    # 构建 llm prompt格式
    prompt = [
        {"role": "system", "content": "你是一个知识问答专家，请从用户输入的文本中提取答案，并返回起止索引位置。"},
        {"role": "user", "content": f"问题：{question}  内容：{content}"},
        {"role": "assistant", "content": f"答案：{answer_text} 起止索引位置：f{answer_start}-f{position}"}
    ]

    # LLM tokenizer 转换格式
    prompt_format = qwen_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    prompt_tokenizer = qwen_tokenizer(prompt_format, truncation=True, max_length=max_length, add_special_tokens=False)

    # 不包含 assistant 的格式化 prompt（用于 label masking）
    no_assistant_format = qwen_tokenizer.apply_chat_template(prompt[:-1], tokenize=False, add_generation_prompt=False)
    no_assistant_tokenizer = qwen_tokenizer(no_assistant_format, truncation=True, max_length=max_length,
                                            add_special_tokens=False)

    # 组合成 完成的输入特征向量
    no_assistant_len = len(no_assistant_tokenizer["input_ids"])
    labels = [-100] * no_assistant_len + prompt_tokenizer["input_ids"][no_assistant_len:]

    # 手动 padding 补齐
    input_ids = prompt_tokenizer["input_ids"]
    attention_mask = prompt_tokenizer["attention_mask"]

    if len(input_ids) < max_length:
        # 获取 tokenizer 填充字符编码
        pad_token_id = qwen_tokenizer.pad_token_id
        # 需要补齐的 长度
        padding_len = max_length - len(input_ids)

        # padding 补齐
        input_ids += [pad_token_id] * padding_len
        attention_mask += [0] * padding_len
        labels += [-100] * padding_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 3.构建 LoRa 配置对象
# LoRA 配置
def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1
    )


# 4.构建 模型训练参数，训练模型
def get_training_args():
    return TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,

        output_dir="./data/lora-qa",
        logging_dir="./data/logs",
        logging_steps=10,
        report_to="none",

        save_strategy="epoch",
        eval_strategy="epoch",
        do_eval=True,
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # gradient_checkpointing=True,
        fp16=True
    )


# 5.预测
def predict_func(epoch, content, question, answer, qwen_tokenizer, peft_qwen_model):
    answer_text = answer["text"]
    answer_start = answer["answer_start"]
    # 起止索引位置
    position = answer_start + len(answer_text)

    # 构建 llm prompt格式
    prompt = [
        {"role": "system",
         "content": "你是一个知识问答专家，请从用户输入的文本中提取答案，并返回起止索引位置，输出格式：【xxx】！"},
        {"role": "user", "content": f"问题：{question}  内容：{content}"}
    ]

    # LLM tokenizer 转换格式
    prompt_format = qwen_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_tokenizer = qwen_tokenizer(prompt_format, truncation=True, max_length=512, add_special_tokens=False,
                                      return_tensors="pt").to("cuda")

    # 预测
    with torch.no_grad():
        pre_outputs = peft_qwen_model.generate(
            **prompt_tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=qwen_tokenizer.pad_token_id,
            eos_token_id=qwen_tokenizer.eos_token_id
        )

    # 获取 生成的token
    genetate_len = prompt_tokenizer["input_ids"].shape[1]
    predicted_answer = qwen_tokenizer.batch_decode(pre_outputs[:, genetate_len:], skip_special_tokens=False)

    predicted_answer_re = re.findall(r'【(.*?)】', predicted_answer[0])
    print(f"问题 {epoch + 1}: {question}")
    print(f"预期答案: {answer_text}")
    # for idx, ans in enumerate(predicted_answer_re):
    #     print(f"预测答案 {idx + 1}: {ans}")
    print(f"预测答案: {predicted_answer}")


# LoRA 微调模型，并进行训练
def train_func():
    # 0.加载 Tokenizer 和 Model
    qwen_tokenizer, qwen_model = get_tokenizer_model("../../models/Qwen/Qwen3-0.6B")

    # 1.读取数据
    train_contents, train_questions, train_answers = read_data("../data/cmrc2018_public/train.json")
    eval_contents, eval_questions, eval_answers = read_data("../data/cmrc2018_public/dev.json")

    # 2.构建Dataset数据集，分词处理
    train_dataset = get_dataset(train_contents, train_questions, train_answers)
    eval_dataset = get_dataset(eval_contents, eval_questions, eval_answers)

    # 分词处理逻辑
    train_dataset = train_dataset.map(lambda dataset: get_tokenizer_result(dataset, qwen_tokenizer), batched=False,
                                      remove_columns=train_dataset.column_names, load_from_cache_file=True)
    eval_dataset = eval_dataset.map(lambda dataset: get_tokenizer_result(dataset, qwen_tokenizer), batched=False,
                                    remove_columns=eval_dataset.column_names, load_from_cache_file=True)

    # 3.构建 LoRa 配置对象
    # LoRA 配置
    lora_config = get_lora_config()
    peft_qwen_model = get_peft_model(qwen_model, lora_config)

    # 4.构建 模型训练参数，训练模型
    training_args = get_training_args()
    trainer = Trainer(
        model=peft_qwen_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=qwen_tokenizer,
        data_collator=DefaultDataCollator()
    )

    print("模型训练开始")
    trainer.train()
    # print("Model device:", next(qwen_model.parameters()).device)
    # print("GPU memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
    print("模型训练完毕")

    # print("模型评估开始")
    # eval_results = trainer.evaluate()
    # print(eval_results)
    # print("模型评估完毕")


# 预测
def pre_func():
    # 0.加载 LoRA 微调后的 Tokenizer 和 Model
    qwen_tokenizer, qwen_model = get_tokenizer_model("../../models/Qwen/Qwen3-0.6B")
    peft_qwen_model = PeftModel.from_pretrained(qwen_model, './data/lora-qa/checkpoint-21')
    peft_qwen_model.cuda()

    # 读取 评估 数据（测试用）
    eval_contents, eval_questions, eval_answers = read_data("../data/cmrc2018_public/dev.json")
    eval_contents, eval_questions, eval_answers = eval_contents[:3], eval_questions[:3], eval_answers[:3]
    for epoch, (content, question, answer) in enumerate(tqdm(zip(eval_contents, eval_questions, eval_answers))):
        # 预测方法
        predict_func(epoch, content, question, answer, qwen_tokenizer, peft_qwen_model)


if __name__ == '__main__':
    # LoRA 微调模型，并进行训练
    # train_func()

    # 预测
    pre_func()
