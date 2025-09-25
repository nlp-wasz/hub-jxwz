import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset

# LoRA
from peft import LoraConfig, TaskType, get_peft_model


# 加载数据（保持与原脚本一致）
train = json.load(open('./cmrc2018_public/train.json', 'r', encoding='utf-8'))
dev = json.load(open('./cmrc2018_public/dev.json', 'r', encoding='utf-8'))


# 准备训练数据（保持结构一致）
def prepare_dataset(data):
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append({
                'answer_start': [qa['answers'][0]['answer_start']],
                'text': [qa['answers'][0]['text']]
            })

    return paragraphs, questions, answers


# 准备训练和验证数据
train_paragraphs, train_questions, train_answers = prepare_dataset(train)
val_paragraphs, val_questions, val_answers = prepare_dataset(dev)

train_dataset_dict = {
    'context': train_paragraphs[:1000],
    'question': train_questions[:1000],
    'answers': train_answers[:1000]
}

val_dataset_dict = {
    'context': val_paragraphs[:100],
    'question': val_questions[:100],
    'answers': val_answers[:100]
}

train_dataset = Dataset.from_dict(train_dataset_dict)
val_dataset = Dataset.from_dict(val_dataset_dict)


# 初始化tokenizer和模型（最小修改：改为Qwen + CausalLM）
model_path = "../models/Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)


# 应用LoRA（最小增量）
def setup_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    return model

model = setup_lora(model)


# 预处理函数（最小修改：构造成chat模板，并mask掉prompt）
def preprocess_function(examples):
    prompts = []
    responses = []

    for q, c, a in zip(examples["question"], examples["context"], examples["answers"]):
        gold = a['text'][0]
        messages = [
            {"role": "system", "content": "你是一个中文机器阅读理解和抽取式问答助手。"},
            {"role": "user", "content": f"请阅读给定的段落并回答问题，且只输出答案文本，不要解释。\n\n段落：{c}\n\n问题：{q}"},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
        responses.append(gold)

    model_inputs = tokenizer(prompts, add_special_tokens=False)
    with tokenizer.as_target_tokenizer():
        labels_inputs = tokenizer(responses, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    labels = []

    for inp_ids, attn, lab_ids in zip(model_inputs["input_ids"], model_inputs["attention_mask"], labels_inputs["input_ids"]):
        seq_input_ids = inp_ids + lab_ids + [tokenizer.eos_token_id]
        seq_attn = attn + [1] * (len(lab_ids) + 1)
        seq_labels = [-100] * len(inp_ids) + lab_ids + [tokenizer.eos_token_id]

        input_ids.append(seq_input_ids)
        attention_mask.append(seq_attn)
        labels.append(seq_labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)


# 训练参数（尽量保留原结构）
training_args = TrainingArguments(
    output_dir="./qa-qwen-lora-model",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

print("开始训练Qwen-LoRA QA模型...")
trainer.train()

print("保存LoRA适配器和tokenizer...")
trainer.save_model()
tokenizer.save_pretrained('./qa-qwen-lora-model')


# 生成式预测（最小修改：用chat模板生成，截掉输入部分）
def predict(context, question, device='cpu'):
    model.to(device)
    messages = [
        {"role": "system", "content": "你是一个中文机器阅读理解和抽取式问答助手。"},
        {"role": "user", "content": f"请阅读给定的段落并回答问题，且只输出答案文本，不要解释。\n\n段落：{context}\n\n问题：{question}"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response.strip()


# 在验证集上测试几个样本
print("\n在验证集上测试:")
for i in range(min(3, len(val_paragraphs))):
    context = val_paragraphs[i]
    question = val_questions[i]
    expected_answer = val_answers[i]['text'][0]
    predicted_answer = predict(context, question, device='cpu')
    print(f"问题 {i + 1}: {question}")
    print(f"预期答案: {expected_answer}")
    print(f"预测答案: {predicted_answer}")
    print(f"匹配: {expected_answer == predicted_answer}")
    print()


