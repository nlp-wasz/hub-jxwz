# 使用 BERT 预训练模型，实现 QA 知识问答任务（并非生成式，而是抽取式，根据content抽取问题答案）
# 与BERT其它任务类型的步骤差不多，主要区别在分词器和模型的选择，以及dataset预分词处理逻不同
# 步骤：1.读取数据  2.构建dataset数据集，以及分词处理  3.构建模型参数，训练模型  4.预测

import json
from datasets import Dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast, TrainingArguments, Trainer, DefaultDataCollator
import torch
from tqdm import tqdm

# 1.读取数据
train_json = json.load(open("../data/cmrc2018_public/train.json", encoding="utf-8"))
eval_json = json.load(open("../data/cmrc2018_public/dev.json", encoding="utf-8"))


# 拆分 train_json 和 eval_json 结果，获取 content，question，answer
def split_json(json_data):
    contents, questions, answers = [], [], []
    # 循环 拆分
    for data in json_data:
        content = data["paragraphs"][0]["context"]
        qas = data["paragraphs"][0]["qas"]
        for qa in qas:
            question = qa["question"]
            answer = qa["answers"][0]

            # 记录
            contents.append(content)
            questions.append(question)
            answers.append(answer)

    return contents[:2000], questions[:2000], answers[:2000]


train_contents, train_questions, train_answers = split_json(train_json["data"])
eval_contents, eval_questions, eval_answers = split_json(eval_json["data"])

# 2.构建dataset数据集，以及分词处理逻辑
tokenizer = BertTokenizerFast.from_pretrained('../../models/google-bert/bert-base-chinese')
train_dataset = Dataset.from_dict({
    "contents": train_contents,
    "questions": train_questions,
    "answers": train_answers
})

eval_dataset = Dataset.from_dict({
    "contents": eval_contents,
    "questions": eval_questions,
    "answers": eval_answers
})


# 分词处理逻辑（分词 并且获取 start_position 和 end_position）
def get_token_result(dataset):
    contents, questions, answers = dataset["contents"], dataset["questions"], dataset["answers"]

    # 将每个问题和对应的文本内容 组成句子对，让分词器处理
    qa_tokenizer = tokenizer(
        questions,
        contents,
        padding=True,
        truncation=True,
        pad_to_multiple_of=8,
        max_length=512,

        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    # 因为 tokenizer 设置了 stride参数，因此返回的qa_tokenizer样本数量 可能会大于原始样本数量（被truncation形成了新的样本）
    # 因此需要设置每个分词处理后的样本 所对应的答案起止位置
    # sample_mapping：分词后的样本 来源于原始样本的索引位置
    # offset_mapping：每个分词后样本中的token，对应与原始样本token的起止位置
    sample_mapping = qa_tokenizer.pop("overflow_to_sample_mapping")
    offset_mapping = qa_tokenizer.pop("offset_mapping")
    start_positions = []
    end_positions = []
    for index, sentence_idx in enumerate(sample_mapping):
        # 获取 sentennce_idx原始样本中的答案
        answer_start = answers[sentence_idx]["answer_start"]
        answer_end = answer_start + len(answers[sentence_idx]["text"])

        # 如果不存在答案，start_position 和 end_position 都为0
        if answer_start is None:
            start_position = 0
            end_position = 0
        else:
            # 存在答案，找出答案在 分词处理后的样本中 对应的位置
            # 而分词样本是 句子对，因此答案只能出现在 第二句
            # sequence_ids()：token对应的句子索引，0：第一句  1：第二句
            sequence_ids = qa_tokenizer.sequence_ids(index)

            # 获取 content（文本内容） 在样本token中的 起止位置
            idx = 0
            while sequence_ids[idx] is None or sequence_ids[idx] != 1:
                idx += 1
            content_start = idx

            while sequence_ids[idx] is not None and sequence_ids[idx] != 0:
                idx += 1
            content_end = idx - 1

            # 答案只能出现在 [content_start, content_end] 范围内
            # 并且 这些位置token对应的offset_mapping，必须与 answer_start和answer_end 有重叠，否则不存在答案信息
            if offset_mapping[index][content_start][0] > answer_end or offset_mapping[index][content_end][
                1] < answer_start:
                start_position = 0
                end_position = 0
            else:
                # 在 content_start 和 content_end 之间，寻找符合条件的 position 位置索引
                idx = content_start
                while idx <= content_end and offset_mapping[index][idx][0] < answer_start:
                    idx += 1
                start_position = idx

                idx = content_end
                while idx >= content_start and offset_mapping[index][idx][1] > answer_end:
                    idx -= 1
                end_position = idx

        # 记录 起止位置
        start_positions.append(start_position)
        end_positions.append(end_position)

    # 将 start_positions 和 end_positions 添加到 qa_tokenizer
    qa_tokenizer["start_positions"] = start_positions
    qa_tokenizer["end_positions"] = end_positions

    return qa_tokenizer


train_dataset = train_dataset.map(get_token_result, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(get_token_result, batched=True, remove_columns=eval_dataset.column_names)

# 3.构建训练参数，训练模型
bert_qa_model = BertForQuestionAnswering.from_pretrained("../../models/google-bert/bert-base-chinese")
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    # no_cuda=True,

    output_dir="./data/qa-train",
    logging_dir="./data/logs",
    logging_steps=50,
    report_to="none",

    save_strategy="epoch",
    eval_strategy="epoch",
    do_eval=True,
    save_total_limit=1,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=bert_qa_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator()
)

print("模型训练开始")
# trainer.train()
print("模型训练完毕")

print("模型评估开始")
# eval_results = trainer.evaluate()
# print(eval_results)
print("模型评估完毕")

# 4.预测
# tokenizer = BertTokenizerFast.from_pretrained('./data/qa-train/checkpoint-338')
# bert_qa_model = BertForQuestionAnswering.from_pretrained('./data/qa-train/checkpoint-338')
bert_qa_model.to("cpu")
for i, qc in enumerate(tqdm(zip(eval_questions[:10], eval_contents[:10]), desc="知识问答")):
    # 获取 content 和 question
    pre_content = qc[1]
    pre_question = qc[0]

    # 分词器处理
    pre_tokenize = tokenizer(pre_question, pre_content, padding=True, truncation=True, max_length=512,
                             return_tensors="pt")

    # 禁用梯度计算，模型预测
    with torch.no_grad():
        pre_outputs = bert_qa_model(**pre_tokenize)

        start_logits = pre_outputs["start_logits"]
        end_logits = pre_outputs["end_logits"]

        # 起止位置 -> (1, token_size)
        start_position = start_logits.argmax(dim=-1).item()
        end_position = end_logits.argmax(dim=-1).item()
        if start_position > end_position:
            # 起始位置 > 结束位置（不符合条件，手动处理 获取最优 起止position）
            # 首先需要得到 content对应token的起止范围（答案只能出现在content中）
            sequence_ids = pre_tokenize.sequence_ids(0)
            idx = 0
            while sequence_ids[idx] is None or sequence_ids[idx] != 1:
                idx += 1
            content_start = idx

            while sequence_ids[idx] is not None and sequence_ids[idx] != 0:
                idx += 1
            content_end = idx - 1

            # 在 content_start 和 content_end 范围内，查找最优起止position
            # min(start_idx + 30, end_position + 1)：限制 起止position 距离
            max_score = float("-inf")
            for start_idx in range(content_start, content_end + 1):
                # for end_idx in range(start_idx, min(start_idx + 50, content_end + 1)):
                for end_idx in range(start_idx, content_end + 1):
                    # 最优 position 条件
                    score = start_logits[0][start_idx].item() + end_logits[0][end_idx].item()
                    if score > max_score:
                        start_position = start_idx
                        end_position = end_idx
                        max_score = score

        # 根据 start_position 和 end_position，获取对应位置的token
        predicted_answer = tokenizer.decode(pre_tokenize.input_ids[0][start_position:end_position + 1],
                                            skip_special_tokens=True)
        predicted_answer = predicted_answer.replace(" ", "")

        print(f"问题 {i + 1}: {pre_question}")
        print(f"预期答案: {eval_answers[i]['text']}")
        print(f"预测答案: {predicted_answer}")
        # print(f"匹配: {eval_answers == predicted_answer}")
