# 模型训练，评估，预测 模块
import json

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import BertTokenizer
from seqeval.metrics.sequence_labeling import get_entities

from Arguments import Arguments
from Preprocess import DatasetPreprocess, BertDataset, BertModelForClsAndNer


class Train(object):
    def train(self, args, train_model, loss_func, optimizer, train_bert_dataloader):
        # 模型训练
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_model.train()
        train_model.to(args.device)

        for epoch in range(args.epochs):
            for batch, inputs in enumerate(train_bert_dataloader):
                # 将 inputs 中的 数据转移到 device中
                inputs_device = {key: value.to(args.device) for key, value in inputs.items()}
                cls_intent_output, ner_slot_output = train_model(inputs_device)

                # 分别计算两个任务的 损失值
                # ①cls任务：intent意图识别
                seq_intent_labels = inputs_device["seq_intent_labels"].view(-1)
                loss_intent = loss_func(cls_intent_output, seq_intent_labels)

                # ②ner任务：slot实体识别
                seq_token_labels = inputs_device["seq_token_labels"]
                seq_token_labels_view = seq_token_labels.view(-1)
                ner_slot_output_view = ner_slot_output.view(-1, ner_slot_output.size(-1))

                # seq_token_labels_view 和 ner_slot_output_view 有些token是无意义的token，不计算损失值
                attention_mask = inputs_device["attention_mask"].view(-1) == 1

                # 根据 attention_mask 对相应位置的 seq_token_labels_view和ner_slot_output_view，去除掉
                seq_token_labels_mask = seq_token_labels_view[attention_mask]
                ner_slot_output_mask = ner_slot_output_view[attention_mask]

                loss_token = loss_func(ner_slot_output_mask, seq_token_labels_mask)

                # 损失值相加
                loss = loss_intent + loss_token
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"第 {epoch + 1} 循环, 第 {batch + 1}批次, loss_intent: {loss_intent.item():.4f}, loss_token: {loss_token.item():.4f}, loss: {loss.item():.4f}")

        # 训练完之后 保存权重 or 保存模型
        # if self.args.is_save_model:
        #     # save_pretrained是Hugging Face库中的方法，自定义的模型继承自torch.Module，不具有save_pretrained方法
        #     self.train_model.save_pretrained(self.args.train_model_dir)
        #
        #     # self.args.train_model_dir 必须是一个文件，不能是文件夹
        #     torch.save(self.train_model, self.args.train_model_dir)

        if args.is_save_model_state:
            torch.save(train_model.state_dict(), args.train_model_state_dir)

    def eval(self, args, train_model, eval_bert_dataloader):
        # 模型评估
        train_model.eval()
        train_model.to(args.device)

        cls_intent_metric_pre = []
        cls_intent_metric_true = []
        ner_slot_metric_pre = []
        ner_slot_metric_true = []
        with torch.no_grad():
            epoch_all = args.epochs * len(eval_bert_dataloader)
            for epoch in range(args.epochs):
                for batch, inputs in enumerate(eval_bert_dataloader):
                    inputs_device = {key: value.to(args.device) for key, value in inputs.items()}

                    # 获取 参数信息
                    attention_masks = inputs_device["attention_mask"].detach().cpu().tolist()
                    seq_intent_labels = inputs_device["seq_intent_labels"].view(-1)
                    seq_token_labels = inputs_device["seq_token_labels"].detach().cpu().tolist()

                    cls_intent_output, ner_slot_output = train_model(inputs_device)
                    # 记录 结果，用于评估
                    # ①cls分类任务，intent类别
                    cls_intent_eval = cls_intent_output.argmax(dim=-1)
                    cls_intent_metric_pre.extend(cls_intent_eval.detach().cpu().tolist())
                    cls_intent_metric_true.extend(seq_intent_labels.detach().cpu().tolist())

                    # ②ner识别任务，token分类
                    ner_slot_evals = ner_slot_output.argmax(dim=-1).detach().cpu().tolist()
                    # 将ner转换为 BIO类型（因为seqeval的评估，需要B- I-类型）
                    for ner_slot_eval, seq_token_label, attention_mask in zip(ner_slot_evals, seq_token_labels,
                                                                              attention_masks):
                        # 计算 有效的token 有多少
                        token_size = sum(attention_mask)
                        ner_slot_eval = ner_slot_eval[1:token_size - 1]
                        seq_token_label = seq_token_label[1:token_size - 1]
                        slotBIO_evals = [args.index_to_slotBIO.get(i) for i in ner_slot_eval]
                        slotBIO_label = [args.index_to_slotBIO.get(i) for i in seq_token_label]

                        ner_slot_metric_pre.append(slotBIO_evals)
                        ner_slot_metric_true.append(slotBIO_label)

                    print(f"第 {epoch + 1} 循环，第 {(batch + 1) * (epoch + 1) + batch + 1}/{epoch_all} 批次")

        # 通过 seqeval.metric 获取指标信息
        # ①cls分类任务，intent类别
        cls_intent_metric = self.get_metric(cls_intent_metric_pre, cls_intent_metric_true, "cls")
        print(f"cls分类任务，accuracy：{cls_intent_metric["accuracy"]}")
        print(f"cls分类任务，precision：{cls_intent_metric["precision"]}")
        print(f"cls分类任务，recall：{cls_intent_metric["recall"]}")
        print(f"cls分类任务，f1：{cls_intent_metric["f1"]}")
        print(f"cls分类任务，report：")
        print(cls_intent_metric["report"])

        # ②ner识别任务，token分类
        ner_slot_metric = self.get_metric(ner_slot_metric_pre, ner_slot_metric_true, "ner")
        print(f"ner实体识别任务，accuracy：{ner_slot_metric["accuracy"]}")
        print(f"ner实体识别任务，precision：{ner_slot_metric["precision"]}")
        print(f"ner实体识别任务，recall：{ner_slot_metric["recall"]}")
        print(f"ner实体识别任务，f1：{ner_slot_metric["f1"]}")
        print(f"ner实体识别任务，report：")
        print(cls_intent_metric["report"])

    def predict(self, args, train_model, pre_texts):
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_dir)

        # 预测
        pre_result = []
        with torch.no_grad():
            for epoch, pre_text in enumerate(pre_texts):
                # 分词处理
                pre_token = tokenizer([pre_text], padding="max_length", truncation=True,
                                      max_length=args.max_seq_length,
                                      return_tensors="pt")
                pre_token = {key: value.to(args.device) for key, value in pre_token.items()}
                cls_intent_output, ner_slot_output = train_model(pre_token)

                # 提取 预测结果
                # ①cls
                cls_max_index = cls_intent_output.argmax(dim=-1).item()
                cls_intent_value = args.index_to_intent.get(cls_max_index)

                # ②ner
                ner_max_index = ner_slot_output.argmax(dim=-1).view(-1).detach().cpu().tolist()
                ner_slot_value = [args.index_to_slotBIO.get(i) for i in ner_max_index]
                ner_slot_value = ner_slot_value[1:len(pre_text) + 1]

                # 从 ner_slot_value 中提取实体
                ner_slot_value = [
                    {"slot": i[0], "content": pre_text[i[1]:i[2] + 1], "start": i[1], "end": i[2]}
                    for i in get_entities(ner_slot_value)
                ]

                print(f"预测文本：{pre_text}")
                print(f"意图：{cls_intent_value}")
                print(f"实体：")
                print(json.dumps(ner_slot_value, indent=2, ensure_ascii=False))
                print("-" * 50)

                pre_result.append({
                    "预测文本": pre_text,
                    "意图": cls_intent_value,
                    "实体": ner_slot_value
                })
                if epoch + 1 == 10:
                    break

        return pre_result

    def get_metric(self, _pre, _true, type):
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        report = None
        if type == "cls":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            accuracy = accuracy_score(_pre, _true)
            precision = precision_score(_pre, _true, average='micro')
            recall = recall_score(_pre, _true, average='micro')
            f1 = f1_score(_pre, _true, average='micro')
            report = classification_report(_pre, _true, zero_division=0)

        if type == "ner":
            from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            accuracy = accuracy_score(_pre, _true)
            precision = precision_score(_pre, _true, average='micro')
            recall = recall_score(_pre, _true, average='micro')
            f1 = f1_score(_pre, _true, average='micro')
            report = classification_report(_pre, _true, zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report
        }


# fastapi 训练方法
def train_func():
    prepreocess = DatasetPreprocess()
    args = prepreocess.args

    # 1.读取 train_process.json 数据，构建数据集
    input_examples = prepreocess.getInputExamples("./data/train_process.json", "train")

    # 2.将 input_examples 转换为 features向量形式
    input_features = prepreocess.getInputFeatures(input_examples, "train")

    # 3.构建 Dataset 和 DataLoader
    dataset = BertDataset(input_features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # 4.创建 训练模型，损失函数，优化器
    train_model = BertModelForClsAndNer()
    train_model.load_state_dict(torch.load(args.train_model_state_dir))  # 加载训练好的权重

    loss_func = CrossEntropyLoss()
    optimizer = Adam(train_model.parameters(), lr=args.lr)

    # 5.训练
    train = Train()
    train.train(args, train_model, loss_func, optimizer, dataloader)

    return "训练完成"


# fastapi 评估方法
def eval_func():
    prepreocess = DatasetPreprocess()
    args = prepreocess.args

    # 1.读取 test_process.json 数据，构建数据集
    input_examples = prepreocess.getInputExamples("./data/test_process.json", "test")

    # 2.将 input_examples 转换为 features向量形式
    input_features = prepreocess.getInputFeatures(input_examples, "test")

    # 3.构建 Dataset 和 DataLoader
    dataset = BertDataset(input_features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # 4.创建 训练模型，损失函数，优化器
    train_model = BertModelForClsAndNer()
    train_model.load_state_dict(torch.load(args.train_model_state_dir))  # 加载训练好的权重

    # 5.训练
    train = Train()
    train.eval(args, train_model, dataloader)


# fastapi 预测方法
def pre_func(pre_text):
    args = Arguments()

    # 1.创建 训练模型，损失函数，优化器
    train_model = BertModelForClsAndNer()
    train_model.load_state_dict(torch.load(args.train_model_state_dir))  # 加载训练好的权重

    # 5.训练
    train = Train()
    pre_res = train.predict(args, train_model, pre_text)

    return pre_res


if __name__ == '__main__':
    # 1.创建 Arguments 参数对象
    args = Arguments()

    # 2.获取 预处理后的 数据集
    dataset_preprocess = DatasetPreprocess()

    # 3.获取 预处理 InputExamples数据（原始数据，未向量化） 和 Input_Features（文本分词向量化，以及对应的标签也手动向量化）
    train_input_examples = dataset_preprocess.getInputExamples("./data/train_process.json", "train")
    train_input_features = dataset_preprocess.getInputFeatures(train_input_examples, "train")

    # 4.创建 BertDataset 数据集对象
    train_bert_dataset = BertDataset(train_input_features)
    train_bert_dataloader = DataLoader(train_bert_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # 5.创建 BertModelForClsAndNer 训练模型
    train_model = BertModelForClsAndNer()
    # 加载训练好的权重信息
    train_model.load_state_dict(torch.load(args.train_model_state_dir))

    # 6.创建 损失函数 和 优化器
    loss_func = CrossEntropyLoss()
    optimizer = Adam(train_model.parameters(), lr=args.lr)

    # 7.创建 模型训练，评估，预测对象
    train = Train()

    # 8.1 模型训练
    # train.train(args, train_model, loss_func, optimizer, train_bert_dataloader)

    # 8.2 模型评估
    eval_input_examples = dataset_preprocess.getInputExamples("./data/test_process.json", "test")
    eval_input_features = dataset_preprocess.getInputFeatures(eval_input_examples, "test")
    eval_bert_dataset = BertDataset(eval_input_features)
    eval_bert_dataloader = DataLoader(eval_bert_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # train.eval(args, train_model, eval_bert_dataloader)

    # 8.3 模型预测
    # 读取 test.json 文件
    with open("./data/test.json", "r", encoding="utf-8") as f:
        test_texts = json.load(f)
        pre_texts = [i["text"] for i in test_texts]

    pre_res = train.predict(args, train_model, pre_texts)
    print(json.dumps(pre_res, indent=2, ensure_ascii=False))
