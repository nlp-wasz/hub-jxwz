# 数据预处理模块（对 所有的训练集和评估集 进行预处理， 分词处理以及获取对应的标签----cls和ner）
import json
from typing import List
import torch
import re

from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, Sequential, Dropout

from Arguments import Arguments


class InputExample(object):
    def __init__(self, text, domain, intent, slot):
        self.text = text
        self.domain = domain
        self.intent = intent
        self.slot = slot


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, seq_domain_labels, seq_intent_labels,
                 seq_token_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.seq_domain_labels = seq_domain_labels
        self.seq_intent_labels = seq_intent_labels
        self.seq_token_labels = seq_token_labels


# 数据预处理
class DatasetPreprocess:
    '''
    数据预处理
    '''
    args = Arguments()

    @classmethod
    def getInputExamples(cls, input_examples_path, data_type) -> List[InputExample]:
        '''
        读取事前处理好的文件，获取里面的text，以及对应的domain、intent、slot，便于后续构建特征向量
        :param input_examples_path: 读取的文件地址
        :param data_type: 读取的数据类型（train，test）
        :return: List[InputExample]
        '''

        with open(input_examples_path, 'r', encoding='utf-8') as f:
            process_examples = json.load(f)

        # 获取每一个经过人工处理后的text，以及对应的domain、intent、slot，便于后续构建特征向量
        input_examples = []
        for example in process_examples:
            text = example['text']
            domain = example['domain']
            intent = example['intent']
            slot = example['slots']

            input_examples.append(InputExample(text, domain, intent, slot))

        return input_examples

    @classmethod
    def getInputFeatures(cls, input_examples, data_type) -> List[InputFeatures]:
        '''
        对每一个样本中的句子进行分词处理（字级别），然后获取对应的 意图和领域分类label 和 token分类label
        :param input_examples: 经过getInputExamples预处理后的 所有样本信息
        :param data_type: 读取的数据类型（train，test）
        :return: List[InputFeatures]
        '''

        tokenizer = BertTokenizer.from_pretrained(cls.args.bert_tokenizer_dir)

        # 遍历 input_examples
        input_features = []
        for example in input_examples:
            text = example.text
            domain = example.domain
            intent = example.intent
            slot = example.slot

            # 使用分词器 对 text按照字级别划分token
            text_token = tokenizer(text, max_length=cls.args.max_seq_length, padding='max_length', truncation=True)

            # domain -> "domain": "translation"
            # intent -> "intent": "TRANSLATION"
            domain_features = [cls.args.domain_to_index.get(domain)]
            intent_features = [cls.args.intent_to_index.get(intent)]

            # slot -> {"content": "智能手机","target": "英语"}  包含许多token类型
            # 获取类型对应的文本 在text中的位置，并将其转换为BIO类型
            slotBIO_list = ["O"] * len(text)
            for slot_key, slot_value in slot.items():
                # slot_key -> "content"    slot_value -> "智能手机"
                # 判断 智能手机 出现在text中的哪个位置（re模块的 findstr()方法）
                slot_index = re.finditer(slot_value, text)

                # 查询到的 slot_index 可能有多个（text中有多个slot_value）
                for match in slot_index:
                    # match -> start()，end() 等方法
                    start_index = match.start()
                    end_index = match.end()
                    # print(slotBIO_list, text, slot_value, start_index, end_index)
                    slotBIO_list[start_index] = "B-" + slot_key
                    slotBIO_list[start_index + 1:end_index] = ["I-" + slot_key] * (end_index - start_index - 1)

            slot_features = [cls.args.slotBIO_to_index.get(i) for i in slotBIO_list]
            # bert分词器在分词后 会在句子前后加上[CLS][SEP]两个特殊字符，编码都是0（这里为了对其，手动对标签添加）
            if len(slot_features) >= cls.args.max_seq_length:
                slot_features = [0] + slot_features + [0]
            else:
                slot_features = [0] + slot_features + [0] + [0] * (cls.args.max_seq_length - len(slot_features) - 2)

            input_features.append(InputFeatures(
                input_ids=torch.tensor(text_token.get('input_ids'), dtype=torch.long, requires_grad=False),
                attention_mask=torch.tensor(text_token.get('attention_mask'), dtype=torch.long, requires_grad=False),
                token_type_ids=torch.tensor(text_token.get('token_type_ids'), dtype=torch.long, requires_grad=False),
                seq_domain_labels=torch.tensor(domain_features, dtype=torch.long, requires_grad=False),
                seq_intent_labels=torch.tensor(intent_features, dtype=torch.long, requires_grad=False),
                seq_token_labels=torch.tensor(slot_features, dtype=torch.long, requires_grad=False)
            ))

        return input_features


# 构建 Dataset 数据集
class BertDataset(Dataset):
    '''
    构建 Dataset 数据集
    '''

    def __init__(self, input_features):
        super(BertDataset, self).__init__()
        self.features = input_features
        self.size = len(input_features)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": self.features[idx].input_ids,
            "attention_mask": self.features[idx].attention_mask,
            "token_type_ids": self.features[idx].token_type_ids,
            "seq_domain_labels": self.features[idx].seq_domain_labels,
            "seq_intent_labels": self.features[idx].seq_intent_labels,
            "seq_token_labels": self.features[idx].seq_token_labels
        }


# 因为需要完成 cls和ner 两项任务，这里采用自定义训练模型
class BertModelForClsAndNer(Module):
    def __init__(self):
        super(BertModelForClsAndNer, self).__init__()
        self.args = Arguments()

        self.bert_model = BertModel.from_pretrained(self.args.bert_model_dir)

        # 自定义全连接层（对 bert_model的输出进行处理，得到输出结果），将维度转换成任务对应的类别数量
        # self.cls_domain_linear = Linear(self.args.hidden_dim, self.args.cls_domain_dim)
        self.cls_intent_linear = Sequential(
            Dropout(self.args.hidden_dropout_prob),
            Linear(self.args.hidden_dim, self.args.cls_intent_dim)
        )
        self.ner_slot_linear = Sequential(
            Dropout(self.args.hidden_dropout_prob),
            Linear(self.args.hidden_dim, self.args.ner_slot_dim)
        )

    def forward(self, inputs):
        # inputs 先经过 原始的BertModel处理后，得到 last_hidden_state，pooler_output
        # last_hidden_state 包含所有样本 所有隐藏层的输出，可以完成NER任务 -> (batch_size, sequence_size, hidden_dim)
        # pooler_output 是最后隐藏层的输出（经过池化操作），可以完成CLS任务 -> (batch_size, hidden_dim)
        outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        # cls_intent_output -> (batch_size, self.args.cls_intent_dim)
        # ner_slot_output -> (batch_size, sequence_size, self.args.ner_slot_dim)
        cls_intent_output = self.cls_intent_linear(pooler_output)
        ner_slot_output = self.ner_slot_linear(last_hidden_state)

        return cls_intent_output, ner_slot_output


if __name__ == '__main__':
    args = Arguments()

    proprecess = DatasetPreprocess()
    input_examples = proprecess.getInputExamples("./data/test_process.json", "test")
    input_features = proprecess.getInputFeatures(input_examples, "test")

    bertDataset = BertDataset(input_features)
    bertDataloader = DataLoader(bertDataset, batch_size=args.batch_size, shuffle=True)

    bertModelForClsAndNer = BertModelForClsAndNer()
    for batch, inputs in enumerate(bertDataloader):
        cls_intent_output, ner_slot_output = bertModelForClsAndNer(inputs)

        print("/")
