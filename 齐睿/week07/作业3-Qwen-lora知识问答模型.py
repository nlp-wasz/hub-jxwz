import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
import os
from tqdm import tqdm
import numpy as np
import re


class QwenQAFinetuner:
    def __init__(self, model_path, data_path, output_dir="./output_qa_qwen"):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.converted_data_path = os.path.join(data_path, "converted_instruction_data.csv")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 使用新的dtype参数替代torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def convert_qa_to_instruction_format(self, data):
        """将QA格式正确转换为指令微调格式"""
        instructions = []
        inputs = []
        outputs = []

        for paragraph in data['data']:
            context = paragraph['paragraphs'][0]['context']
            for qa in paragraph['paragraphs'][0]['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']

                # 更清晰的指令格式
                instructions.append("根据上下文回答问题")
                inputs.append(f"上下文: {context}\n问题: {question}")
                outputs.append(answer)  # 仅包含答案，不包含其他内容

        return pd.DataFrame({
            "instruction": instructions,
            "input": inputs,
            "output": outputs
        })

    def load_or_convert_data(self):
        """加载或转换数据"""
        # 检查是否已存在转换后的数据
        if os.path.exists(self.converted_data_path):
            print("加载已转换的数据...")
            return pd.read_csv(self.converted_data_path)

        print("转换数据格式...")
        # 加载原始数据
        train_data = json.load(open(os.path.join(self.data_path, 'train.json'), encoding='utf-8'))
        dev_data = json.load(open(os.path.join(self.data_path, 'dev.json'), encoding='utf-8'))

        # 转换数据格式
        train_df = self.convert_qa_to_instruction_format(train_data)
        dev_df = self.convert_qa_to_instruction_format(dev_data)

        # 合并训练和验证数据
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)

        # 保存转换后的数据
        combined_df.to_csv(self.converted_data_path, index=False, encoding='utf-8')
        print(f"数据已转换并保存到: {self.converted_data_path}")

        return combined_df

    def process_func(self, example, max_length=512):
        """处理单个样本的函数 - 确保模型学习直接回答"""
        # 构建更直接的提示格式
        prompt_text = f"<|im_start|>system\n你是一个问答助手，请根据上下文直接回答问题，不要添加任何解释或额外内容。<|im_end|>\n" \
                      f"<|im_start|>user\n{example['input']}<|im_end|>\n" \
                      f"<|im_start|>assistant\n"

        # Tokenize指令和输入部分
        prompt = self.tokenizer(prompt_text, add_special_tokens=False)

        # Tokenize响应部分 - 确保是直接答案
        response = self.tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

        # 组合输入ID和注意力掩码
        input_ids = prompt["input_ids"] + response["input_ids"]
        attention_mask = prompt["attention_mask"] + response["attention_mask"]

        # 构建标签
        labels = [-100] * len(prompt["input_ids"]) + response["input_ids"]

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

    def setup_lora(self):
        """设置LoRA配置"""
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        return self.model

    def train(self, train_size=1000, eval_size=100, epochs=3):
        """训练模型"""
        # 加载数据
        data_df = self.load_or_convert_data()

        # 转换为Hugging Face Dataset
        dataset = Dataset.from_pandas(data_df)

        # 划分训练集和验证集
        total_size = len(dataset)
        train_indices = list(range(min(train_size, total_size)))
        eval_indices = list(range(total_size - min(eval_size, total_size), total_size))

        train_dataset = dataset.select(train_indices)
        eval_dataset = dataset.select(eval_indices)

        # 处理数据
        processed_train = train_dataset.map(
            lambda x: self.process_func(x),
            remove_columns=train_dataset.column_names,
            batched=False
        )

        processed_eval = eval_dataset.map(
            lambda x: self.process_func(x),
            remove_columns=eval_dataset.column_names,
            batched=False
        )

        # 设置LoRA
        self.model = self.setup_lora()

        # 确保模型处于训练模式
        self.model.train()

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=50,
            eval_steps=50,
            num_train_epochs=epochs,
            save_steps=100,
            learning_rate=1e-4,
            gradient_checkpointing=False,
            report_to="none",
            do_eval=True,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_eval,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                pad_to_multiple_of=8
            ),
        )

        # 开始训练
        print("开始训练...")
        trainer.train()

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"模型已保存到: {self.output_dir}")

        return trainer

    def clean_model_response(self, response):
        """清理模型响应，移除思考过程和无关内容"""
        # 移除所有思考模式前缀
        patterns = [
            r'^好的，.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^我现在需要.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^根据.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^首先，.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^让我.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^用户问的是.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^我需要.*?(?=答案|回答|所以|因此|最终|结论|：|$)',
            r'^下拉菜单.*',  # 特别处理下拉菜单等无关内容
            r'^选项包括.*',
            r'^戰國無雙.*',  # 处理特定游戏名称相关的内容
        ]

        for pattern in patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)

        # 移除所有标点符号开头的部分
        response = re.sub(r'^[，。：；！？、]+', '', response)

        # 移除多余的空格和换行
        response = response.strip()

        # 如果清理后为空，返回原始响应
        if not response:
            return response

        # 如果响应仍然包含思考模式，尝试提取最后一部分
        if "答案" in response or "回答" in response:
            # 尝试找到"答案"或"回答"后面的内容
            match = re.search(r'(?:答案|回答)[：:]\s*(.*)', response)
            if match:
                response = match.group(1)

        return response

    def predict(self, context, question, device='cuda'):
        """预测单个问题的答案"""
        # 构建更简洁的指令格式
        input_text = f"上下文: {context}\n问题: {question}"

        messages = [
            {"role": "system", "content": "你是一个问答助手，请根据上下文直接回答问题，不要添加任何解释或额外内容。"},
            {"role": "user", "content": input_text}
        ]

        # 应用聊天模板
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize输入
        model_inputs = self.tokenizer([formatted_text], return_tensors="pt").to(device)

        # 生成预测 - 使用更严格的参数
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=50,  # 减少生成长度
                do_sample=False,  # 使用贪婪搜索
                temperature=0.1,  # 更低的温度
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 增加重复惩罚
                no_repeat_ngram_size=3,
                early_stopping=True,
                num_return_sequences=1,
            )

        # 提取生成的文本
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 清理响应
        cleaned_response = self.clean_model_response(response)

        return cleaned_response.strip()

    def evaluate(self, test_data, device='cuda'):
        """在测试集上评估模型"""
        results = []

        for i, item in enumerate(tqdm(test_data, desc="评估模型")):
            context = item['context']
            question = item['question']
            expected_answer = item['answers']['text'][0]

            try:
                predicted_answer = self.predict(context, question, device)

                # 更灵活的匹配逻辑
                def normalize_text(text):
                    text = re.sub(r'\s+', '', text)  # 移除所有空白字符
                    text = re.sub(r'[，。：；！？、]', '', text)  # 移除标点
                    return text.lower()

                norm_expected = normalize_text(expected_answer)
                norm_predicted = normalize_text(predicted_answer)

                # 多种匹配条件
                is_match = (
                        norm_expected in norm_predicted or
                        norm_predicted in norm_expected or
                        len(set(norm_expected) & set(norm_predicted)) / max(len(norm_expected),
                                                                            len(norm_predicted)) > 0.7
                )

                results.append({
                    'context': context,
                    'question': question,
                    'expected': expected_answer,
                    'predicted': predicted_answer,
                    'match': is_match
                })
            except Exception as e:
                print(f"处理第 {i} 个样本时出错: {e}")
                results.append({
                    'context': context,
                    'question': question,
                    'expected': expected_answer,
                    'predicted': f"错误: {e}",
                    'match': False
                })

        return results


def load_test_data(data_path):
    """加载测试数据"""
    test_data = json.load(open(os.path.join(data_path, 'dev.json'), encoding='utf-8'))

    # 提取测试样本
    test_samples = []
    for paragraph in test_data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            # 确保获取完整的答案文本
            answer_text = qa['answers'][0]['text']
            test_samples.append({
                'context': context,
                'question': qa['question'],
                'answers': {'text': [answer_text]}  # 确保格式正确
            })

    return test_samples


def main():
    """主函数"""
    # 参数设置
    model_path = "../asset/models/Qwen/Qwen3-0.6B/"
    data_path = "./cmrc2018_public"
    output_dir = "./output_qa_qwen"

    # 初始化微调器
    finetuner = QwenQAFinetuner(model_path, data_path, output_dir)

    # 检查是否已训练模型
    if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print("未找到训练好的模型，开始训练...")
        # 训练模型
        finetuner.train(train_size=50, eval_size=10, epochs=10)
    else:
        print("加载已训练的模型...")
        # 加载已训练的LoRA权重
        from peft import PeftModel
        finetuner.model = PeftModel.from_pretrained(finetuner.model, output_dir)

    # 加载测试数据
    test_data = load_test_data(data_path)[:10]  # 只测试前10个样本

    # 评估模型
    results = finetuner.evaluate(test_data)

    # 打印结果
    print("\n评估结果:")
    for i, result in enumerate(results):
        print(f"\n样本 {i + 1}:")
        print(f"问题: {result['question']}")
        print(f"预期答案: {result['expected']}")
        print(f"预测答案: {result['predicted']}")
        print(f"匹配: {result['match']}")

    # 计算准确率
    accuracy = sum(1 for r in results if r['match']) / len(results)
    print(f"\n准确率: {accuracy:.2%}")


if __name__ == "__main__":
    main()