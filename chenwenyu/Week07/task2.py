import codecs
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
import logging
from typing import Dict, List, Optional
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EntityRecognitionTrainer:
    def __init__(self, model_name: str = "../../../models/Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        
    def _setup_device(self) -> torch.device:
        """设置设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def bio_to_word_tags(self, chars: List[str], tags: List[str]) -> List[str]:
        """
        将BIO标签序列转换为单词和对应的标签。
        
        Args:
            chars: 字符序列
            tags: BIO标签序列
            
        Returns:
            格式化的实体列表
        """
        results = []
        current_word = ''
        current_tag = ''

        for char, tag in zip(chars, tags):
            if tag.startswith('B-'):
                # 保存上一个实体
                if current_word:
                    results.append(f"{current_word} : {current_tag}")
                # 开始新实体
                current_word = char
                current_tag = tag[2:]
            elif tag.startswith('I-'):
                current_word += char
            else:  # O标签
                if current_word:
                    results.append(f"{current_word} : {current_tag}")
                    current_word = ''
                    current_tag = ''

        # 处理句末可能遗留的实体
        if current_word:
            results.append(f"{current_word} : {current_tag}")

        return results if results else ["没有识别出实体"]

    def load_data(self, data_config: Dict) -> Dataset:
        """加载和预处理数据"""
        logger.info("加载训练数据...")
        
        # 读取原始数据
        train_lines = codecs.open(data_config['train_sentences']).readlines()[:data_config.get('max_samples', 1000)]
        train_lines = [x.replace(' ', '').strip() for x in train_lines]
        
        train_tags = codecs.open(data_config['train_tags']).readlines()[:data_config.get('max_samples', 1000)]
        train_tags = [x.strip().split(' ') for x in train_tags]

        # 构建数据集
        data = []
        for text, tags in zip(train_lines, train_tags):
            entities = self.bio_to_word_tags(text, tags)
            data.append({
                "text": text,
                "entities": "\n".join(entities)
            })
            
        return Dataset.from_list(data)

    def initialize_model(self):
        """初始化模型和tokenizer"""
        logger.info(f"加载模型: {self.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 显式设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"  # Qwen的EOS token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 确保模型配置与tokenizer一致
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # 准备模型用于k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora(self) -> None:
        """设置LoRA配置"""
        logger.info("设置LoRA配置...")
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 增加rank以获得更好性能
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
            
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def preprocess_function(self, examples: Dict, max_length: int = 512) -> Dict:
        """
        数据预处理函数 - 使用chat template
        """
        # 构建消息
        messages = [
            {
                "role": "system", 
                "content": "你是一个实体识别专家。请从给定的文本中提取所有实体，并按照'实体 : 类型'的格式输出。"
            },
            {
                "role": "user", 
                "content": f"请从以下文本中提取实体：{examples['text']}"
            },
            {
                "role": "assistant", 
                "content": examples['entities']
            }
        ]
        
        # 应用chat template
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize
        model_inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,  # 在collator中动态padding
        )
        
        # 标签就是输入ID（因果语言建模）
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    def setup_training_args(self) -> TrainingArguments:
        """设置训练参数"""
        return TrainingArguments(
            output_dir="./output_ner_qwen",
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=50,
            #eval_steps=100,
            #save_steps=200,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            num_train_epochs=8,
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,
            report_to="none",
            ddp_find_unused_parameters=False,
        )

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        # 这里可以添加自定义的实体识别评估指标
        return {"accuracy": 0.0}  # 占位符

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """训练模型"""
        logger.info("开始训练...")
        
        # 初始化模型
        self.initialize_model()
        self.setup_lora()
        
        # 预处理数据
        tokenized_train = train_dataset.map(
            lambda x: self.preprocess_function(x),
            remove_columns=train_dataset.column_names,
            desc="Tokenizing训练数据"
        )
        
        tokenized_eval = None
        if eval_dataset:
            tokenized_eval = eval_dataset.map(
                lambda x: self.preprocess_function(x),
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing验证数据"
            )

        # 设置训练参数
        training_args = self.setup_training_args()
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        # 回调函数
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        return trainer

    def predict(self, text: str, max_new_tokens: int = 128) -> str:
        """预测实体"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未初始化，请先加载模型")
            
        messages = [
            {
                "role": "system", 
                "content": "你是一个实体识别专家。请从给定的文本中提取所有实体，并按照'实体 : 类型'的格式输出。"
            },
            {
                "role": "user", 
                "content": f"请从以下文本中提取实体：{text}"
            }
        ]
        
        # 应用chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)
        
        # 生成配置 - 非采样模式
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,    #使用贪婪解码，适合实体识别这种确定性任务
            "num_beams": 1,  # 单beam，相当于贪婪解码
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # 避免重复
            "length_penalty": 1.0,  # 长度惩罚系数
        }

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response

def main():
    """主函数"""
    # 配置
    config = {
        'model_name': "../../../models/Qwen/Qwen2.5-0.5B",  # 使用更新的模型
        'data_config': {
            'train_sentences': '../msra/train/sentences.txt',
            'train_tags': '../msra/train/tags.txt',
            'max_samples': 2000
        }
    }
    
    # 创建训练器
    trainer = EntityRecognitionTrainer(config['model_name'])
    
    # 加载数据
    dataset = trainer.load_data(config['data_config'])
    
    # 分割训练集和验证集
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # 训练
    trainer.train(train_dataset, eval_dataset)
    
    # 测试
    test_text = "帮我导航到四川大学望江校区东门"
    result = trainer.predict(test_text)
    print(f"输入: {test_text}")
    print(f"识别结果: {result}")

if __name__ == "__main__":
    main()
