"""
外卖评价BERT模型训练模块
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import os
import warnings
from typing import Optional, Dict, Any



class WaimaiBertTrainer:
    """外卖BERT模型训练器"""
    
    def __init__(self, 
                 data_path: str = "../作业数据-waimai_10k.csv",
                 pretrained_model_path: str = "../models/google-bert/bert-base-chinese",
                 output_model_path: str = "../models/waimai-bert",
                 max_length: int = 50,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        初始化训练器
        
        Args:
            data_path: 训练数据路径
            pretrained_model_path: 预训练BERT模型路径
            output_model_path: 输出模型路径
            max_length: 最大序列长度
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_path = data_path
        self.pretrained_model_path = pretrained_model_path
        self.output_model_path = output_model_path
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state
        
        # 设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print_device_info()
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.test_dataset = None
    
    def _print_device_info(self):
        """打印设备信息"""
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU名称: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("未检测到GPU，将使用CPU训练")
    
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("加载数据...")
        
        # 加载数据
        dataset_df = pd.read_csv(self.data_path)
        texts = dataset_df['review'].tolist()
        labels = dataset_df['label'].tolist()
        
        print(f"数据集大小: {len(texts)}")
        print(f"正面样本: {sum(labels)}, 负面样本: {len(labels) - sum(labels)}")
        
        # 划分训练集和测试集
        x_train, x_test, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )
        
        print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")
        
        return x_train, x_test, train_labels, test_labels
    
    def load_model(self):
        """加载预训练模型"""
        print(f"加载预训练模型: {self.pretrained_model_path}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.pretrained_model_path, 
            num_labels=2
        )
        self.model = self.model.to(self.device)
        
        print("模型加载完成")
    
    def prepare_datasets(self, x_train, x_test, train_labels, test_labels):
        """准备训练数据集"""
        print(f"准备数据集，最大序列长度: {self.max_length}")
        
        # 编码文本
        train_encodings = self.tokenizer(
            x_train, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        test_encodings = self.tokenizer(
            x_test, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        
        # 创建Dataset对象
        self.train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        self.test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_labels
        })
        
        print("数据集准备完成")
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    def setup_trainer(self, **kwargs):
        """设置训练器"""
        # 默认训练参数
        default_args = {
            'output_dir': '../results',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'warmup_steps': 300,
            'weight_decay': 0.01,
            'learning_rate': 2e-5,
            'logging_dir': './logs',
            'logging_steps': 50,
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
            'load_best_model_at_end': True,
            'metric_for_best_model': "accuracy",
            'greater_is_better': True,
            'fp16': torch.cuda.is_available(),
            'dataloader_pin_memory': torch.cuda.is_available(),
            'remove_unused_columns': False,
            'report_to': None,
        }
        
        # 根据GPU调整批次大小
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory > 8:
                default_args['per_device_train_batch_size'] = 32
                default_args['per_device_eval_batch_size'] = 32
            else:
                default_args['per_device_train_batch_size'] = 16
                default_args['per_device_eval_batch_size'] = 16
        else:
            default_args['per_device_train_batch_size'] = 4
            default_args['per_device_eval_batch_size'] = 4
            default_args['fp16'] = False
            default_args['dataloader_pin_memory'] = False
        
        # 更新参数
        default_args.update(kwargs)
        
        print(f"训练参数:")
        print(f"- 训练轮数: {default_args['num_train_epochs']}")
        print(f"- 批次大小: {default_args['per_device_train_batch_size']}")
        print(f"- 学习率: {default_args['learning_rate']}")
        print(f"- 混合精度: {default_args['fp16']}")
        
        # 创建训练参数
        training_args = TrainingArguments(**default_args)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("训练器设置完成")
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先调用setup_trainer()")
        
        print("开始训练模型...")
        self.trainer.train()
        
        print("训练完成")
    
    def evaluate(self) -> Dict[str, Any]:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("训练器未初始化")
        
        print("开始评估模型...")
        eval_results = self.trainer.evaluate()
        print(f"评估结果: {eval_results}")
        
        return eval_results
    
    def save_model(self, save_path: Optional[str] = None):
        """保存模型"""
        if save_path is None:
            save_path = self.output_model_path
        
        print(f"保存微调后的模型到: {save_path}")
        
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print("模型保存完成！")
        
        return save_path
    
    def full_training_pipeline(self, **training_kwargs) -> Dict[str, Any]:
        """完整的训练流程"""
        try:
            # 1. 加载数据
            x_train, x_test, train_labels, test_labels = self.load_and_prepare_data()
            
            # 2. 加载模型
            self.load_model()
            
            # 3. 准备数据集
            self.prepare_datasets(x_train, x_test, train_labels, test_labels)
            
            # 4. 设置训练器
            self.setup_trainer(**training_kwargs)
            
            # 5. 训练
            self.train()
            
            # 6. 评估
            eval_results = self.evaluate()
            
            # 7. 保存模型
            model_path = self.save_model()
            
            return {
                'success': True,
                'eval_results': eval_results,
                'model_path': model_path,
                'message': '训练完成'
            }
            
        except Exception as e:
            print(f"训练失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': '训练失败'
            }

def train_waimai_bert(data_path: str = "../作业数据-waimai_10k.csv",
                     pretrained_model_path: str = "../models/google-bert/bert-base-chinese",
                     output_model_path: str = "../models/waimai-bert",
                     **kwargs) -> Dict[str, Any]:
    """
    外卖BERT模型训练的便捷函数
    
    Args:
        data_path: 训练数据路径
        pretrained_model_path: 预训练BERT模型路径
        output_model_path: 输出模型路径
        **kwargs: 其他训练参数
    
    Returns:
        Dict[str, Any]: 训练结果
    """
    trainer = WaimaiBertTrainer(
        data_path=data_path,
        pretrained_model_path=pretrained_model_path,
        output_model_path=output_model_path
    )
    
    return trainer.full_training_pipeline(**kwargs)

# 主程序入口
if __name__ == "__main__":
    print("=" * 60)
    print("外卖评价BERT模型训练")
    print("=" * 60)
    
    # 执行训练
    result = train_waimai_bert()
    
    if result['success']:
        print("\n" + "=" * 60)
        print("训练成功完成！")
        print(f"模型保存路径: {result['model_path']}")
        print(f"准确率: {result['eval_results'].get('eval_accuracy', 'N/A'):.4f}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("训练失败！")
        print(f"错误信息: {result['error']}")
        print("=" * 60)
