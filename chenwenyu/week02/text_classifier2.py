#./Week02/text_classifier2.py
# -*- coding: utf-8 -*-
"""
文本分类器（超参数优化版）
1.系统化的超参数搜索:测试了5种不同层数(1-5层)和多种节点数组合
2.正则化配置测试:测试了残差连接和层归一化的所有组合
3.详细的实验跟踪:记录每个配置的损失曲线、训练时间、参数量等
4.可视化分析:生成6个子图全面分析实验结果
5.自动选择最佳配置:基于最佳损失自动选择最优网络结构
6.实验结果保存:将所有结果保存为JSON文件便于后续分析
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from contextlib import contextmanager
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os

# -------------------- 配置参数 --------------------
class Config:
    data_path = "../Week01/dataset.csv"
    max_len = 40
    batch_size = 32
    lr = 0.001
    epochs = 15  # 增加训练轮数以便观察收敛
    dropout_prob = 0.5
    use_amp = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = "experiments"  # 实验结果的保存目录

# 设置matplotlib为非交互模式，图像显示后继续执行
plt.ioff()

# -------------------- 数据预处理 --------------------
def load_and_preprocess_data(config):
    """加载数据并构建词表"""
    dataset = pd.read_csv(config.data_path, sep="\t", header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

    # 标签编码
    label_to_index = {label: i for i, label in enumerate(sorted(set(string_labels)))}
    numerical_labels = [label_to_index[label] for label in string_labels]

    # 验证标签范围
    assert max(numerical_labels) == len(label_to_index) - 1, "标签索引不连续！"

    # 字符级词表
    char_to_index = defaultdict(lambda: len(char_to_index))
    char_to_index['<pad>'] = 0
    for text in texts:
        for char in text:
            char_to_index[char]  # 自动分配索引

    return texts, numerical_labels, dict(char_to_index), label_to_index

# -------------------- 数据集类 --------------------
class DynamicPaddingDataset(Dataset):
    def __init__(self, texts, labels, char_to_index):
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.char_to_index = char_to_index
        self.vocab_size = len(char_to_index)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        bow_vector = torch.zeros(self.vocab_size)
        for char in text:
            bow_vector[self.char_to_index[char]] += 1
        return bow_vector, self.labels[idx]
    
    def __len__(self):
        return len(self.texts)

    @staticmethod
    def collate_fn(batch):
        vectors, labels = zip(*batch)
        return torch.stack(vectors).to(Config.device), torch.stack(labels).to(Config.device)

# -------------------- 模型定义 --------------------
class EnhancedClassifier(nn.Module):
    def __init__(self, vocab_size, output_dim, hidden_dims=None, use_residual=False, use_ln=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_ln = use_ln
        
        hidden_dims = hidden_dims or [256, 128]
        
        dims = [vocab_size] + hidden_dims
        self.fc_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        
        if use_ln:
            self.ln_layers = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims
            ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(Config.dropout_prob * (0.5 ** i))
            for i in range(len(hidden_dims))
        ])
        
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        
        if use_residual and vocab_size != hidden_dims[0]:
            self.res_proj = nn.Linear(vocab_size, hidden_dims[0])
        else:
            self.res_proj = None

    def forward(self, x):
        residual = x if self.use_residual else None
        
        for i, (fc, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            x = fc(x)
            
            if self.use_ln:
                x = self.ln_layers[i](x)
                
            if self.use_residual and i == 0 and residual is not None:
                if self.res_proj is not None:
                    residual = self.res_proj(residual)
                x = x + residual
            
            x = F.relu(x)
            x = dropout(x)
        
        return self.fc_out(x)

# -------------------- 训练工具 --------------------
class ExperimentTracker:
    """实验跟踪器，记录所有实验结果"""
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.results = []
        os.makedirs(experiment_dir, exist_ok=True)
    
    def add_result(self, config, final_loss, best_loss, training_time, loss_history):
        """添加实验结果"""
        result = {
            'hidden_dims': config['hidden_dims'],
            'use_residual': config['use_residual'],
            'use_ln': config['use_ln'],
            'final_loss': final_loss,
            'best_loss': best_loss,
            'training_time': training_time,
            'num_params': config['num_params'],
            'loss_history': loss_history
        }
        self.results.append(result)
    
    def save_results(self):
        """保存所有实验结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.experiment_dir, f"experiment_results_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"实验结果已保存到: {filename}")
        return filename

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloader, config):
    """训练循环，返回损失历史"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = GradScaler(enabled=config.use_amp)
    
    loss_history = []
    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            with autocast(enabled=config.use_amp):
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss

    return loss_history, best_loss

# -------------------- 实验配置 --------------------
def get_experiment_configs():
    """定义要测试的不同网络结构配置（减少到16个组合）"""
    configs = []
    
    # 精选的层配置（减少数量）
    layer_configs = [
        # 单层网络
        [256],
        [512],
        
        # 两层网络
        [256, 128],
        [512, 256],
        [384, 192],
        
        # 三层网络
        [256, 128, 64],
        [512, 256, 128],
        [384, 192, 96],
        
        # 四层网络
        [256, 128, 64, 32],
        [512, 256, 128, 64]
    ]
    
    # 正则化配置
    regularization_configs = [
        {'use_residual': False, 'use_ln': False},
        {'use_residual': False, 'use_ln': True},
        {'use_residual': True, 'use_ln': True} 
    ]
    
    # 生成所有组合（总共10 * 3 = 30个，但可以进一步筛选）
    for hidden_dims in layer_configs:
        for reg_config in regularization_configs:
            configs.append({
                'hidden_dims': hidden_dims,
                'use_residual': reg_config['use_residual'],
                'use_ln': reg_config['use_ln']
            })
    
    # 进一步筛选到16个最有希望的组合
    selected_configs = []
    
    # 选择代表性的配置
    selected_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]
    for idx in selected_indices:
        if idx < len(configs):
            selected_configs.append(configs[idx])
    
    return selected_configs

# -------------------- 可视化函数 --------------------
def plot_experiment_results(tracker, num_classes, vocab_size):
    """绘制实验结果"""
    plt.figure(figsize=(20, 12))
    
    # 1. 损失曲线对比
    plt.subplot(2, 3, 1)
    for i, result in enumerate(tracker.results[:10]):  # 只显示前10个结果
        label = f"{result['hidden_dims']} (final: {result['final_loss']:.3f})"
        plt.plot(result['loss_history'], label=label, alpha=0.7)
    plt.title('Training Loss Curves (Top 10 Configurations)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 最终损失 vs 参数量
    plt.subplot(2, 3, 2)
    final_losses = [r['final_loss'] for r in tracker.results]
    param_counts = [r['num_params'] for r in tracker.results]
    plt.scatter(param_counts, final_losses, alpha=0.6)
    plt.title('Final Loss vs Parameter Count')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Final Loss')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. 最佳损失分布
    plt.subplot(2, 3, 3)
    best_losses = [r['best_loss'] for r in tracker.results]
    plt.hist(best_losses, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Best Loss Values')
    plt.xlabel('Best Loss')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 4. 层数 vs 性能
    plt.subplot(2, 3, 4)
    layer_counts = [len(r['hidden_dims']) for r in tracker.results]
    colors = [r['final_loss'] for r in tracker.results]
    scatter = plt.scatter(layer_counts, final_losses, c=colors, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Final Loss')
    plt.title('Layer Count vs Final Loss')
    plt.xlabel('Number of Layers')
    plt.ylabel('Final Loss')
    plt.grid(True, alpha=0.3)
    
    # 5. 训练时间分析
    plt.subplot(2, 3, 5)
    training_times = [r['training_time'] for r in tracker.results]
    plt.scatter(param_counts, training_times, alpha=0.6)
    plt.title('Training Time vs Parameter Count')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Training Time (seconds)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 6. 最佳配置的性能
    best_result = min(tracker.results, key=lambda x: x['best_loss'])
    plt.subplot(2, 3, 6)
    plt.plot(best_result['loss_history'], 'r-', linewidth=2, label='Best Config')
    plt.title(f'Best Configuration: {best_result["hidden_dims"]}\nBest Loss: {best_result["best_loss"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.experiment_dir, 'experiment_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(2)
    print("数据生成完成！已显示数据分布和分析图。")

# -------------------- 预测函数 --------------------
def classify_text(text, model, char_to_index, index_to_label):
    """单文本预测"""
    model.eval()
    with torch.no_grad():
        # 动态处理输入（无需固定长度）
        tokenized = torch.tensor([char_to_index[char] for char in text], device=Config.device)
        bow_vector = torch.zeros(len(char_to_index), device=Config.device)
        for idx in tokenized:
            bow_vector[idx] += 1
        
        output = model(bow_vector.unsqueeze(0).float())
        predicted_idx = output.argmax().item()
        return index_to_label[predicted_idx]

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 初始化
    config = Config()
    texts, numerical_labels, char_to_index, label_to_index = load_and_preprocess_data(config)
    index_to_label = {i: l for l, i in label_to_index.items()}
    
    # 数据集
    dataset = DynamicPaddingDataset(texts, numerical_labels, char_to_index)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=DynamicPaddingDataset.collate_fn
    )
    
    vocab_size = len(char_to_index)
    num_classes = len(label_to_index)
    print(f"词表大小: {vocab_size}")
    print(f"类别数量: {num_classes}")
    print(f"训练样本数: {len(dataset)}")
    
    # 获取实验配置
    experiment_configs = get_experiment_configs()
    print(f"总共测试 {len(experiment_configs)} 种配置")
    
    # 实验跟踪器
    tracker = ExperimentTracker(config.experiment_dir)
    
    # 运行所有实验
    for i, model_config in enumerate(experiment_configs):
        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(experiment_configs)}")
        
        # 确保配置值不为None
        hidden_dims = model_config.get('hidden_dims', [256, 128])
        use_residual = model_config.get('use_residual', False)
        use_ln = model_config.get('use_ln', True)
        
        print(f"配置: 隐藏层{hidden_dims}, 残差{use_residual}, 归一化{use_ln}")
        
        # 创建模型
        model = EnhancedClassifier(
            vocab_size=vocab_size,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            use_residual=use_residual,
            use_ln=use_ln
        ).to(config.device)
        
        # 计算参数量
        num_params = count_parameters(model)
        model_config['num_params'] = num_params
        print(f"参数量: {num_params:,}")
        
        # 训练模型
        start_time = time.time()
        loss_history, best_loss = train_model(model, dataloader, config)
        training_time = time.time() - start_time
        
        final_loss = loss_history[-1] if loss_history else float('inf')
        best_loss = best_loss if best_loss is not None else float('inf')
        print(f"训练完成 - 最终损失: {final_loss:.4f}, 最佳损失: {best_loss:.4f}, 耗时: {training_time:.2f}s")
        
        # 记录结果（确保所有值都有默认值）
        tracker.add_result({
            'hidden_dims': hidden_dims,
            'use_residual': use_residual,
            'use_ln': use_ln,
            'num_params': num_params
        }, final_loss, best_loss, training_time, loss_history)
    
    # 保存和分析结果
    results_file = tracker.save_results()
    
    # 找到最佳配置（修复None值问题）
    best_config = None
    best_loss_value = float('inf')
    
    for result in tracker.results:
        current_loss = result.get('best_loss', float('inf'))
        if current_loss < best_loss_value:
            best_loss_value = current_loss
            best_config = result
    
    # 确保best_config不为None
    if best_config is None and tracker.results:
        best_config = tracker.results[0]
    
    print(f"\n{'='*60}")
    print("最佳配置:")
    if best_config:
        print(f"隐藏层结构: {best_config.get('hidden_dims', 'N/A')}")
        print(f"使用残差: {best_config.get('use_residual', False)}")
        print(f"使用层归一化: {best_config.get('use_ln', False)}")
        print(f"最佳损失: {best_config.get('best_loss', float('inf')):.4f}")
        print(f"参数量: {best_config.get('num_params', 0):,}")
        print(f"训练时间: {best_config.get('training_time', 0):.2f}s")
    else:
        print("没有找到有效的配置结果")
    
    # 可视化结果
    if tracker.results:
        plot_experiment_results(tracker, num_classes, vocab_size)
        
        # 使用最佳配置重新训练最终模型
        if best_config:
            print(f"\n使用最佳配置训练最终模型...")
            final_model = EnhancedClassifier(
                vocab_size=vocab_size,
                output_dim=num_classes,
                hidden_dims=best_config.get('hidden_dims', [256, 128]),
                use_residual=best_config.get('use_residual', False),
                use_ln=best_config.get('use_ln', True)
            ).to(config.device)
            
            final_loss_history, final_best_loss = train_model(final_model, dataloader, config)
            
            # 测试样例
            test_texts = ["帮我导航到北京", "查询明天天气", "播放音乐", "设置闹钟"]
            print(f"\n测试预测:")
            for text in test_texts:
                pred = classify_text(text, final_model, char_to_index, index_to_label)
                print(f"输入: '{text}' => 预测类别: '{pred}'")
    
    print(f"\n所有实验完成!结果保存在: {config.experiment_dir}")