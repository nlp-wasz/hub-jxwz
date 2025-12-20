# -*- coding: utf-8 -*-
"""
文本分类器（增强版）
1. 使用Dropout和更深的网络结构提升模型泛化能力
2. 动态填充输入，适应变长文本
3. 混合精度训练提升效率
4. 早停与学习率调度优化训练过程
5. 详细注释与打印，便于理解与调试
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

# -------------------- 配置参数 --------------------
class Config:
    data_path = "../Week01/dataset.csv"
    max_len = 40               # 动态填充时实际最大长度可能更小
    batch_size = 32
    hidden_dim = 256
    embed_dim = 64             # 词嵌入维度（若使用嵌入层）
    lr = 0.001
    epochs = 10
    dropout_prob = 0.5
    use_amp = True             # 启用混合精度训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# -------------------- 数据集类（动态填充） --------------------
class DynamicPaddingDataset(Dataset):
    def __init__(self, texts, labels, char_to_index):
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.char_to_index = char_to_index
        self.vocab_size = len(char_to_index)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 生成BoW向量 (维度=vocab_size)
        bow_vector = torch.zeros(self.vocab_size)
        for char in text:
            bow_vector[self.char_to_index[char]] += 1
        return bow_vector, self.labels[idx]
    
    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.texts)

    @staticmethod
    def collate_fn(batch):
        vectors, labels = zip(*batch)
        return torch.stack(vectors).to(Config.device), torch.stack(labels).to(Config.device)

"""
模型定义

"""
import torch.nn as nn
import torch.nn.functional as F

class EnhancedClassifier(nn.Module):
    """增强版分类器（含三项改进）"""
    def __init__(self, vocab_size, output_dim, hidden_dims=None, use_residual=False, use_ln=True):
        """
        Args:
            vocab_size: 词表大小
            output_dim: 输出类别数
            hidden_dims: 各隐藏层维度列表，如[256, 128, 64]
            use_residual: 是否启用残差连接
            use_ln: 是否启用层归一化
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_ln = use_ln
        
        # 改进3：动态隐藏层维度配置
        hidden_dims = hidden_dims or [256, 128]  # 默认两层结构
        
        # 构建全连接层
        dims = [vocab_size] + hidden_dims
        self.fc_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        
        # 改进1：添加层归一化
        if use_ln:
            self.ln_layers = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims
            ])
        
        # 动态Dropout率（随网络深度递减）
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(Config.dropout_prob * (0.5 ** i))  # 逐层递减
            for i in range(len(hidden_dims))
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        
        # 改进2：残差连接所需的维度匹配
        if use_residual and vocab_size != hidden_dims[0]:
            self.res_proj = nn.Linear(vocab_size, hidden_dims[0])
        else:
            self.res_proj = None

    def forward(self, x):
        residual = x if self.use_residual else None
        
        for i, (fc, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            x = fc(x)
            
            # 改进1：层归一化
            if self.use_ln:
                x = self.ln_layers[i](x)
                
            # 改进2：残差连接（仅第一层）
            if self.use_residual and i == 0 and residual is not None:
                if self.res_proj is not None:
                    residual = self.res_proj(residual)
                x = x + residual
            
            x = F.relu(x)
            x = dropout(x)
        
        return self.fc_out(x)

# -------------------- 训练工具 --------------------
@contextmanager
def full_print():
    """临时完整打印张量"""
    old_threshold = torch._tensor_str.PRINT_OPTS.threshold
    torch.set_printoptions(threshold=10_000)
    yield
    torch.set_printoptions(threshold=old_threshold)

def train_model(model, dataloader, config):
    """训练循环（含混合精度和早停）"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = GradScaler(enabled=config.use_amp)
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
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停与保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')

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
    index_to_char = {i: c for c, i in char_to_index.items()}
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
    print(f"词表大小: {vocab_size}")  # 应该是2823
    print(f"输入数据形状示例: {dataset[0][0].shape}")  # 检查实际输入维度


    # 模型
    if False:    #原始调用方式
        model = EnhancedClassifier(
            vocab_size=len(char_to_index),
            hidden_dim=config.hidden_dim,
            output_dim=len(label_to_index)
        ).to(config.device)

    num_classes = len(label_to_index)
    print(f"数据集中类别数量: {num_classes}, 最大标签索引: {max(numerical_labels)}")  # 确保类别数正确

    # 启用所有改进的配置
    enhanced_model = EnhancedClassifier(
        vocab_size=2823,
        output_dim=num_classes,
        hidden_dims=[384, 192, 96],  # 3层动态维度
        use_residual=True,  # 启用残差
        use_ln=True        # 启用层归一化
    )

    # 训练
    train_model(enhanced_model, dataloader, config)

    # 测试样例
    test_texts = ["帮我导航到北京", "查询明天天气"]
    for text in test_texts:
        pred = classify_text(text, enhanced_model, char_to_index, index_to_label)
        print(f"输入: '{text}' => 预测类别: '{pred}'")

    # 打印模型信息
    print("\n模型结构:")
    print(enhanced_model)
    print(f"\n词表大小: {len(char_to_index)}, 类别数: {len(label_to_index)}")