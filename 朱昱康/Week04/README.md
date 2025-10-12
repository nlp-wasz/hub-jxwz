# 外卖评论文本分类系统

基于BERT的中文文本分类系统，用于外卖评论的情感分析。

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 创建模型目录
mkdir -p assets/models/google-bert

# 下载BERT中文预训练模型
# 下载地址：https://huggingface.co/google-bert/bert-base-chinese
```

### 3. 训练模型

```bash
# 使用简化版训练脚本
python train_simple.py
```

训练完成后，模型将保存在：`./models/fine_tuned_bert_simple/`

### 4. 启动服务

```bash
# 启动FastAPI服务
python main.py

# 服务地址：http://localhost:8000
```

## API使用

### API使用

### GET请求
```bash
curl "http://localhost:8000/classify?text=这家餐厅很好吃"
```

### POST请求
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "服务态度很好"}'
```

### 示例返回
```json
{
  "text": "服务态度很好",
  "sentiment": "正面",
  "confidence": 0.9234
}
```

### API文档
访问 http://localhost:8000/docs 查看交互式文档

## 性能压测

使用Apache Bench进行压测：

```bash
# 安装ab（macOS已内置）

# 测试GET接口
ab -n 1000 -c 10 "http://localhost:8000/classify?text=很好吃"
