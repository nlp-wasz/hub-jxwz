# 外卖评价分类系统

基于BERT预训练模型的外卖评价文本分类系统，可以将评价文本分类为"好评"或"差评"。

## 项目结构

```
TakeawayEvaluationClassification/
  - api/                     # API接口定义
  - assets/                  # 静态资源
    - dataset/               # 数据集
    - models/                # 预训练模型
    - weights/               # 模型权重
  - core/                    # 核心代码
    - model/                 # 模型定义
    - train/                 # 训练相关代码
  - schema/                  # 数据模型
  - config.py                # 配置文件
  - logger.py                # 日志配置
  - main.py                  # 主程序入口
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- FastAPI
- Transformers
- Pydantic

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 上运行，可以通过 `http://localhost:8000/docs` 访问API文档。

### API使用

#### 评价文本分类

```bash
curl -X POST "http://localhost:8000/api/evaluation_classify/bert" \
    -H "Content-Type: application/json" \
    -d '{"request_id": "123", "request_evaluation": "这家外卖很好吃，包装也很好！"}'
```

## 模型训练

训练脚本位于 `core/train/train_bert.py`

## 项目特点

1. 基于BERT预训练模型，提供高精度的评价文本分类
2. 使用FastAPI提供高性能的RESTful API
3. 支持批量文本分类
