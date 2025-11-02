#!/usr/bin/env python3
"""
外卖评价情感分类服务启动脚本
"""
import os
import sys
import subprocess
import time
import threading
from waimai_controller import start_server
import requests

# 添加训练代码路径到系统路径
sys.path.append('training_code')


def check_model_exists():
    """检查模型是否存在"""
    # model_path = "./models/waimai-bert"
    #
    # if not os.path.exists(model_path):
    #     return False
    #
    # # 检查必要的模型文件
    # required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    # for file in required_files:
    #     if not os.path.exists(os.path.join(model_path, file)):
    #         return False

    return True


def start_training():
    """启动训练脚本"""
    print("开始训练BERT模型...")
    print("这可能需要几分钟到几十分钟，请耐心等待...")

    try:
        # 方法1: 直接导入训练模块
        try:
            from training_code.waimai_train_bert import train_waimai_bert

            print("使用直接导入方式训练...")
            result = train_waimai_bert(
                data_path="./作业数据-waimai_10k.csv",
                pretrained_model_path="./models/google-bert/bert-base-chinese",
                output_model_path="./models/waimai-bert",
                num_train_epochs=2,  # 减少训练轮数以节省时间
            )

            if result['success']:
                print("模型训练完成 ✓")
                print(f"准确率: {result['eval_results'].get('eval_accuracy', 'N/A'):.4f}")
                return True
            else:
                print(f"训练失败: {result['error']}")
                return False

        except ImportError as import_error:
            print(f"导入训练模块失败: {import_error}")
            print("尝试使用子进程方式...")

            # 方法2: 子进程方式（备用）
            result = subprocess.run([
                sys.executable, "training_code/waimai_train_bert.py"
            ], check=True, capture_output=True, text=True, cwd=".")

            print("模型训练完成 ✓")
            print(result.stdout)
            return True

    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False


def start_api_service():
    """启动API服务"""

    # 在后台线程中启动服务



def main():
    """主函数"""
    print("=" * 60)
    print("外卖评价情感分类服务启动程序")
    print("=" * 60)

    # 1. 检查模型是否存在，如果不存在则训练
    if not check_model_exists():
        print("\n未发现训练好的模型，开始训练...")
        if not start_training():
            print("训练失败，退出程序")
            return
    else:
        print("\n发现已训练的模型，跳过训练步骤")

    # 2. 启动API服务
    print("启动FastAPI服务...")
    start_server(host="0.0.0.0", port=8000, reload=False)




if __name__ == "__main__":
    main()
