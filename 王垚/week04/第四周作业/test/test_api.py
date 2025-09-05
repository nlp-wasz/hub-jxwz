"""
测试外卖评价情感分类API
"""
import requests
import json
import time

# API服务地址
API_BASE_URL = "http://localhost:8000"

def test_single_classify():
    """测试单条文本分类"""
    print("=== 测试单条文本分类 ===")
    
    test_data = {
        "request_id": "test_001",
        "text": "菜品很好吃，送餐也很快，五星好评！"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/classify", json=test_data)
        result = response.json()
        
        print(f"请求状态: {response.status_code}")
        print(f"响应结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"测试失败: {e}")

def test_batch_classify():
    """测试批量文本分类"""
    print("\n=== 测试批量文本分类 ===")
    
    test_data = {
        "request_id": "test_002",
        "text": [
            "菜品很好吃，送餐也很快，五星好评！",
            "味道一般，而且送餐太慢了，差评",
            "性价比不错，下次还会点",
            "菜品质量不行，不推荐",
            "服务态度很好，菜品也新鲜"
        ]
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/classify", json=test_data)
        result = response.json()
        
        print(f"请求状态: {response.status_code}")
        print(f"响应结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"测试失败: {e}")





def main():
    """主测试函数"""
    print("开始测试外卖评价情感分类API...")
    print(f"API地址: {API_BASE_URL}")
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(2)
    
    # 执行各项测试
    test_single_classify()
    test_batch_classify()

    
    print("\n测试完成！")
    print(f"API文档地址: {API_BASE_URL}/docs")

if __name__ == "__main__":
    main()
