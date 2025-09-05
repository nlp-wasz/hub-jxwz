#!/usr/bin/env python3
"""
使用ab (Apache Bench) 测试外卖评价情感分类API的并发性能
"""
import subprocess
import time
import requests
import json
import os
import sys



def run_ab_test(concurrency, total_requests=100):
    """
    运行ab测试
    
    Args:
        concurrency: 并发数
        total_requests: 总请求数
    """
    print(f"\n{'='*60}")
    print(f"测试并发数: {concurrency}, 总请求数: {total_requests}")
    print(f"{'='*60}")
    
    # 准备测试数据
    test_data = {
        "request_id": f"ab_test_c{concurrency}",
        "text": "菜品很好吃，送餐也很快，五星好评！"
    }
    
    # 创建临时数据文件
    data_file = f"test_data_c{concurrency}.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    
    try:
        # 构建ab命令
        cmd = [
            'C:\\Users\\13744\\PycharmProjects\\aitest\\tools\\httpd-2.4.65-250724-Win64-VS17\\Apache24\\bin\\ab.exe',
            '-n', str(total_requests),  # 总请求数
            '-c', str(concurrency),     # 并发数
            '-p', data_file,            # POST数据文件
            '-T', 'application/json',   # Content-Type
            '-H', 'Accept: application/json',  # Accept头
            'http://localhost:8000/classify'
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行ab测试
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print("✓ 测试完成")
            print("\n" + "="*40 + " 测试结果 " + "="*40)
            print(result.stdout)
            
            # 提取关键指标
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Time taken for tests:' in line:
                    print(f"📊 总耗时: {line.split(':')[1].strip()}")
                elif 'Requests per second:' in line:
                    print(f"📊 QPS: {line.split(':')[1].strip()}")
                elif 'Time per request:' in line and 'mean' in line:
                    print(f"📊 平均响应时间: {line.split(':')[1].strip()}")
                elif 'Failed requests:' in line:
                    print(f"📊 失败请求: {line.split(':')[1].strip()}")
        else:
            print("✗ 测试失败")
            print("错误输出:", result.stderr)
    
    except subprocess.TimeoutExpired:
        print("✗ 测试超时")
    except Exception as e:
        print(f"✗ 测试异常: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(data_file):
            os.remove(data_file)

def main():
    """主函数"""
    print("🚀 外卖评价情感分类API并发性能测试")
    print("="*80)
    
    # 测试参数
    test_cases = [
        (1, 50),    # 1并发，50请求
        (5, 100),   # 5并发，100请求
        (10, 200),  # 10并发，200请求
    ]
    
    print(f"\n将进行 {len(test_cases)} 组测试...")
    
    for concurrency, total_requests in test_cases:
        run_ab_test(concurrency, total_requests)
        print("\n等待5秒后进行下一组测试...")
        time.sleep(5)
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    main()
