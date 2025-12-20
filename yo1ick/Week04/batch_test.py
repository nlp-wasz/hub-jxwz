import requests
import json
import time
import threading
import statistics

# 测试数据
test_texts = [
    "很快，好吃，味道足，量大",
    "没有送水没有送水没有送水",
    "太差了，送餐慢，味道也不好",
    "一般般，没什么特别的地方",
    "超级好吃，强烈推荐！"
]


def test_single_request():
    """测试单个请求"""
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(url, headers=headers, json={"text": test_texts[0]})
    end_time = time.time()

    if response.status_code == 200:
        result = response.json()
        print(f"请求成功: {result}")
        print(f"处理时间: {end_time - start_time:.4f}秒")
    else:
        print(f"请求失败: {response.status_code}")


def test_concurrent_requests(concurrency=5, num_requests=20):
    """测试并发请求"""
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}

    results = []
    errors = 0
    lock = threading.Lock()

    def make_request(text, index):
        nonlocal errors
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json={"text": text}, timeout=10)
            end_time = time.time()

            with lock:
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "index": index,
                        "processing_time": end_time - start_time,
                        "result": result
                    })
                else:
                    errors += 1
                    print(f"请求 {index} 失败: {response.status_code}")
        except Exception as e:
            with lock:
                errors += 1
                print(f"请求 {index} 异常: {e}")

    threads = []
    start_time = time.time()

    for i in range(num_requests):
        text = test_texts[i % len(test_texts)]
        thread = threading.Thread(target=make_request, args=(text, i))
        threads.append(thread)
        thread.start()

        # 控制并发数
        if len(threads) >= concurrency:
            for t in threads:
                t.join()
            threads = []

    # 等待剩余线程完成
    for t in threads:
        t.join()

    total_time = time.time() - start_time

    # 计算统计数据
    processing_times = [r["processing_time"] for r in results]
    avg_time = statistics.mean(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0

    print(f"\n并发数: {concurrency}, 总请求数: {num_requests}")
    print(f"成功请求: {len(results)}, 失败请求: {errors}")
    print(f"总耗时: {total_time:.4f}秒")
    print(f"平均处理时间: {avg_time:.4f}秒")
    print(f"最大处理时间: {max_time:.4f}秒")
    print(f"最小处理时间: {min_time:.4f}秒")
    print(f"QPS: {len(results) / total_time:.2f}")


if __name__ == "__main__":
    print("测试单请求:")
    test_single_request()

    print("\n测试并发性能:")
    for concurrency in [1, 5, 10]:
        test_concurrent_requests(concurrency=concurrency, num_requests=50)
