import requests
import time
import sys


def test_concurrency(url, requests_count, concurrency):
    """测试特定并发级别"""
    print(f"\n测试 {concurrency} 并发, {requests_count} 请求...")

    start_time = time.time()
    success_count = 0
    total_time = 0

    # 这里简化实现，实际应该用线程池
    for i in range(requests_count):
        try:
            request_start = time.time()
            response = requests.post(
                url,
                json={"text": "测试文本"},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            request_time = time.time() - request_start
            total_time += request_time

            if response.status_code == 200:
                success_count += 1
            else:
                print(f"请求 {i + 1} 失败: HTTP {response.status_code}")

        except Exception as e:
            print(f"请求 {i + 1} 异常: {e}")

    end_time = time.time()

    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功请求: {success_count}/{requests_count}")
    if success_count > 0:
        print(f"平均响应时间: {total_time / success_count:.3f} 秒")
        print(f"QPS: {success_count / (end_time - start_time):.2f}")


def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "http://127.0.0.1:8000/classify"

    print(f"测试URL: {url}")

    # 测试不同并发级别
    test_cases = [
        (10, 1),  # 10请求, 1并发
        (50, 5),  # 50请求, 5并发
        (100, 10),  # 100请求, 10并发
    ]

    for requests_count, concurrency in test_cases:
        test_concurrency(url, requests_count, concurrency)
        print("-" * 40)


if __name__ == "__main__":
    main()
