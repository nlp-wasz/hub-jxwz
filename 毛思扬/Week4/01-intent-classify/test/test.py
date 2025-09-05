import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os

# macOS系统中常用的中文字体
zh_font = fm.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']

# 读取 ab 测试结果
def parse_ab_results(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取关键指标
    rps = float(re.search(r'Requests per second:\s+([0-9.]+)', content).group(1))
    time_per_request = float(re.search(r'Time per request:\s+([0-9.]+).*\(mean\)', content).group(1))
    
    # 提取连接时间数据
    connection_times = re.search(r'Connect:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
    processing_times = re.search(r'Processing:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
    waiting_times = re.search(r'Waiting:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
    total_times = re.search(r'Total:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
    
    return {
        'rps': rps,
        'time_per_request': time_per_request,
        'connection_times': [float(x) for x in connection_times.groups()] if connection_times else [],
        'processing_times': [float(x) for x in processing_times.groups()] if processing_times else [],
        'waiting_times': [float(x) for x in waiting_times.groups()] if waiting_times else [],
        'total_times': [float(x) for x in total_times.groups()] if total_times else []
    }

# 绘制性能指标图表
def plot_performance_metrics(results, save_path='ab_test_results.png'):
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Apache Bench 压测结果分析', fontsize=16, fontweight='bold')
    
    # 1. RPS 和平均响应时间
    ax1.bar(['每秒请求数'], [results['rps']], color='skyblue')
    ax1.set_ylabel('请求数/秒', fontsize=12)
    ax1.set_title(f'每秒请求数: {results["rps"]:.2f}', fontsize=14)
    
    ax2.bar(['平均响应时间 (毫秒)'], [results['time_per_request']], color='lightgreen')
    ax2.set_ylabel('毫秒', fontsize=12)
    ax2.set_title(f'平均响应时间: {results["time_per_request"]:.2f} 毫秒', fontsize=14)
    
    # 3. 连接时间分布
    if results['total_times']:
        metrics_names = ['最小值', '平均值', '标准差', '中位数', '最大值']
        times_data = [
            results['connection_times'],
            results['processing_times'],
            results['waiting_times'],
            results['total_times']
        ]
        labels = ['连接时间', '处理时间', '等待时间', '总时间']
        
        x = np.arange(len(metrics_names))
        width = 0.2
        
        for i, (data, label) in enumerate(zip(times_data, labels)):
            if data:
                ax3.bar(x + i*width, data, width, label=label, alpha=0.8)
        
        ax3.set_xlabel('指标', fontsize=12)
        ax3.set_ylabel('时间 (毫秒)', fontsize=12)
        ax3.set_title('连接时间分布', fontsize=14)
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(metrics_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 百分位响应时间
    if results['total_times'] and len(results['total_times']) >= 5:
        # 使用总时间的各个百分位数据
        percentiles_data = {
            '50%': results['total_times'][3],   # 中位数
            '95%': results['total_times'][4],   # 95% 百分位
        }
        
        # 如果有完整的百分位数据，可以添加更多点
        if len(results['total_times']) >= 5:
            percentiles_x = list(percentiles_data.keys())
            percentiles_y = list(percentiles_data.values())
            
            ax4.plot(percentiles_x, percentiles_y, marker='o', linewidth=2, markersize=8, color='red')
            ax4.set_xlabel('百分位', fontsize=12)
            ax4.set_ylabel('响应时间 (毫秒)', fontsize=12)
            ax4.set_title('响应时间百分位分布', fontsize=14)
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (x, y) in enumerate(zip(percentiles_x, percentiles_y)):
                ax4.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # 确保保存路径存在
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    if save_dir != '.' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存图像到本地
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    print(f"图像已保存到: {os.path.abspath(save_path)}")
    
    # 显示图像
    plt.show()
    
    return os.path.abspath(save_path)

# 主函数
if __name__ == "__main__":
    input_files = ['test/ab_results_bert', 'test/ab_results_gpt', 'test/ab_results_regex', 'test/ab_results_tfidf']

    for input_file in input_files:
        # 解析结果
        results = parse_ab_results(input_file+'.txt')
        
        # 绘制图表并保存到本地
        saved_path = plot_performance_metrics(results, input_file+'.png')
        

