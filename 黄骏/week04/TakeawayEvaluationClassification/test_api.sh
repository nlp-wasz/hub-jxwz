#!/bin/bash

# 定义变量
URL="http://localhost:8000/api/v1/evaluation_classify/bert"
REQUEST_COUNT=100
TEST_DATA_FILE="test/data.json"
OUTPUT_DIR="test_results"
CONCURRENCY_LEVELS=(1 5 10 100)

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 检查测试数据文件是否存在
if [ ! -f "$TEST_DATA_FILE" ]; then
    echo "创建测试数据文件 $TEST_DATA_FILE..."
    cat > "$TEST_DATA_FILE" <<EOF
{
  "request_id": "test-123",
  "request_evaluation": "这家外卖很好吃，包装也很好！"
}
EOF
fi

echo "外卖评价分类API性能测试"
echo "=================================="
echo "测试URL: $URL"
echo "请求数量: $REQUEST_COUNT"
echo "测试日期: $(date)"
echo "=================================="
echo ""

# 创建结果汇总文件
SUMMARY_FILE="$OUTPUT_DIR/测试汇总报告.txt"
echo "外卖评价分类API性能测试汇总" > $SUMMARY_FILE
echo "测试日期: $(date)" >> $SUMMARY_FILE
echo "测试URL: $URL" >> $SUMMARY_FILE
echo "请求数量: $REQUEST_COUNT" >> $SUMMARY_FILE
echo "=================================" >> $SUMMARY_FILE
printf "%-10s %-15s %-15s %-15s %-15s\n" "并发数" "总耗时(秒)" "每秒请求数" "平均延迟(ms)" "失败请求数" >> $SUMMARY_FILE
echo "=================================" >> $SUMMARY_FILE

# 测试不同并发级别
for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
    echo "测试并发数: $CONCURRENCY"
    echo "--------------------------------"
    
    OUTPUT_FILE="$OUTPUT_DIR/ab_test_c${CONCURRENCY}.txt"
    
    # 运行测试
    ab -n $REQUEST_COUNT -c $CONCURRENCY -p $TEST_DATA_FILE -T 'application/json' -H 'accept: application/json' $URL > $OUTPUT_FILE

    
    # 提取关键性能指标
    TOTAL_TIME=$(grep "Time taken for tests:" $OUTPUT_FILE | awk '{print $5}')
    RPS=$(grep "Requests per second:" $OUTPUT_FILE | awk '{print $4}')
    MEAN_LATENCY=$(grep "Time per request:" $OUTPUT_FILE | head -1 | awk '{print $4}')
    FAILED_REQUESTS=$(grep "Failed requests:" $OUTPUT_FILE | awk '{print $3}')
    
    # 打印结果
    echo "总耗时: $TOTAL_TIME 秒"
    echo "每秒请求数: $RPS"
    echo "平均延迟: $MEAN_LATENCY 毫秒"
    echo "失败请求数: $FAILED_REQUESTS"
    echo ""
    
    # 添加到汇总报告
    printf "%-10s %-15s %-15s %-15s %-15s\n" "$CONCURRENCY" "$TOTAL_TIME" "$RPS" "$MEAN_LATENCY" "$FAILED_REQUESTS" >> $SUMMARY_FILE
done

# 打印汇总报告路径
echo "测试完成！"
echo "详细测试报告可在 $OUTPUT_DIR 目录中找到"
echo "汇总报告: $SUMMARY_FILE"

# 显示汇总报告
echo ""
echo "测试汇总报告:"
echo "================================="
cat $SUMMARY_FILE 