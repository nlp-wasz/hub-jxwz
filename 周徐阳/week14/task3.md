## RAG 选公式 + MCP Tool 执行的问答流程

目标：在已有 `tools.py` 中实现的 10 个公式计算函数基础上，结合 RAG 召回与重排，自动选择最相关的公式工具（MCP），完成问答并返回可追溯的计算结果。

### 1) MCP 服务定义（示例）
- 将 `tools.py` 中的函数暴露为 MCP tool，命名建议与函数名一致，便于映射：
  - `simulate_retail_revenue`
  - `estimate_monthly_rent`
  - `nonlinear_interaction`
  - `dissolved_oxygen_rate`
  - `inventory_step`
  - `predict_milk_yield`
  - `complex_system_response`
  - `next_body_weight`
  - `discounted_derivative_price`
  - `average_daily_gain`
- MCP server 示例（fastmcp）配置要点：
  - tool 名与函数名一致。
  - 参数 schema 按函数签名声明。
  - 对有随机性的工具（`simulate_retail_revenue`）暴露 `seed` 以便复现。

### 2) 构建 RAG 检索索引
- 语料：每个公式的摘要 + 变量含义 + 适用场景，可从 `task2.md` 或原 PDF 提炼为文档。
- Embedding：可用 `text2vec` / `bge` / `qwen-embedding` 等向量模型。
- 索引：FAISS 或纯 numpy 余弦相似度皆可，存储字段：
  ```json
  {
    "id": "retail_revenue",
    "tool": "simulate_retail_revenue",
    "title": "零售销售额随机波动模型",
    "desc": "Revenue = Input_kg * (Base_price + Uniform(fluctuation_min, fluctuation_max))，用于进货量与价格波动的销售额预测。"
  }
  ```
- 重排（可选）：使用 rerank 模型对 Top K（如 8）做二次排序，最终输出 Top 1/3 形成候选白名单。

### 3) 问答推理流程（运行时）
1. **检索**：对用户问题做 embedding，取 Top K 公式文档；如启用 rerank，再压缩到 Top 3。
2. **工具白名单**：从 Top 3 提取对应的 MCP tool 名称，作为允许调用的列表。
3. **参数抽取**：用小模型 / 规则 / prompt 生成调用参数 JSON（示例 prompt：“根据工具签名生成参数，不要推理其他公式”）。
4. **调用工具**：按白名单顺序尝试执行；若第一个报错，可回退到下一个候选。捕获异常并记录。
5. **结果组装**：返回：
   - 选中的公式/工具名
   - 输入参数
   - 计算结果
   - 简要解释（可引用公式）

### 4) 参考伪代码（Python 风格）
```python
from rag_index import embed, search   # 自行实现的向量检索
from mcp_client import call_tool      # MCP 调用封装

def answer(question: str):
    hits = search(question, top_k=8)           # [{'tool': 'predict_milk_yield', 'score': 0.72}, ...]
    candidates = rerank(hits, top_n=3)         # 可选
    whitelist = [h['tool'] for h in candidates]

    for tool_name in whitelist:
        params = llm_extract_params(question, tool_name)  # 生成调用参数的 JSON
        try:
            result = call_tool(tool_name, params)
            return {
                "tool": tool_name,
                "params": params,
                "result": result,
                "explain": f"使用 {tool_name} 计算，基于公式 {tool_name} 的定义"
            }
        except Exception as e:
            log_error(tool_name, e)
    return {"error": "no tool succeeded"}
```

### 5) 关键落地细节
- **参数安全网**：对必填字段做类型/范围校验；对对数、平方根、除零等在工具内已有保护。
- **随机性控制**：调用 `simulate_retail_revenue` 时传入 `seed` 以便复现。
- **可观测性**：记录检索得分、白名单、调用参数、结果与异常，便于审计。
- **多轮问答**：可把上一轮的选公式结果和参数填充到下一轮 prompt，减少反复检索。

### 6) 交付物
- 本文档：`task3.md`（当前文件）。
- 工具实现：`tools.py`（已完成，可直接暴露为 MCP tool）。
- 索引构建：将公式元数据持久化（如 `formula_index.jsonl`）并离线向量化，运行时加载检索即可。
