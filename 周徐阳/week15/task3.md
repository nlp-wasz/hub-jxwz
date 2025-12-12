## Task3：DeepResearch 并发生成与 ReAct 反馈

- 目标：让不同章节并发生成，并在每章内加入 ReAct 风格的自反反馈，提升章节质量。
- 核心改动：`09_DeepResearch.py` 新增并发章节流水线（`asyncio.Semaphore` + `asyncio.gather`），每章草稿都会通过 `ReflectionAgent` 审核，不满足则结合反馈二次改写。
- 可调参数：`deep_research(..., max_concurrent_sections=3, max_revision_rounds=2, max_sections=5)`，默认 3 章并发、每章最多 2 轮反馈。

### 使用方式
1) 确认环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL`（脚本内已有默认值，可按需替换）。  
2) 直接运行 `python 09_DeepResearch.py`，或在其他脚本中调用 `await deep_research(topic, ...)` 指定并发/反馈参数。  
3) 输出会包含摘要、章节正文、结论与展望以及引用来源列表。

### 流程亮点
- 大纲阶段仍由 Orchestrator 生成 JSON 结构，后续章节会按规划并发处理。  
- 每个章节：搜索 → 抓取 Top2 网页 → DraftingAgent 按 ReAct 线索+成文 → ReflectionAgent 判定是否“满足”，否则给出子问题反馈后继续改写（最多 `max_revision_rounds` 轮）。  
- 整合阶段：Orchestrator 汇总全部章节，补充摘要、结论与引用。

### 提示
- 如遇限流，可下调 `max_concurrent_sections`；若需要更严格的审稿，可上调 `max_revision_rounds`。  
- 章节内仍基于抓取内容与搜索摘要撰写，保持引用 URL。***
