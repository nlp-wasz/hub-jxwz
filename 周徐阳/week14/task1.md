路线1理解与三方案对比

  - 路线1要点（见 README.md 中“路线 1：RAG + Qwen Thinking 模型”）
      - 用 mineru/deepseek-ocr/qwen3-vl 将 PDF 公式解析为 LaTeX，构建可检索的公式语料。
      - RAG 召回 topN 公式（可加 rerank），把用户问句 + 候选公式交给 Qwen Thinking。
      - Qwen Thinking 要么直接推理，要么生成 Python/sympy 代码在沙箱执行，输出数值答案。
      - 优势：处理非标准文本、语义理解强、交互自然；劣势：精度与成本、溯源难。
      - 适用：语义解释、多步推理；不适用：严格高精度计算。
  - 方案1（待选方案/solu1）
      - 流程：dots.ocr 解析 PDF→data_prepare.py 合并分页 Markdown→Qwen3 Embedding 做相似度召回→Qwen3-235B-Thinking 直接依据 top8 公式生成答案（未落地代码执行）→正则抽取 "answer"，无法提取或无答案时默认 10。入
        口 code/main.py。
      - 特点：完全符合路线1（LLM 主导推理，不生成代码）；利用 Thinking 模型再次筛公式，缓解检索误差；有针对“万元”等单位和缺参估值的提示。
      - 风险/不足：依赖超大闭源 API 成本高；未实际执行代码，数值可靠性弱；正则抽取鲁棒性有限，默认值 10 可能掩盖错误。
  - 方案2（待选方案/solu2）
      - 流程：read_pdf_or_md.py 抽取 PDF/MD 文本→人工或文本匹配得到 matched.csv（问题-知识对）→本地 Qwen2.5-7B 生成答案并写 content_list2.csv。
      - 特点：轻量、本地运行，无复杂管线；直接把匹配到的背景 + 问题送入小模型回答，提示中要求无法算则估计。
      - 风险/不足：检索仅基于简单匹配（缺乏向量召回与 rerank）；小模型推理与数值精度弱；无代码执行与后处理，偏离路线1 的“Thinking+代码”思路，结果稳定性难保。
  - 方案3（待选方案/solu3）
      - 流程：1) 1_run_inference.py.py 解析 documents.zip（PyMuPDF 直接读 PDF/MD 文本）→text2vec 语义索引→Top3 文档送入本地 Ollama qwen3:8b，用结构化 Prompt 让模型输出 JSON（思考过程/意图/公式/参数/答案），并
        用 Sympy 尝试代入计算，日志写入 user_data/run_log_final.jsonl。2) 2_generate_submission.py.py 基于意图和关键词后处理，映射成单一数值。
      - 特点：完整 RAG + 本地 LLM，且有代码级 Sympy 计算尝试，符合路线1“检索+Thinking+（可选）代码执行”的精神；两阶段流水线、日志可追溯。
      - 风险/不足：OCR/公式结构化缺失（直接用纯文本）；Prompt JSON 解析易碎；Sympy 仅在参数齐全时发挥作用，后处理对意图/关键词依赖大，可能将文本结论硬映射为数值。
  - 综合建议
      - 若坚持路线1目标（LLM 推理 + 代码执行 + RAG），方案3 最接近可落地版本，可在此基础上补充公式结构化与更稳健的 JSON/数值抽取。
      - 方案1 可作为强基线（利用大模型兜底），但需补充代码求值或更强后处理来提高数值可信度。
      - 方案2 适合作为轻量对照，但需加强检索与数值验证后再用于正式流水线。
