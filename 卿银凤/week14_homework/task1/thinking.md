# 文档公式解析与智能问答 - 技术路线1实现分析

## 项目背景与目标

本项目旨在构建一个高精度、端到端的文档公式智能问答系统（Formula-QAS），实现对文档中公式的准确识别、结构解析、语义理解和基于公式的精确计算与问答。

### 核心任务流程
用户提问 → 关联计算公式 → 抽取输入参数 → 代入计算 → 得到结果 → 汇总回答

### 应用场景
- 变压器公式计算：输入功率、尺寸、温度等参数，计算价格
- 商品销量与仓储计算：基于历史销量、仓库容量等，计算最优备货量
- 不适合常规大模型或传统RAG（知识问答）的场景

---

## 技术路线1：RAG + Qwen Thinking 模型（LLM主导计算）

### 核心思想
利用大型语言模型的强大推理能力和上下文理解能力来完成公式的识别、参数的提取和最终的计算。采用工具调用（Tool-Use/Code Interpreter）模式，让Qwen Thinking模型在识别出公式和参数后，生成Python代码（如使用sympy或numpy），然后在沙箱环境中执行代码并返回结果。

### 优势与劣势

**优势：**
- 高度智能与灵活：可以理解复杂的多步推理和上下文依赖
- 简化流程：LLM可以同时处理公式识别、参数抽取和问答生成
- 自然语言交互好：问答体验更自然流畅，能生成详细的解释

**劣势：**
- 计算精度难以保证：LLM在处理浮点运算、高精度计算方面容易出错
- 成本高昂与延迟：调用大型闭源模型的成本和延迟较高
- 计算验证困难：计算过程是黑箱，难以提供可信的计算步骤溯源

### 实施步骤

#### 步骤1：PDF公式解析
将文档中的公式进行结构解析，并生成对应的LaTeX公式。

#### 步骤2：RAG检索和排序
用户的提问与公式进行相似度计算，可加入rerank过程，选择得到top1-8待选公式。

#### 步骤3：LLM推理与计算
- 使用qwen-3 thinking模型，输入用户提问 + 公式latex，生成对应的代码或sympy，并执行代码，返回结果
- 使用qwen-3 thinking模型，直接推理得到答案

---

## 三个解决方案详细分析


### 方案1（solu1）：纯向量检索 + 大模型QA

#### 技术栈
- **PDF解析**：dots.ocr VLM模型（通过vllm推理）
- **向量检索**：Qwen3-Embedding-0.6B
- **问答模型**：Qwen3-32B / Qwen3-235B-A22B-Thinking-2507

#### 实现流程

**1. PDF数据处理**
```python
# 使用dots.ocr批量转换PDF为Markdown
for file in ./xfdata/*.pdf; do 
    python3 dots_ocr/parser.py "$file" 
done

# 合并分页的Markdown文件
python data_prepare.py
```

核心代码逻辑（data_prepare.py）：
- 遍历每个子目录，找到所有`page_x.md`文件
- 按页码自然排序（使用正则提取数字）
- 将所有页面内容合并为单个文档
- 输出为Excel格式，每个PDF一行

**2. 向量检索与问答**

核心流程（main.py）：
```python
# 1. 加载文档内容
md_data = pd.read_excel("./user_data/tmp_data/md.xlsx")
contents.extend(md_data["text"].values)

# 2. 使用Qwen3-Embedding进行向量化
embedder = SentenceTransformer('Qwen3-Embedding-0.6B')
corpus_embeddings = embedder.encode_document(corpus, convert_to_tensor=True)

# 3. 对每个问题进行检索
query_embedding = embedder.encode_query(query, convert_to_tensor=True)
similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
scores, indices = torch.topk(similarity_scores, k=8)

# 4. 构建Prompt并调用LLM
template = '''
你是一位理工科的博士，下面给出问题query和多个参考公式列表，需要你给出最终的计算结果。
具体做法：
1. 参考公式存在多个，但真正用来计算的最多只有一个
2. 需要根据query选择合适的公式，确保计算公式的因变量都在query里提供
3. 如果缺失1-2个因变量，可以用常识估计值
4. 计算结果需要数值（小数或整数），不要无理数、分数
5. 以JSON格式输出，例如 {"answer": "100"}
...
'''
```

**3. 后处理策略**
- 针对无答案的问题，统一回复常数值10
- 修改大模型预测明显错误的地方
- 处理单位换算问题（如"万元"需要转换）
- 使用正则表达式提取JSON中的answer字段

#### 关键技术点

1. **向量检索优化**
   - 使用Qwen3-Embedding而非BGE，精度更高但速度较慢
   - Top-K设置为8，平衡召回率和精度

2. **Prompt工程**
   - 明确指定输出格式（JSON）
   - 提供详细的计算规则和边界情况处理
   - 要求模型给出计算公式、过程和结果

3. **错误处理**
   - 对于解析失败的情况，返回默认值10
   - 使用正则表达式多层次提取答案

#### 改进空间
- 优化prompt，减少badcase
- 调整召回策略，对问题进行分类标签化
- 增加重排步骤（虽然作者尝试后未见提升）

#### 得分情况
- 纯代码预测：2482.52分
- 修改badcase后：476分

---


### 方案2（solu2）：文本匹配 + 本地LLM

#### 技术栈
- **文档解析**：PyPDFLoader（LangChain）
- **问答模型**：Qwen2.5-7B（本地部署）
- **匹配策略**：基于文本匹配的知识关联

#### 实现流程

**1. 文档内容提取（read_pdf_or_md.py）**
```python
# 遍历文件夹，处理PDF和Markdown
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        full_text = "\n".join(doc.page_content for doc in documents)
    elif filename.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    
    records.append({"filename": filename, "content": full_text})

# 保存为CSV
df = pd.DataFrame(records)
df.to_csv("output.csv", index=False)
```

**2. 问题与知识匹配**
- 基于文本匹配得到每个问题对应的知识
- 生成matched.csv文件，包含问题和对应的背景知识

**3. 本地LLM推理（code.py）**
```python
# 加载本地Qwen2.5-7B模型
model_name = "./qwen2.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 构建Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7
)

# 对每个问题进行推理
for question, knowledge in matched_pairs:
    prompt = f"""
    请仔细阅读以下数学知识，并用一个数字回答我的数学问题。
    只需要回答一个数字，可以是小数或多位数。
    如果无法计算但可以估计结果，就回复："估计是（估计结果）"
    不需要任何多余的解释或计算过程。
    
    我的问题是：{question}
    我的背景知识是：{knowledge}
    
    再次强调只要回答我不知道或者一个数或者估计结果作为回答。
    请注意其中"万元"等词语，在回答中要给出对应的阿拉伯数字。
    """
    
    # 使用Qwen的对话模板
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 生成回答
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
```

#### 关键技术点

1. **文本匹配策略**
   - 预先建立问题与知识的映射关系
   - 避免实时检索，提高推理效率

2. **Prompt设计**
   - 强调只返回数字，不要解释
   - 支持估计值的返回
   - 特别注意单位换算（万元等）

3. **本地部署优势**
   - 成本低，无API调用费用
   - 延迟可控
   - 数据隐私保护

#### 特点分析
- **简化流程**：通过预匹配减少实时检索开销
- **轻量级**：使用7B模型，对硬件要求相对较低
- **直接推理**：不生成代码，直接让模型计算

---


### 方案3（solu3）：RAG + Ollama + SymPy计算引擎

#### 技术栈
- **文档解析**：PyMuPDF（fitz）
- **向量检索**：sentence-transformers（shibing624/text2vec-base-chinese）
- **推理引擎**：Ollama + Qwen3-8B
- **符号计算**：SymPy

#### 实现流程

**阶段一：LLM推理（1_run_inference.py）**

**1. 构建知识库**
```python
def build_simple_knowledge_base():
    """从ZIP文件中解析所有PDF和Markdown文档，提取纯文本"""
    knowledge_base = []
    with zipfile.ZipFile(DOCUMENTS_ZIP_PATH, 'r') as zf:
        for doc_path in doc_files:
            if doc_path.endswith('.pdf'):
                # 使用PyMuPDF提取文本
                with fitz.open(stream=content_bytes, filetype="pdf") as doc:
                    for page in doc:
                        full_text += page.get_text() + "\n"
            elif doc_path.endswith('.md'):
                full_text = content_bytes.decode('utf-8')
            
            knowledge_base.append({"id": doc_id, "full_text": full_text})
    return knowledge_base
```

**2. 向量检索（Top-3）**
```python
# 加载中文句向量模型
embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
corpus_embeddings = torch.tensor(embed_model.encode(corpus_texts))

# 对每个问题检索Top-3相关文档
query_embedding = torch.tensor(embed_model.encode(query_text))
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_k_scores, top_k_indices = torch.topk(cos_scores, k=3)
```

**3. 结构化Prompt设计**
```python
prompt_template = """
你是一位极其严谨且聪明的科学助理。

**第一步：选择最佳文档**
仔细阅读用户问题和下面提供的所有"备选文档"。
选择出与用户问题**最相关**的那个文档。

**第二步：分析问题意图**
判断问题意图属于以下三种之一：
1. "numerical_calculation": 问题要求计算一个最终的数值
2. "formula_retrieval": 问题询问"如何"预测/评估，要求返回核心的数学模型
3. "conclusion_derivation": 问题询问"是否"会产生某种影响，要求基于模型进行逻辑推断

**第三步：严格按照JSON格式返回结果**
{{
  "thought_process": "描述完整思考步骤",
  "question_intent": "意图类型",
  "structured_data": {{
    "formula": "核心数学公式",
    "parameters": {{"变量名1": "数值1"}},
    "guessed_parameters": {{"缺失变量名": "猜测值", "说明": "猜测依据"}}
  }},
  "llm_answer": "最终答案"
}}

**第四步：关于"数值计算"的特殊指令**
- 检查参数完整性
- 仅在缺少少量参数且能做出合理猜测时，填充guessed_parameters
- 如果无法猜测，返回 "Unable to determine"
"""
```

**4. SymPy精确计算**
```python
def calculate_with_sympy(formula_string: str, params: dict) -> Union[float, None]:
    """使用SymPy对公式和参数进行精确计算"""
    try:
        # 处理LaTeX公式
        if '\\' in formula_string:
            formula_string = formula_string.replace('\\cdot', '*')
            expr = parse_latex(formula_string)
        else:
            # 处理普通公式
            formula_string = formula_string.replace('×', '*').replace('÷', '/')
            expr = sympify(formula_string)
        
        # 参数替换
        safe_params = {k: float(v) for k, v in params.items()}
        result = expr.subs(safe_params).evalf()
        return float(result)
    except Exception:
        return None
```

**5. 推理与计算结合**
```python
# 调用Ollama进行推理
response = ollama.chat(
    model='qwen3:8b',
    messages=[{'role': 'user', 'content': prompt}]
)
llm_raw_output = response['message']['content']

# 解析JSON响应
llm_json = json.loads(json_str)
intent = llm_json.get("question_intent")

# 根据意图处理
if intent == "numerical_calculation":
    formula = structured_data.get("formula")
    params = structured_data.get("parameters", {})
    guessed_params = structured_data.get("guessed_parameters", {})
    final_params = {**guessed_params, **params}
    
    # 使用SymPy计算
    sympy_result = calculate_with_sympy(formula, final_params)
    
    if sympy_result is not None:
        final_answer = sympy_result
    else:
        # 回退到LLM的答案
        final_answer = float(llm_answer)
```

**阶段二：后处理（2_generate_submission.py）**

```python
def intelligent_answer_cleaning(log_record: dict) -> float:
    """将复杂的LLM回答智能地映射为单一的数值"""
    
    final_answer = log_record.get("final_answer")
    
    # 规则1: 如果已经是数字，直接采纳
    if isinstance(final_answer, (int, float)):
        return float(final_answer)
    
    # 规则2: 解析意图进行映射
    intent = llm_json.get("question_intent")
    llm_answer_text = str(llm_json.get("llm_answer", "")).lower()
    
    # 规则3: 根据意图映射
    if intent == "formula_retrieval":
        # 公式检索：返回1.0表示成功识别
        if "unable to determine" in str(final_answer).lower():
            return 0.0
        return 1.0
    
    elif intent == "conclusion_derivation":
        # 结论推导：根据关键词映射
        positive_keywords = ["是", "会", "出现", "正面", "增长"]
        negative_keywords = ["否", "不会", "负面", "减少"]
        
        if any(kw in llm_answer_text for kw in positive_keywords):
            return 1.0
        elif any(kw in llm_answer_text for kw in negative_keywords):
            return -1.0
        else:
            return 0.0
    
    # 规则4: 其他情况返回安全值
    return 0.0
```

#### 关键技术点

1. **分阶段处理**
   - 第一阶段：LLM推理，生成详细日志
   - 第二阶段：后处理，将复杂回答映射为数值

2. **意图识别**
   - numerical_calculation：数值计算
   - formula_retrieval：公式检索
   - conclusion_derivation：结论推导

3. **混合计算策略**
   - 优先使用SymPy进行精确计算
   - SymPy失败时回退到LLM的答案
   - 支持参数猜测机制

4. **鲁棒性设计**
   - JSON解析失败时启动正则抢救模式
   - 多层次的错误处理
   - 安全的兜底值（0.0）

5. **本地部署**
   - 使用Ollama管理模型
   - 无需API调用，成本低
   - 可离线运行

#### 优势分析
- **精确计算**：使用SymPy保证计算精度
- **结构化输出**：强制JSON格式，便于解析
- **意图驱动**：根据不同意图采用不同策略
- **可追溯性**：详细的日志记录，便于调试

---


## 技术路线1的核心实现思路总结

### 整体架构

```
用户提问
    ↓
PDF/MD文档解析（提取文本和公式）
    ↓
向量化（Embedding）
    ↓
相似度检索（Top-K）
    ↓
构建Prompt（问题 + 候选公式）
    ↓
LLM推理（Qwen Thinking）
    ↓
结果提取与后处理
    ↓
最终答案
```

### 三个方案的对比

| 维度 | 方案1 | 方案2 | 方案3 |
|------|-------|-------|-------|
| **PDF解析** | dots.ocr VLM | PyPDFLoader | PyMuPDF |
| **向量模型** | Qwen3-Embedding-0.6B | 无（预匹配） | text2vec-base-chinese |
| **LLM** | Qwen3-235B（API） | Qwen2.5-7B（本地） | Qwen3-8B（Ollama） |
| **Top-K** | 8 | 预匹配 | 3 |
| **计算方式** | LLM直接推理 | LLM直接推理 | SymPy + LLM |
| **输出格式** | JSON | 纯数字 | 结构化JSON |
| **后处理** | 正则提取 + 规则修正 | 直接使用 | 意图驱动映射 |
| **部署方式** | API调用 | 本地部署 | 本地部署（Ollama） |
| **成本** | 高 | 低 | 低 |
| **精度保证** | 依赖LLM | 依赖LLM | SymPy精确计算 |

### 关键技术要点

#### 1. PDF公式解析

**挑战：**
- PDF中的公式可能是图片或特殊编码
- 需要转换为LaTeX或可计算的格式

**解决方案：**
- 方案1：使用VLM模型（dots.ocr）进行视觉理解
- 方案2/3：使用文本提取工具（PyPDFLoader/PyMuPDF）

#### 2. RAG检索策略

**核心流程：**
```python
# 1. 文档向量化
corpus_embeddings = embedder.encode_document(corpus)

# 2. 问题向量化
query_embedding = embedder.encode_query(query)

# 3. 相似度计算
similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)

# 4. Top-K选择
scores, indices = torch.topk(similarity_scores, k=top_k)
```

**优化点：**
- 选择合适的Embedding模型（Qwen3-Embedding vs text2vec）
- 调整Top-K值（3-8之间）
- 可选：加入Rerank步骤

#### 3. Prompt工程

**设计原则：**
- 明确任务目标（计算数值、检索公式、推导结论）
- 指定输出格式（JSON）
- 提供详细的规则和边界情况处理
- 支持参数缺失时的估计机制

**示例结构：**
```
角色定义 + 任务描述
    ↓
步骤分解（选择文档 → 分析意图 → 提取信息 → 计算）
    ↓
输出格式要求（JSON Schema）
    ↓
特殊情况处理（参数缺失、无法计算等）
```

#### 4. LLM推理与计算

**方案1/2：直接推理**
- 优点：简单直接，一步到位
- 缺点：计算精度依赖LLM，可能出错

**方案3：混合计算**
```python
# 1. LLM提取公式和参数
llm_output = llm.invoke(prompt)
formula = extract_formula(llm_output)
params = extract_parameters(llm_output)

# 2. SymPy精确计算
result = sympy_calculate(formula, params)

# 3. 失败时回退到LLM答案
if result is None:
    result = extract_llm_answer(llm_output)
```


## 总结

技术路线1（RAG + Qwen Thinking）的核心是利用大模型的理解能力来处理复杂的公式问答任务。三个方案展示了不同的实现策略：

- **方案1**：追求高精度，使用大模型（235B）和高质量Embedding
- **方案2**：追求简单高效，预匹配 + 本地小模型
- **方案3**：追求精确计算，混合SymPy + 结构化输出

每个方案都有其适用场景，实际应用中可以根据需求（精度、成本、延迟）选择合适的方案，或者结合多个方案的优点进行改进。

关键是要理解：**LLM擅长理解和推理，但不擅长精确计算**。因此，最佳实践是让LLM负责"理解问题、提取信息"，让专业工具（SymPy、NumPy）负责"精确计算"。
