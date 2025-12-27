import asyncio
import os
import json
import requests
import urllib.parse
from typing import List, Dict, Any

# 设置环境变量（如果尚未设置）
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
if "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 假设 agents 库可用
try:
    from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
        set_default_openai_api, set_tracing_disabled
    
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
except ImportError:
    print("Warning: 'agents' library not found. Code may not run without it.")
    # Mocking for syntax check if library is missing
    class Agent: pass
    class AsyncOpenAI: pass
    class OpenAIChatCompletionsModel: pass
    class ModelSettings: pass
    class Runner: pass
    def set_default_openai_api(x): pass
    def set_tracing_disabled(x): pass

# --- 配置 ---
MODEL_NAME = "qwen-max"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"

# 初始化客户端
llm_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
external_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 工具函数 (Jina Search & Crawl) ---

def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    # print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}&hl=zh-cn"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "no-content"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        results = response.json().get('data', [])
        formatted_results = []
        for res in results:
            formatted_results.append({
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "snippet": res.get("content", "")
            })
        return json.dumps(formatted_results, ensure_ascii=False)
    except Exception as e:
        print(f"Error during Jina Search: {e}")
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)

def crawl_jina(url: str) -> str:
    """通过jina抓取完整网页内容"""
    # print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "content",
            "X-Content-Type": "markdown"
        }
        response = requests.get("https://r.jina.ai/" + url, headers=headers, timeout=20)
        response.raise_for_status()
        content = response.json().get("data", {}).get("content", f"无法抓取 URL: {url} 的内容。")
        return content
    except Exception as e:
        print(f"Error during Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"

async def async_search_jina(query: str) -> str:
    return await asyncio.to_thread(search_jina, query)

async def async_crawl_jina(url: str) -> str:
    return await asyncio.to_thread(crawl_jina, url)

# --- Agent 定义 ---

# 1. Orchestrator (大纲与整合)
orchestrator_system_prompt = """
你是一名深度研究专家和项目经理。你的任务是：
1. **研究规划:** 根据主题生成详尽的大纲（JSON格式）。
2. **报告整合:** 将各章节整合成最终报告。
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

# 2. Drafting Agent (起草与修改)
drafting_system_prompt = """
你是一名专业的内容撰稿人。你的任务是根据提供的资料撰写或修改报告章节。
要求：
- 严格围绕主题。
- 基于提供的资料。
- 使用 Markdown 格式。
- 必须包含引用来源。
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

# 3. Reviewer Agent (点评与反馈) - NEW
reviewer_system_prompt = """
你是一名严苛的主编和内容质量控制专家。你的任务是评估撰稿人提交的章节草稿。
请从以下维度进行评估：
1. **相关性**: 是否紧扣章节主题？
2. **完整性**: 信息是否详实？是否有遗漏关键点？
3. **准确性**: 是否有明显的逻辑错误？引用是否规范？
4. **可读性**: 结构是否清晰？

输出格式要求：
如果你认为草稿质量合格，请仅输出：PASS
如果你认为需要修改，请输出具体的修改建议（Critique），并在最后附上：REVISE
"""
ReviewerAgent = Agent(
    "Content Quality Reviewer",
    instructions=reviewer_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

# --- 核心流程函数 ---

async def process_section(section: Dict[str, Any], max_iterations: int = 3) -> str:
    """
    处理单个章节：检索 -> 抓取 -> (起草 -> 评估 -> 修改)循环
    """
    section_title = section.get("section_title")
    search_keywords = section.get("search_keywords")
    print(f"[{section_title}] 开始处理...")

    # 1. 检索与抓取 (Context Gathering)
    section_query = f"{section_title} {search_keywords}"
    search_results_str = await async_search_jina(section_query)
    
    try:
        search_results = json.loads(search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2] # 取前2个
    except:
        urls_to_crawl = []

    crawled_content = []
    # 并发抓取
    crawl_tasks = [async_crawl_jina(url) for url in urls_to_crawl]
    if crawl_tasks:
        crawled_results = await asyncio.gather(*crawl_tasks)
        for i, content in enumerate(crawled_results):
            crawled_content.append(f"--- Source: {urls_to_crawl[i]} ---\n{content[:2000]}...\n")
    
    raw_materials = "\n\n".join(crawled_content)
    context = f"**搜索摘要:**\n{search_results_str[:1000]}\n\n**详细内容:**\n{raw_materials}"

    # 2. 生成与反馈循环 (Generation & Feedback Loop)
    current_draft = ""
    feedback = ""
    
    for i in range(max_iterations):
        print(f"[{section_title}] Iteration {i+1}/{max_iterations}")
        
        # --- Phase A: Drafting / Revising ---
        if i == 0:
            # 初次起草
            draft_prompt = f"""
            **任务**: 撰写章节 "{section_title}"
            **参考资料**:
            {context}
            
            请撰写本章节内容。
            """
            response = await Runner.run(DraftingAgent, draft_prompt)
            current_draft = response.final_output
        else:
            # 根据反馈修改
            revise_prompt = f"""
            **任务**: 修改章节 "{section_title}"
            
            **上一版草稿**:
            {current_draft}
            
            **主编反馈意见**:
            {feedback}
            
            **参考资料**:
            {context}
            
            请根据反馈意见重新撰写本章节，确保解决所有指出的问题。
            """
            response = await Runner.run(DraftingAgent, revise_prompt)
            current_draft = response.final_output

        # --- Phase B: Reviewing ---
        # 如果是最后一次迭代，就不需要 Review 了，直接作为最终结果
        if i == max_iterations - 1:
            print(f"[{section_title}] 达到最大迭代次数，停止优化。")
            break

        review_prompt = f"""
        **待评审章节**: {section_title}
        
        **草稿内容**:
        {current_draft}
        
        请进行评审。合格回复 PASS，不合格回复建议并以 REVISE 结尾。
        """
        review_response = await Runner.run(ReviewerAgent, review_prompt)
        review_result = review_response.final_output

        if "PASS" in review_result:
            print(f"[{section_title}] 评审通过！")
            break
        else:
            feedback = review_result.replace("REVISE", "").strip()
            print(f"[{section_title}] 评审未通过，意见: {feedback[:100]}...")

    print(f"[{section_title}] 完成。")
    return f"## {section_title}\n\n{current_draft}"


async def deep_research(query: str, max_sections: int = 5) -> str:
    print(f"\n--- Deep Research Started: {query} ---\n")

    # Step 1: 初步检索
    print("Step 1: 初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # Step 2: 生成大纲
    print("Step 2: 生成大纲...")
    outline_prompt = f"""研究主题: {query}
初步搜索结果: {initial_search_results_str[:2000]}
生成详细的报告大纲 JSON，包含 'title' 和 'sections' (含 'section_title', 'search_keywords')。
"""
    try:
        outline_response = await Runner.run(DeepResearchAgent, outline_prompt)
        # 简单的 JSON 提取逻辑
        json_str = outline_response.final_output
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
            
        outline_json = json.loads(json_str.strip())
    except Exception as e:
        print(f"大纲生成出错: {e}, 使用默认结构。")
        outline_json = {
            "title": f"{query} 研究报告",
            "sections": [
                {"section_title": "背景", "search_keywords": query},
                {"section_title": "核心技术", "search_keywords": f"{query} technology"},
                {"section_title": "应用", "search_keywords": f"{query} application"}
            ]
        }

    research_title = outline_json.get("title", query)
    sections = outline_json.get("sections", [])[:max_sections]
    print(f"报告标题: {research_title}, 包含 {len(sections)} 个章节。")

    # Step 3: 并发生成所有章节 (Concurrent Generation with ReAct)
    print("\nStep 3: 并发生成章节 (ReAct Loop enabled)...")
    
    # 创建所有章节的任务
    section_tasks = [process_section(section) for section in sections]
    
    # 等待所有任务完成
    drafted_sections = await asyncio.gather(*section_tasks)

    # Step 4: 整合报告
    print("\nStep 4: 整合最终报告...")
    full_body = "\n\n".join(drafted_sections)
    
    final_prompt = f"""
    整合以下章节为完整报告：
    标题: {research_title}
    
    章节内容:
    {full_body}
    
    要求: 添加摘要、结论、引用列表，保持格式优美。
    """
    
    final_report_response = await Runner.run(DeepResearchAgent, final_prompt)
    return final_report_response.final_output

async def main():
    topic = "Agentic AI在软件开发中的最新应用和挑战"
    report = await deep_research(topic)
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "DeepResearch_Report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存至: {output_path}")
    print(report[:500] + "...")

if __name__ == "__main__":
    asyncio.run(main())
