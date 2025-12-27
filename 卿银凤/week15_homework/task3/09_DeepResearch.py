import asyncio
import time
import os
os.environ["OPENAI_API_KEY"] = "sk-be3c9c14e12046f59f6e0c5f9bad8fbf"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import List, Dict, Any, Tuple

from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"
API_KEY = os.getenv("OPENAI_API_KEY", "sk-be3c9c14e12046f59f6e0c5f9bad8fbf")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

llm_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

model_settings = ModelSettings(model=MODEL_NAME, client=llm_client, temperature=0.3)

# --- 外部工具（Jina Search & Crawl） ---
JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"


def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
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
    """通过jina抓取完整网页内容，返回Markdown格式的文本"""
    print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
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


external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 代理定义 ---
orchestrator_system_prompt = """
你是一名深度研究专家和项目经理。你的任务是协调整个研究项目，包括：
1. **研究规划 (生成大纲):** 根据用户提供的研究主题和初步搜索结果，生成一个详尽、逻辑严密、结构清晰的报告大纲。
2. **报告整合 (组装):** 在所有章节内容起草完成后，将它们整合在一起，形成一篇流畅、连贯、格式优美的最终研究报告。
"""

DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

drafting_system_prompt = """
你是一名专业的内容撰稿人。你的任务是将提供的原始网页抓取内容和搜索结果，根据指定的章节主题，撰写成一篇结构合理、重点突出、信息准确的报告章节。
你必须严格遵守以下规则：
1. **聚焦主题:** 严格围绕给定的 '章节主题' 进行撰写。
2. **信息来源:** 只能使用提供的 '原始网页内容' 和 '搜索结果摘要' 中的信息。
3. **格式:** 使用 Markdown 格式。
4. **引用:** 对于文中引用的关键事实和数据，必须在段落末尾用脚注或括号标记引用的来源 URL。
"""

DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

# ReAct 评估代理
react_evaluator_prompt = """
你是一名严格的内容质量评估专家，采用 ReAct (Reasoning + Acting) 模式进行评估。

对于每个章节内容，你需要：
1. **Thought (思考):** 分析章节内容的优缺点
2. **Observation (观察):** 指出具体问题所在
3. **Action (行动建议):** 给出明确的改进建议

评估维度：
- 内容完整性：是否覆盖了章节主题的关键点
- 逻辑连贯性：段落之间是否有清晰的逻辑关系
- 信息准确性：是否有明显的事实错误或臆测
- 引用规范性：是否正确标注了信息来源
- 专业深度：分析是否足够深入

请以JSON格式输出评估结果：
{
    "score": 1-10的评分,
    "thought": "你的思考过程",
    "observation": "观察到的具体问题",
    "action": "具体的改进建议",
    "pass": true/false (score >= 7 为 true)
}
只输出JSON，不要其他内容。
"""

ReactEvaluatorAgent = Agent(
    "ReAct Quality Evaluator",
    instructions=react_evaluator_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)

# 内容优化代理
refiner_system_prompt = """
你是一名内容优化专家。根据评估反馈，对章节内容进行针对性的改进和优化。
你需要：
1. 保留原文中的优秀部分
2. 根据反馈建议进行针对性修改
3. 确保修改后的内容更加完整、准确、专业
4. 保持 Markdown 格式和引用规范
"""

RefinerAgent = Agent(
    "Content Refiner",
    instructions=refiner_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client),
)


# --- ReAct 章节生成流程 ---
async def react_generate_section(
    section_title: str,
    search_keywords: str,
    section_index: int,
    max_iterations: int = 3
) -> Tuple[str, List[Dict]]:
    """
    使用 ReAct 机制生成单个章节，包含评估-反馈-优化循环
    返回: (最终章节内容, 迭代历史记录)
    """
    print(f"\n{'='*60}")
    print(f"[章节 {section_index}] 开始并行生成: {section_title}")
    print(f"{'='*60}")
    
    iteration_history = []
    
    # Step 1: 检索
    print(f"[章节 {section_index}] Step 1: 精确检索...")
    section_query = f"{section_title} {search_keywords}"
    section_search_results_str = await async_search_jina(section_query)
    
    # Step 2: 抓取网页内容
    print(f"[章节 {section_index}] Step 2: 抓取网页内容...")
    try:
        search_results = json.loads(section_search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        urls_to_crawl = []
    
    # 并行抓取多个URL
    crawl_tasks = [async_crawl_jina(url) for url in urls_to_crawl]
    crawled_contents = await asyncio.gather(*crawl_tasks, return_exceptions=True)
    
    raw_materials = ""
    for i, (url, content) in enumerate(zip(urls_to_crawl, crawled_contents)):
        if isinstance(content, Exception):
            content = f"抓取失败: {content}"
        raw_materials += f"--- URL: {url} ---\n{str(content)[:3000]}...\n\n"
    
    # Step 3: 初始起草
    print(f"[章节 {section_index}] Step 3: 初始内容起草...")
    draft_prompt = f"""
**章节主题:** {section_title}
**搜索关键词:** {search_keywords}

**搜索结果摘要:**
{section_search_results_str[:3000]}

**原始网页内容:**
{raw_materials}

请根据上述信息，撰写 "{section_title}" 这一章节的详细内容。
要求：内容充实、逻辑清晰、有数据支撑、标注引用来源。
"""
    
    try:
        draft_response = await Runner.run(DraftingAgent, draft_prompt)
        current_draft = draft_response.final_output
    except Exception as e:
        return f"## {section_title}\n\n章节生成失败: {e}", []
    
    # Step 4: ReAct 迭代优化循环
    for iteration in range(max_iterations):
        print(f"\n[章节 {section_index}] ReAct 迭代 {iteration + 1}/{max_iterations}")
        
        # Thought + Observation: 评估当前内容
        eval_prompt = f"""
请评估以下章节内容的质量：

**章节主题:** {section_title}
**章节内容:**
{current_draft}

请按照 ReAct 模式进行评估，输出JSON格式的评估结果。
"""
        
        try:
            eval_response = await Runner.run(ReactEvaluatorAgent, eval_prompt)
            eval_text = eval_response.final_output.strip()
            # 清理可能的markdown代码块标记
            eval_text = eval_text.strip("```json").strip("```").strip()
            evaluation = json.loads(eval_text)
        except Exception as e:
            print(f"[章节 {section_index}] 评估解析失败: {e}")
            evaluation = {"score": 7, "pass": True, "thought": "评估失败，使用当前版本", 
                         "observation": "", "action": ""}
        
        iteration_record = {
            "iteration": iteration + 1,
            "score": evaluation.get("score", 0),
            "thought": evaluation.get("thought", ""),
            "observation": evaluation.get("observation", ""),
            "action": evaluation.get("action", ""),
            "passed": evaluation.get("pass", False)
        }
        iteration_history.append(iteration_record)
        
        print(f"  -> Thought: {evaluation.get('thought', '')[:100]}...")
        print(f"  -> Score: {evaluation.get('score', 'N/A')}/10")
        print(f"  -> Pass: {evaluation.get('pass', False)}")
        
        # 如果通过评估，结束迭代
        if evaluation.get("pass", False):
            print(f"[章节 {section_index}] ✓ 章节质量达标，完成生成")
            break
        
        # Action: 根据反馈优化内容
        if iteration < max_iterations - 1:
            print(f"[章节 {section_index}] 根据反馈进行优化...")
            refine_prompt = f"""
**原始章节内容:**
{current_draft}

**评估反馈:**
- 思考: {evaluation.get('thought', '')}
- 观察到的问题: {evaluation.get('observation', '')}
- 改进建议: {evaluation.get('action', '')}

**原始素材 (可补充引用):**
{raw_materials[:2000]}

请根据上述反馈，优化章节内容。保留优秀部分，针对性改进问题。
"""
            try:
                refine_response = await Runner.run(RefinerAgent, refine_prompt)
                current_draft = refine_response.final_output
                print(f"[章节 {section_index}] 优化完成，进入下一轮评估")
            except Exception as e:
                print(f"[章节 {section_index}] 优化失败: {e}，保留当前版本")
                break
    
    final_content = f"## {section_title}\n\n{current_draft}"
    print(f"[章节 {section_index}] 章节生成完成，共迭代 {len(iteration_history)} 次")
    
    return final_content, iteration_history


async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 并行检索/生成 -> ReAct优化 -> 整合
    """
    print(f"\n{'#'*70}")
    print(f"# Deep Research (并行生成 + ReAct机制): {query}")
    print(f"{'#'*70}\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # 2. 生成研究大纲
    print("\nStep 2: 基于初步结果生成研究大纲...")
    outline_prompt = f"""研究主题: {query}
初步搜索结果摘要: {initial_search_results_str}

请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords' (用于精确检索的关键词)。

示例输出 JSON 格式如下，只要json，不要有其他输出
{{
    "title": "关于 XX 的深度研究报告",
    "sections": [
        {{"section_title": "引言与背景", "search_keywords": "历史, 现状"}},
        {{"section_title": "核心要素与机制", "search_keywords": "关键概念, 工作原理"}},
        {{"section_title": "应用与影响", "search_keywords": "行业应用, 社会影响"}}
    ]
}}
"""
    try:
        outline_response = await Runner.run(DeepResearchAgent, outline_prompt)
        outline_text = outline_response.final_output.strip("```json").strip("```")
        outline_json = json.loads(outline_text)
    except Exception as e:
        print(f"Error generating outline: {e}. Using default structure.")
        outline_json = {
            "title": f"关于 {query} 的深度研究报告",
            "sections": [
                {"section_title": "引言与背景", "search_keywords": f"{query}, 历史, 现状"},
                {"section_title": "核心技术与原理", "search_keywords": f"{query}, 工作原理, 关键技术"},
                {"section_title": "应用场景与案例", "search_keywords": f"{query}, 行业应用, 实践案例"},
                {"section_title": "挑战与展望", "search_keywords": f"{query}, 发展趋势, 挑战"}
            ]
        }

    research_title = outline_json.get("title", f"关于 {query} 的深度研究报告")
    sections = outline_json.get("sections", [])[:max_sections]

    print(f"\n报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节，将并行生成...")
    for i, s in enumerate(sections):
        print(f"  {i+1}. {s.get('section_title')}")

    # 3. 并行生成所有章节 (每个章节内部使用ReAct机制)
    print(f"\nStep 3: 并行生成 {len(sections)} 个章节 (ReAct模式)...")
    
    section_tasks = [
        react_generate_section(
            section_title=section.get("section_title"),
            search_keywords=section.get("search_keywords"),
            section_index=i + 1,
            max_iterations=3
        )
        for i, section in enumerate(sections)
    ]
    
    start = time.time()
    # 并行执行所有章节生成任务
    results = await asyncio.gather(*section_tasks, return_exceptions=True)
    print(f"并发执行: {time.time()-start:.2f}秒")
    print(f"所有结果: {results}")
    
    # 收集结果
    drafted_sections = []
    all_iteration_history = {}
    
    for i, result in enumerate(results):
        section_title = sections[i].get("section_title")
        if isinstance(result, Exception):
            drafted_sections.append(f"## {section_title}\n\n章节生成失败: {result}")
            all_iteration_history[section_title] = []
        else:
            content, history = result
            drafted_sections.append(content)
            all_iteration_history[section_title] = history

    # 打印 ReAct 迭代统计
    print(f"\n{'='*70}")
    print("ReAct 迭代统计:")
    print(f"{'='*70}")
    for section_title, history in all_iteration_history.items():
        if history:
            final_score = history[-1].get('score', 'N/A')
            iterations = len(history)
            print(f"  - {section_title}: {iterations}次迭代, 最终评分: {final_score}/10")

    # 4. 报告整合
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    final_prompt = f"""
请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

**报告标题:** {research_title}

**已起草的章节内容:**
{full_report_draft}

**任务要求:**
1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
2. 保持各章节之间的连贯性，添加适当的过渡语句。
3. 在报告末尾添加一个**【结论与展望】**部分（如果大纲中没有）。
4. 添加一个**【引用来源】**列表，列出所有章节中提到的 URL。
5. 整体报告必须格式优美，使用 Markdown 格式。
"""

    try:
        final_report = await Runner.run(DeepResearchAgent, final_prompt)
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"


async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic)
    print("\n" + "="*70)
    print("最终研究报告")
    print("="*70)
    print(final_report)


if __name__ == "__main__":
    asyncio.run(main())
