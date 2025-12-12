import asyncio
import os
os.environ["OPENAI_API_KEY"] = "sk-ea07bf0880504b75a31b1bce38437fcf"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import List, Dict, Any

# 假设以下导入能够正常工作，它们通常来自 agents 库
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"  # 假设这是 AliCloud 兼容模式下的一个模型名称
API_KEY = os.getenv("OPENAI_API_KEY", "sk-ea07bf0880504b75a31b1bce38437fcf")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 AsyncOpenAI 客户端
llm_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 定义模型设置
model_settings = ModelSettings(
    model=MODEL_NAME,
    client=llm_client,
    temperature=0.3
)

# --- 2. 外部工具（Jina Search & Crawl） ---
JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"

def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
    try:
        # 确保查询参数是 URL 编码的
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}&hl=zh-cn"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "no-content"  # Jina Search 默认返回摘要和引用
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # 抛出 HTTP 错误

        # Jina Search 返回的是一个包含结果的 JSON 结构，提取关键信息
        results = response.json().get('data', [])

        # 提取标题、链接和摘要
        formatted_results = []
        for res in results:
            formatted_results.append({
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "snippet": res.get("content", "")
            })

        return json.dumps(formatted_results, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        print(f"Error during Jina Search: {e}")
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)
    except Exception as e:
        print(f"Unexpected error in Jina Search: {e}")
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


async def crawl_jina(url: str) -> str:
    """通过jina抓取完整网页内容，返回Markdown格式的文本"""
    print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
    try:
        # Jina Reader API
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "content",  # 请求返回完整内容
            "X-Content-Type": "markdown"  # 请求返回 Markdown 格式
        }
        # 使用 r.jina.ai 作为代理
        response = requests.get("https://r.jina.ai/" + url, headers=headers, timeout=20)
        response.raise_for_status()

        # 返回内容通常在 'data' 字段的 'content' 中
        content = response.json().get("data", {}).get("content", f"无法抓取 URL: {url} 的内容。")

        return content
    except requests.exceptions.RequestException as e:
        print(f"Error during Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"
    except Exception as e:
        print(f"Unexpected error in Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"


# 将同步函数包装成异步，以便在 Agents 异步环境中使用
async def async_search_jina(query: str) -> str:
    """异步调用 Jina 搜索"""
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    """异步调用 Jina 抓取"""
    return await asyncio.to_thread(crawl_jina, url)

REVIEW_CRITERIA = """
请严格按以下标准评估章节内容质量：
1. 【完整性】是否覆盖章节主题核心要点？有无重大遗漏？
2. 【准确性】关键事实、数据、术语是否与原始材料一致？有无捏造？
3. 【逻辑性】结构是否清晰（引言-主体-小结）？段落衔接是否自然？
4. 【语言规范】是否用语专业、简洁、无口语化/重复？有无语法错误？
5. 【原创性】是否避免直接复制原文？是否合理改写与整合？

请输出 JSON 格式结果：
{
  "is_acceptable": true/false,
  "score": 0~10,
  "issues": ["问题1", "问题2", ...],
  "suggestions": ["建议1", "建议2", ...],
  "missing_keywords": ["应提及但缺失的关键词", ...]
}
"""

external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 3. 代理定义 (Agents) ---
orchestrator_system_prompt = """
你是一名深度研究专家和项目经理。你的任务是协调整个研究项目，包括：
1. **研究规划 (生成大纲):** 根据用户提供的研究主题和初步搜索结果，生成一个详尽、逻辑严密、结构清晰的报告大纲。大纲必须以严格的 JSON 格式输出，用于指导后续的章节内容检索和起草工作。
2. **报告整合 (组装):** 在所有章节内容起草完成后，将它们整合在一起，形成一篇流畅、连贯、格式优美的最终研究报告。报告必须包括摘要、完整的章节内容、结论和引用来源列表。
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

ReviewAgent = Agent(
    "Research Report Reviewer",
    instructions=REVIEW_CRITERIA,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# 3.2. 内容起草代理 (Drafting Agent)
drafting_system_prompt = """
你是一名专业的内容撰稿人。你的任务是将提供的原始网页抓取内容和搜索结果，根据指定的章节主题，撰写成一篇结构合理、重点突出、信息准确的报告章节。
你必须严格遵守以下规则：
1. **聚焦主题:** 严格围绕给定的 '章节主题' 进行撰写。
2. **信息来源:** 只能使用提供的 '原始网页内容' 和 '搜索结果摘要' 中的信息。
3. **格式:** 使用 Markdown 格式。
4. **引用:** 对于文中引用的关键事实和数据，必须在段落末尾用脚注或括号标记引用的来源 URL，例如 [来源: URL]。
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

async def review_and_revise(
    section_title: str,
    draft: str,
    raw_materials: str,
    search_summary: str,
    max_retries: int = 2
) -> str:
    current_draft = draft
    for attempt in range(max_retries + 1):
        # —— Step: Review ——
        review_prompt = f"""
你是一名专业的内容质量审核专家。

**章节主题：** {section_title}

**当前草稿：**
{current_draft}

**原始参考材料（用于核对事实）：**
{raw_materials[:4000]}...

**初检搜索摘要（辅助判断覆盖度）：**
{search_summary[:2000]}...

{REVIEW_CRITERIA}
请严格按 JSON 格式输出，不要任何额外文本。
"""

        try:
            review_result = await Runner.run(
                ReviewAgent,  # ← 假设你已定义 ReviewAgent
                review_prompt,
            )
            feedback = json.loads(review_result.final_output.strip("```json\n").strip("\n```"))
        except Exception as e:
            print(f"⚠️ [{section_title}] Review 解析失败，跳过审核: {e}")
            return current_draft  # 容错：审核失败则接受当前稿

        is_ok = feedback.get("is_acceptable", False)
        score = feedback.get("score", 0)
        issues = feedback.get("issues", [])
        suggestions = feedback.get("suggestions", [])

        print(f"🔍 [{section_title}] 审核 #{attempt + 1} | 分数: {score}/10 | 通过: {is_ok}")
        if issues:
            print(f"   ❗ 问题: {'; '.join(issues[:3])}")

        if is_ok or attempt == max_retries:
            # 接受终稿（即使最后一次不合格，也终止循环）
            return current_draft

        # —— Step: Revise ——
        revise_prompt = f"""
你是一名资深内容编辑。请根据审核反馈修改章节草稿。

**章节主题：** {section_title}

**当前草稿（需修改）：**
{current_draft}

**审核反馈：**
- 问题：{issues}
- 建议：{suggestions}
- 缺失关键词：{feedback.get('missing_keywords', [])}

**修改要求：**
1. 保留原意，仅提升质量；
2. 补充缺失要点，修正事实错误；
3. 优化逻辑流与语言表达；
4. 输出完整修订版，不要解释。

请直接输出修订后的章节正文（Markdown 格式）。
"""

        try:
            revised = await Runner.run(
                DraftingAgent,  # 可复用起草 Agent，或定义 ReviseAgent
                revise_prompt,
            )
            current_draft = revised.final_output
            print(f"✏️ [{section_title}] 已修订（第 {attempt + 1} 次）")
        except Exception as e:
            print(f"⚠️ [{section_title}] 修订失败，保留上一版: {e}")
            break  # 修订失败则终止循环

    return current_draft

async def process_section(section: dict) -> str:
    section_title = section.get("section_title", "Untitled")
    search_keywords = section.get("search_keywords", "")
    print(f"\n🚀 [{section_title}] 启动处理...")

    # 1. 检索 + 抓取（同前）
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    section_search_results_str = await async_search_jina(section_query)

    urls_to_crawl = []
    try:
        search_results = json.loads(section_search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except Exception as e:
        print(f"⚠️ [{section_title}] 解析搜索结果失败: {e}")

    crawl_tasks = [async_crawl_jina(url) for url in urls_to_crawl]
    crawl_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)

    crawled_content = []
    for url, content in zip(urls_to_crawl, crawl_results):
        if isinstance(content, Exception):
            crawled_content.append(f"--- URL: {url} ---\n[抓取失败]\n")
        else:
            crawled_content.append(f"--- URL: {url} ---\n{str(content)[:3000]}...\n")
    raw_materials = "\n\n".join(crawled_content)

    # 2. 初稿生成
    draft_prompt = f"""
    **章节主题:** {section_title}
    **搜索摘要:** {section_search_results_str[:3000]}...
    **原始材料:** {raw_materials}

    请撰写本章内容（500–800字），结构清晰、专业严谨。
    """
    try:
        draft_result = await Runner.run(DraftingAgent, draft_prompt)
        first_draft = draft_result.final_output
    except Exception as e:
        return f"## {section_title}\n\n[起草失败: {e}]"

    # 3. ✅ ReAct 循环：Review → Revise → Accept
    final_draft = await review_and_revise(
        section_title=section_title,
        draft=first_draft,
        raw_materials=raw_materials,
        search_summary=section_search_results_str,
        max_retries=2
    )

    print(f"✅ [{section_title}] 终稿确认")
    return f"## {section_title}\n\n{final_draft}"

# --- 4. 深度研究核心流程 ---

async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 检索 -> 抓取 -> 起草 -> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)
    print(initial_search_results_str)

    # 2. 生成研究大纲 (使用 JSON 模式确保结构化输出)
    print("\nStep 2: 基于初步结果生成研究大纲...")

    init_prompt = f"""研究主题: {query}
初步搜索结果摘要: {initial_search_results_str}
"""

    outline_prompt = init_prompt + """请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords' (用于精确检索的关键词)。

示例输出 JSON 格式如下，只要json，不要有其他输出
{
    "title": "关于 XX 的深度研究报告",
    "sections": [
        {"section_title": "引言与背景", "search_keywords": "历史, 现状"},
        {"section_title": "核心要素与机制", "search_keywords": "关键概念, 工作原理"},
        {"section_title": "应用与影响", "search_keywords": "行业应用, 社会影响"}
    ]
}
"""
    try:
        # 调用 Orchestrator Agent 生成 JSON 格式的大纲
        outline_response = await Runner.run(
            DeepResearchAgent,
            outline_prompt,
        )
        print(outline_response)
        outline_json = json.loads(outline_response.final_output.strip("```json").strip("```"))

    except Exception as e:
        print(f"Error generating outline: {e}. Falling back to a simple structure.")
        # 失败时提供默认大纲
        outline_json = {
            "title": f"关于 {query} 的深度研究报告",
            "sections": [
                {"section_title": "引言与背景", "search_keywords": f"{query}, 历史, 现状"},
                {"section_title": "核心要素与机制", "search_keywords": f"{query}, 工作原理, 关键技术"},
                {"section_title": "应用与影响", "search_keywords": f"{query}, 行业应用, 社会影响"},
                {"section_title": "结论与展望", "search_keywords": f"{query}, 发展趋势, 挑战"}
            ]
        }

    research_title = outline_json.get("title", f"关于 {query} 的深度研究报告")
    sections = outline_json.get("sections", [])
    if len(sections) > max_sections:
        sections = sections[:max_sections]

    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")

    # 3. 逐章进行检索、抓取和起草
    drafted_sections = []

    # 主流程：并发处理所有章节
    drafted_sections = await asyncio.gather(
        *(process_section(section) for section in sections),
        return_exceptions=True  # 防止一个章节失败导致整体崩溃
    )

    # 4. 报告整合与最终输出 (调用 Orchestrator Agent)
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    final_prompt = f"""
    请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

    **报告标题:** {research_title}

    **已起草的章节内容:**
    {full_report_draft}

    **任务要求:**
    1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
    2. 保持各章节之间的连贯性。
    3. 在报告末尾添加一个**【结论与展望】**部分（如果大纲中没有）。
    4. 添加一个**【引用来源】**列表，列出所有章节中提到的 URL。
    5. 整体报告必须格式优美，使用 Markdown 格式。
    """

    try:
        final_report = await Runner.run(
            DeepResearchAgent,
            final_prompt,
        )
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"

async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic)
    print(final_report)


# 使用 Runner 启动异步主函数
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except NameError:
        # Fallback to standard asyncio run if Runner is not defined or preferred
        asyncio.run(main())