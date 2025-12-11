# --- 新增：章节 ReAct 评估与迭代优化 ---
import asyncio
import os
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
API_KEY = os.getenv("OPENAI_API_KEY", "sk-c4395731abd4446b8642c7734c8dbf56")
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
async def evaluate_and_refine_section(
    section_title: str,
    initial_draft: str,
    search_results_str: str,
    raw_materials: str,
    drafting_agent: Agent,
    orchestrator_agent: Agent,
    max_refinements: int = 2
) -> str:
    """
    使用 ReAct 机制对章节初稿进行评估和迭代优化。
    """
    current_draft = initial_draft
    for attempt in range(max_refinements + 1):
        if attempt == 0:
            # 第一次是原始草稿，直接进入评估
            pass
        else:
            print(f"-> [ReAct] 第 {attempt} 次重写: {section_title}")
            # 根据反馈重写
            rewrite_prompt = f"""
你是一名专业的内容撰稿人。请根据以下反馈，重写该章节。

**章节主题:** {section_title}
**原始材料 (供参考):**
{raw_materials}

**当前草稿:**
{current_draft}

**专家反馈:**
{feedback}

**任务:** 请严格根据反馈修改，输出改进后的完整章节内容（Markdown格式）。
"""
            try:
                refined_response = await Runner.run(drafting_agent, rewrite_prompt)
                current_draft = refined_response.final_output
            except Exception as e:
                print(f"-> 重写失败: {e}")
                break

        # 评估当前草稿（即使是初稿也评估）
        eval_prompt = f"""
你是一名资深研究编辑，请对以下章节草稿进行严格评估。
**章节主题:** {section_title}
**应覆盖要点（来自搜索摘要）:** {search_results_str[:2000]}
**当前草稿:**{current_draft}
请按以下格式输出评估结果（必须是 JSON）：
{{
  "is_satisfactory": true/false,
  "issues": ["问题1", "问题2", ...],
  "suggestions": "具体的、可操作的改进建议"
}}
"""
        try:
            eval_response = await Runner.run(orchestrator_agent, eval_prompt)
            eval_text = eval_response.final_output.strip()
            # 清理可能的 Markdown 代码块
            if eval_text.startswith("```json"):
                eval_text = eval_text[7:-3]
            evaluation = json.loads(eval_text)
        except Exception as e:
            print(f"-> 评估解析失败，视为满意: {e}")
            return current_draft  # 安全回退

        if evaluation.get("is_satisfactory", True):
            print(f"-> [ReAct] 章节通过评估: {section_title}")
            return current_draft
        else:
            feedback = evaluation.get("suggestions", "无具体建议")
            print(f"-> [ReAct] 需要改进 ({section_title}): {feedback[:100]}...")

    print(f"-> [ReAct] 达到最大重试次数，返回最后版本: {section_title}")
    return current_draft

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
def crawl_jina(url: str) -> str:
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
async def async_search_jina(query: str) -> str:
    """异步调用 Jina 搜索"""
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    """异步调用 Jina 抓取"""
    return await asyncio.to_thread(crawl_jina, url)

external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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

# --- 修改后的 deep_research 函数（支持并行 + ReAct）---
async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 并行检索/抓取/起草（带 ReAct 反馈）-> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # 2. 生成研究大纲
    print("\nStep 2: 基于初步结果生成研究大纲...")
    init_prompt = f"""研究主题: {query}\n初步搜索结果摘要: {initial_search_results_str}"""
    outline_prompt = init_prompt + """
请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords'。
示例输出 JSON 格式如下，只要json，不要有其他输出
{ "title": "关于 XX 的深度研究报告", "sections": [ {"section_title": "引言与背景", "search_keywords": "历史, 现状"} ]}
"""

    try:
        outline_response = await Runner.run(DeepResearchAgent, outline_prompt)
        outline_json = json.loads(outline_response.final_output.strip("```json").strip("```"))
    except Exception as e:
        print(f"Error generating outline: {e}. Falling back to a simple structure.")
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
    sections = outline_json.get("sections", [])[:max_sections]
    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")

    # 3. 并行处理所有章节（检索 + 抓取 + 起草 + ReAct 优化）
    async def process_section(section: Dict[str, Any]) -> str:
        section_title = section["section_title"]
        search_keywords = section["search_keywords"]
        print(f"\n--- 开始处理章节: {section_title} ---")

        # 3.1 精确检索
        section_query = f"{section_title} {search_keywords}"
        section_search_results_str = await async_search_jina(section_query)

        # 3.2 抓取前2个链接
        crawled_content = []
        try:
            search_results = json.loads(section_search_results_str)
            urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
            for url in urls_to_crawl:
                content = await async_crawl_jina(url)
                crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")
        except Exception as e:
            print(f"Warning: 抓取失败 {e}")

        raw_materials = "\n\n".join(crawled_content)

        # 3.3 初稿起草
        draft_prompt = f"""**章节主题:** {section_title}
                           **搜索结果摘要:** {section_search_results_str[:3000]}...
                           **原始网页内容:**{raw_materials}
                           请撰写本章节内容（Markdown格式），务必引用来源。"""
        try:
            initial_response = await Runner.run(DraftingAgent, draft_prompt)
            initial_draft = initial_response.final_output
        except Exception as e:
            return f"## {section_title}\n\n起草失败: {e}"

        # 3.4 ReAct 评估与优化
        refined_draft = await evaluate_and_refine_section(
            section_title=section_title,
            initial_draft=initial_draft,
            search_results_str=section_search_results_str,
            raw_materials=raw_materials,
            drafting_agent=DraftingAgent,
            orchestrator_agent=DeepResearchAgent
        )

        return f"## {section_title}\n\n{refined_draft}"

    # 并行执行所有章节处理
    print("\nStep 3: 并行生成并优化所有章节（含 ReAct 反馈）...")
    drafted_sections = await asyncio.gather(
        *(process_section(section) for section in sections),
        return_exceptions=True
    )

    # 处理异常
    final_sections = []
    for i, result in enumerate(drafted_sections):
        if isinstance(result, Exception):
            section_title = sections[i]["section_title"]
            final_sections.append(f"## {section_title}\n\n处理过程中发生错误: {str(result)}")
        else:
            final_sections.append(result)

    # 4. 报告整合
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(final_sections)
    final_prompt = f"""
请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

**报告标题:** {research_title}
**已起草的章节内容:**
{full_report_draft}

**任务要求:**
1. 在报告开头添加一个**【摘要】**，总结主要发现。
2. 保持章节连贯性。
3. 在末尾添加**【结论与展望】**（若缺失）。
4. 添加**【引用来源】**列表（提取所有 URL）。
5. 使用 Markdown 格式，排版优美。
"""
    try:
        final_report = await Runner.run(DeepResearchAgent, final_prompt)
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {e}\n\n已完成的章节:\n{full_report_draft}"
