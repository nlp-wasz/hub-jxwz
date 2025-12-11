import asyncio
import os

os.environ["OPENAI_API_KEY"] = "sk"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple

from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"
API_KEY = os.getenv("OPENAI_API_KEY", "sk")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 AsyncOpenAI 客户端
llm_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

model_settings = ModelSettings(
    model=MODEL_NAME,
    client=llm_client,
    temperature=0.3
)

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
    except requests.exceptions.RequestException as e:
        print(f"Error during Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"
    except Exception as e:
        print(f"Unexpected error in Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"


# 异步工具包装
async def async_search_jina(query: str) -> str:
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    return await asyncio.to_thread(crawl_jina, url)


external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

orchestrator_system_prompt = """
你是一名深度研究专家、项目经理兼质量评估师。你的核心任务包括三部分：
1. **研究规划**: 根据用户主题和初步搜索结果，生成JSON格式的详细报告大纲（含章节标题+搜索关键词）。
2. **质量评估 (ReAct 核心)**: 对每个章节的草稿进行评估，判断是否符合要求，输出评估结果和优化建议。评估维度：
   - 相关性：是否严格围绕章节主题
   - 完整性：信息是否全面，是否有遗漏的关键要点
   - 准确性：引用是否规范，数据是否准确
   - 结构：格式是否清晰，逻辑是否连贯
3. **报告整合**: 所有章节优化完成后，整合为完整报告（含摘要、结论、引用）。

# ReAct 评估输出格式（必须严格遵守）
{
    "evaluation": "pass" 或 "needs_improvement",  # pass=无需修改，needs_improvement=需要优化
    "reason": "评估理由（简要说明为什么通过或需要优化）",
    "suggestions": [
        "具体优化建议1（如：补充XX案例、修正XX引用、调整结构）",
        "具体优化建议2"
    ],
    "additional_search_keywords": "如需补充搜索，填写新增关键词；无需则填空字符串"
}
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

drafting_system_prompt = """
你是一名专业内容撰稿人，支持根据反馈迭代优化。你的任务是：
1. 首次起草：根据章节主题、搜索结果、抓取内容，撰写Markdown格式章节（严格聚焦主题、规范引用）。
2. 迭代优化：如果收到优化建议，针对建议修改章节内容，可结合新增搜索结果补充信息。

必须遵守的规则：
- 只使用提供的搜索结果和抓取内容，不编造信息
- 引用必须标注来源URL：[来源: URL]
- 格式为标准Markdown（二级标题以下层级，避免使用一级标题）
- 优化时保留原有有效内容，只修改需要优化的部分
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


async def generate_and_refine_section(
        section: Dict[str, str],
        max_iterations: int = 2  # 最大迭代次数（避免无限循环）
) -> Tuple[str, bool]:
    """
    单章节的生成+ReAct反馈迭代流程
    :param section: 章节信息（含section_title和search_keywords）
    :param max_iterations: 最大迭代次数
    :return: (最终章节内容, 是否通过评估)
    """
    section_title = section.get("section_title", "未命名章节")
    search_keywords = section.get("search_keywords", "")
    print(f"\n=== 开始处理章节：{section_title} ===")

    # 初始化：首次检索
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    search_results_str = await async_search_jina(section_query)
    print(f"[章节{section_title}] 首次搜索完成")

    # 筛选并抓取网页
    urls_to_crawl = []
    try:
        search_results = json.loads(search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        print(f"[章节{section_title}] 搜索结果解析失败，跳过抓取")

    crawled_content = []
    for url in urls_to_crawl:
        content = await async_crawl_jina(url)
        crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")
    raw_materials = "\n\n".join(crawled_content)
    print(f"[章节{section_title}] 抓取完成（{len(urls_to_crawl)}个网页）")

    # 迭代生成与优化（ReAct核心循环）
    current_draft = ""
    passed = False
    additional_search_results = ""  # 存储新增搜索结果
    evaluation_json = {"evaluation": "pass", "suggestions": [], "additional_search_keywords": ""}  # 初始化默认值

    for iteration in range(1, max_iterations + 1):
        print(f"\n[章节{section_title}] 第{iteration}次迭代生成")

        # 构造起草提示词（包含历史反馈和新增信息）
        draft_prompt = f"""
        **章节主题:** {section_title}
        **基础搜索关键词:** {search_keywords}
        **搜索结果摘要:**
        {search_results_str[:3000]}...
        {f"**补充搜索结果:**\n{additional_search_results[:3000]}..." if additional_search_results else ""}
        **原始网页内容:**
        {raw_materials}
        """

        # 如果是迭代优化，添加反馈信息
        if iteration > 1:
            draft_prompt += f"""
            **上一轮评估结果:** {evaluation_json['evaluation']}
            **评估理由:** {evaluation_json['reason']}
            **优化建议:** {chr(10).join(evaluation_json['suggestions'])}

            请根据以上反馈优化章节内容，保留原有有效信息，针对性修改问题部分。
            """

        draft_prompt += "\n请撰写/优化该章节的详细内容，严格遵守Markdown格式和引用规范。"

        # 调用起草代理生成/优化章节
        try:
            draft_response = await Runner.run(DraftingAgent, draft_prompt)
            current_draft = draft_response.final_output
            print(f"[章节{section_title}] 第{iteration}次生成完成")
        except Exception as e:
            print(f"[章节{section_title}] 第{iteration}次生成失败: {e}")
            current_draft = f"## {section_title}\n\n章节生成失败: {str(e)}"
            break

        # 调用协调代理进行ReAct评估
        evaluation_prompt = f"""
        请评估以下章节草稿是否符合要求：
        **章节主题:** {section_title}
        **章节草稿:**
        {current_draft[:5000]}...  # 截断过长内容，避免超出上下文
        """
        try:
            eval_response = await Runner.run(DeepResearchAgent, evaluation_prompt)
            # 清理输出（处理可能的代码块标记和空白字符）
            eval_output = eval_response.final_output.strip().strip("```json").strip("```").strip()
            evaluation_json = json.loads(eval_output)
            print(f"[章节{section_title}] 第{iteration}次评估结果: {evaluation_json['evaluation']}")
            print(f"[章节{section_title}] 优化建议: {evaluation_json['suggestions']}")
        except json.JSONDecodeError as e:
            print(f"[章节{section_title}] 评估结果JSON解析失败: {e}，默认通过")
            print(f"原始评估输出: {eval_response.final_output if 'eval_response' in locals() else '无'}")
            evaluation_json = {"evaluation": "pass", "suggestions": [], "additional_search_keywords": ""}
        except Exception as e:
            print(f"[章节{section_title}] 评估失败，默认通过: {e}")
            evaluation_json = {"evaluation": "pass", "suggestions": [], "additional_search_keywords": ""}

        # 根据评估结果判断是否继续迭代
        if evaluation_json["evaluation"] == "pass":
            passed = True
            print(f"[章节{section_title}] 评估通过，无需继续迭代")
            break
        else:
            # 需要优化：检查是否需要补充搜索（修复缩进和变量名）
            additional_keywords = evaluation_json.get("additional_search_keywords", "").strip()
            if additional_keywords and iteration < max_iterations:
                print(f"[章节{section_title}] 进行补充搜索，关键词: {additional_keywords}")
                # 修复：中文变量名改为英文，且纳入if缩进
                additional_search_query = f"{section_title} {additional_keywords}"
                additional_search_results = await async_search_jina(additional_search_query)
                # 补充抓取新增搜索结果的网页
                try:
                    new_search_results = json.loads(additional_search_results)
                    new_urls = [res['url'] for res in new_search_results if res.get('url')][:1]  # 新增1个网页
                    for url in new_urls:
                        new_content = await async_crawl_jina(url)
                        raw_materials += f"\n\n--- 补充 URL: {url} ---\n{new_content[:3000]}...\n"
                except:
                    print(f"[章节{section_title}] 补充搜索结果解析失败")

            # 修复：elif改为if，判断是否达最大迭代次数
            if iteration == max_iterations:
                print(f"[章节{section_title}] 已达最大迭代次数，停止优化")
                break

    # 格式化最终章节（添加标题，避免重复标题）
    if not current_draft.startswith(f"## {section_title}"):
        final_section = f"## {section_title}\n\n{current_draft}"
    else:
        final_section = current_draft
    print(f"\n=== 章节处理完成：{section_title} ===\n")
    return final_section, passed


# --- 深度研究核心流程（并行+ReAct） ---
async def deep_research(query: str, max_sections: int = 5, max_iterations: int = 2) -> str:
    """
    并行生成章节 + ReAct反馈迭代的深度研究流程
    """
    print(f"\n--- 开始深度研究：{query} ---\n")

    # Step 1: 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # Step 2: 生成研究大纲
    print("\nStep 2: 生成研究大纲...")
    outline_prompt = f"""研究主题: {query}
初步搜索结果摘要: {initial_search_results_str}
请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords'。
示例输出 JSON 格式如下，只要json，不要有其他输出：
{{
    "title": "关于 XX 的深度研究报告",
    "sections": [
        {{"section_title": "引言与背景", "search_keywords": "历史, 现状"}},
        {{"section_title": "核心要素与机制", "search_keywords": "关键概念, 工作原理"}}
    ]
}}
"""
    try:
        outline_response = await Runner.run(DeepResearchAgent, outline_prompt)
        outline_output = outline_response.final_output.strip().strip("```json").strip("```").strip()
        outline_json = json.loads(outline_output)
    except Exception as e:
        print(f"大纲生成失败，使用默认大纲: {e}")
        outline_json = {
            "title": f"关于 {query} 的深度研究报告",
            "sections": [
                {"section_title": "引言与背景", "search_keywords": f"{query}, 历史, 现状"},
                {"section_title": "核心要素与机制", "search_keywords": f"{query}, 工作原理, 关键技术"},
                {"section_title": "应用与影响", "search_keywords": f"{query}, 行业应用, 社会影响"},
                {"section_title": "挑战与解决方案", "search_keywords": f"{query}, 面临挑战, 应对策略"},
                {"section_title": "发展趋势", "search_keywords": f"{query}, 未来趋势, 技术演进"}
            ]
        }

    research_title = outline_json.get("title", f"关于 {query} 的深度研究报告")
    sections = outline_json.get("sections", [])[:max_sections]  # 限制最大章节数
    print(f"报告标题: {research_title}")
    print(f"规划章节数: {len(sections)} (并行处理)")

    # Step 3: 并行处理所有章节（核心改动：同时生成多个章节）
    print("\nStep 3: 并行生成并优化所有章节...")
    # 创建所有章节的任务（并行执行）
    section_tasks = [generate_and_refine_section(section, max_iterations) for section in sections]
    # 等待所有任务完成（返回结果列表）
    section_results = await asyncio.gather(*section_tasks, return_exceptions=True)  # 增加return_exceptions，避免单个任务失败导致整体崩溃

    # 整理章节结果（过滤失败章节和异常）
    drafted_sections = []
    for result in section_results:
        if isinstance(result, Tuple) and len(result) == 2 and result[0]:
            drafted_sections.append(result[0])
        else:
            print(f"警告：某个章节处理失败，跳过该章节。失败信息: {result}")
    print(f"\n所有章节处理完成，有效章节数: {len(drafted_sections)}")

    # Step 4: 整合最终报告
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)
    final_prompt = f"""
请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

**报告标题:** {research_title}
**已完成章节内容:**
{full_report_draft}

**整合要求:**
1. 开头添加【摘要】（300-500字，总结核心发现和结论）
2. 优化章节间的连贯性（添加过渡语句）
3. 末尾添加【结论与展望】（总结全文+未来发展预判）
4. 整理【引用来源】列表（提取所有章节中的URL，去重后按顺序排列）
5. 格式统一为Markdown，标题层级清晰（一级标题为报告标题，二级为章节标题）
"""
    try:
        final_report = await Runner.run(DeepResearchAgent, final_prompt)
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {str(e)}\n\n已完成章节草稿:\n{full_report_draft}"


# --- 入口函数 ---
async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    # 执行深度研究（并行章节+2轮ReAct迭代）
    final_report = await deep_research(research_topic, max_sections=5, max_iterations=2)
    # 打印最终报告（或写入文件）
    print("\n" + "=" * 50 + " 最终研究报告 " + "=" * 50 + "\n")
    print(final_report)

    # 可选：将报告写入文件
    with open("agentic_ai_research_report.md", "w", encoding="utf-8") as f:
        f.write(final_report)
    print("\n报告已保存到: agentic_ai_research_report.md")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        # 兼容旧版Python的降级处理
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        except Exception as e2:
            print(f"兼容模式执行也失败: {str(e2)}")