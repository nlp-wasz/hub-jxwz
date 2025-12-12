import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
# 获取环境变量
llm_api_key = os.getenv('DASHSCOPE_API_KEY')
JINA_API_KEY = os.getenv('JINA_API_KEY')
os.environ["OPENAI_API_KEY"] = llm_api_key

# os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
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
            "X-Content-Type": "text"  # 请求返回文本格式
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

# --- 新增：质量评估代理 ---
evaluation_system_prompt = """
你是专业的内容质量评估专家。你的任务是对起草的章节内容进行多维度评估，识别问题并提供改进建议。

评估标准：
1. **信息准确性**: 内容是否基于提供的原始材料，事实是否准确
2. **结构逻辑性**: 章节结构是否清晰合理，逻辑是否连贯
3. **内容完整性**: 是否覆盖了章节主题的关键方面
4. **语言质量**: 表达是否清晰专业，语法是否正确
5. **引用规范性**: 是否正确标注信息来源

请返回严格的JSON格式评估结果：
{
    "score": 0-10的评分,
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"], 
    "suggestions": ["改进建议1", "改进建议2"],
    "pass": true/false (是否达到质量阈值)
}
"""
EvaluationAgent = Agent(
    "Quality Evaluation Specialist",
    instructions=evaluation_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# --- 新增：实时监控代理 ---
monitoring_system_prompt = """
你是实时质量监控代理，在内容生成过程中持续监控质量指标：
1. **内容相关性检查**: 确保内容与章节主题高度相关
2. **事实一致性验证**: 检查内容是否与原始材料一致  
3. **结构完整性评估**: 评估章节结构是否完整合理
4. **语言流畅度监控**: 检查语言表达是否流畅专业

当发现严重问题时，及时提供修正建议。
"""
MonitoringAgent = Agent(
    "Real-time Quality Monitor",
    instructions=monitoring_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


# --- 新增：评估工具函数 ---
async def evaluate_section_quality(section_title: str, content: str, source_materials: str) -> Dict[str, Any]:
    """评估章节质量，返回评分和改进建议"""
    print(f"-> [质量评估] 正在评估章节: {section_title}")

    evaluation_prompt = f"""
    请评估以下章节的质量：

    章节标题: {section_title}
    章节内容: {content}
    参考来源材料: {source_materials[:1000]}...

    请基于以下标准进行严格评估：
    1. 信息准确性（基于参考材料）
    2. 结构逻辑性
    3. 内容完整性  
    4. 语言质量
    5. 引用规范性

    返回严格的JSON格式：
    {{
        "score": 0-10的评分,
        "strengths": ["优点1", "优点2"],
        "weaknesses": ["不足1", "不足2"],
        "suggestions": ["改进建议1", "改进建议2"],
        "pass": true/false (score >= 7.5为通过)
    }}
    """

    try:
        evaluation_result = await Runner.run(
            EvaluationAgent,
            evaluation_prompt,
        )
        # 改进的JSON解析逻辑
        output = evaluation_result.final_output.strip()
        # 移除markdown代码块标记
        output = output.strip("```json").strip("```").strip()
        # 尝试提取第一个JSON对象
        if output:
            # 查找第一个{和对应的}
            start_idx = output.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(output)):
                    if output[i] == '{':
                        brace_count += 1
                    elif output[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = output[start_idx:i + 1]
                            eval_data = json.loads(json_str)
                            return eval_data
        raise ValueError("无法解析JSON")
    except Exception as e:
        print(f"评估失败: {e}")
        return {"score": 5, "pass": False, "suggestions": ["评估过程出现错误"]}


async def monitor_drafting_progress(section_title: str, current_content: str, stage: str) -> Dict[str, Any]:
    """实时监控起草进度和质量"""
    print(f"-> [实时监控] {stage}阶段检查: {section_title}")

    monitor_prompt = f"""
    正在起草章节: {section_title}
    当前阶段: {stage}
    当前内容: {current_content}

    请检查是否存在以下严重问题：
    1. 内容严重偏离主题
    2. 事实与来源材料明显矛盾
    3. 结构混乱无法理解
    4. 语言表达存在严重问题

    返回JSON格式：
    {{
        "has_critical_issues": true/false,
        "issues": ["问题描述1", "问题描述2"],
        "suggestions": ["修正建议1", "修正建议2"]
    }}
    """

    try:
        monitor_result = await Runner.run(
            MonitoringAgent,
            monitor_prompt,
        )
        # 改进的JSON解析逻辑
        output = monitor_result.final_output.strip()
        # 移除markdown代码块标记
        output = output.strip("```json").strip("```").strip()
        # 尝试提取第一个JSON对象
        if output:
            # 查找第一个{和对应的}
            start_idx = output.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(output)):
                    if output[i] == '{':
                        brace_count += 1
                    elif output[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = output[start_idx:i + 1]
                            monitor_data = json.loads(json_str)
                            return monitor_data
        # 如果解析失败，返回默认值（无严重问题）
        return {"has_critical_issues": False, "issues": [], "suggestions": []}
    except Exception as e:
        print(f"监控检查失败: {e}")
        return {"has_critical_issues": False, "issues": [], "suggestions": []}


# --- 替换原有的章节处理函数 ---
async def process_section_with_quality_control(section_title: str, search_keywords: str, max_retries: int = 3) -> str:
    """带质量控制的章节处理流程"""
    print(f"-> 开始处理章节: {section_title}")

    # 1. 精确检索
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    section_search_results_str = await async_search_jina(section_query)

    # 2. 筛选并抓取链接
    try:
        search_results = json.loads(section_search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        print(f"Warning: Failed to parse search results for crawl in section {section_title}.")
        urls_to_crawl = []

    crawled_content = []
    for url in urls_to_crawl:
        content = await async_crawl_jina(url)
        crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")

    raw_materials = "\n\n".join(crawled_content)

    best_content = None
    best_score = 0
    improvement_history = []

    for attempt in range(max_retries):
        print(f"  第{attempt + 1}轮起草尝试...")

        # 3. 分阶段起草与实时监控
        draft_stages = [
            ("大纲生成", "为本章节生成详细的内容大纲"),
            ("内容填充", "基于大纲填充具体内容"),
            ("细节完善", "完善细节和引用"),
            ("最终润色", "进行语言润色和格式优化")
        ]

        current_draft = ""
        for stage_name, stage_task in draft_stages:
            # 实时监控当前阶段
            monitor_result = await monitor_drafting_progress(section_title, current_draft, stage_name)

            if monitor_result.get("has_critical_issues", False):
                print(f"  ⚠ 在{stage_name}阶段发现严重问题: {monitor_result.get('issues', [])}")
                # 基于监控建议进行调整
                adjustment_prompt = f"""
                当前章节: {section_title}
                当前内容: {current_draft}
                发现问题: {monitor_result.get('issues', [])}
                改进建议: {monitor_result.get('suggestions', [])}

                请根据上述反馈重新进行{stage_name}。
                """
                try:
                    adjusted_draft = await Runner.run(DraftingAgent, adjustment_prompt)
                    current_draft = adjusted_draft.final_output
                except Exception as e:
                    print(f"  调整失败: {e}")

            # 生成当前阶段内容
            stage_prompt = f"""
            **章节主题:** {section_title}
            **阶段任务:** {stage_task}

            **搜索结果摘要:**
            {section_search_results_str[:3000]}... 

            **原始网页内容:**
            {raw_materials}

            **当前已有内容:**
            {current_draft}

            请继续完成{stage_name}阶段的工作。
            """

            try:
                stage_result = await Runner.run(DraftingAgent, stage_prompt)
                current_draft = stage_result.final_output
                print(f"  ✓ 完成{stage_name}阶段")
            except Exception as e:
                print(f"  ❌ {stage_name}阶段失败: {e}")

        # 4. 完整性质量评估
        evaluation_result = await evaluate_section_quality(section_title, current_draft, raw_materials)

        current_score = evaluation_result.get("score", 0)
        improvement_history.append({
            "attempt": attempt + 1,
            "score": current_score,
            "suggestions": evaluation_result.get("suggestions", [])
        })

        print(f"  质量评分: {current_score}/10")

        # 更新最佳内容
        if current_score > best_score:
            best_content = current_draft
            best_score = current_score

        # 检查是否通过质量阈值
        if evaluation_result.get("pass", False):
            print(f"  ✓ 章节 '{section_title}' 通过质量评估")
            return f"## {section_title}\n\n{current_draft}"

        # 准备下一轮迭代
        if attempt < max_retries - 1:
            print(f"  🔄 准备重新起草...")
            # 基于评估建议准备下一轮
            retry_prompt = f"""
            章节主题: {section_title}

            上一轮内容: {current_draft}

            评估反馈:
            - 优点: {evaluation_result.get('strengths', [])}
            - 不足: {evaluation_result.get('weaknesses', [])}  
            - 建议: {evaluation_result.get('suggestions', [])}

            请基于上述反馈重新起草本章节，重点改进指出的问题。
            """

            try:
                retry_result = await Runner.run(DraftingAgent, retry_prompt)
                current_draft = retry_result.final_output
            except Exception as e:
                print(f"  重新起草失败: {e}")

    # 输出迭代历史
    if improvement_history:
        print(f"  📊 章节 '{section_title}' 质量改进历程:")
        for step in improvement_history:
            print(f"    第{step['attempt']}轮: 评分 {step['score']}")

    # 返回最佳内容（即使未达到阈值）
    if best_content:
        print(f"  ⚠ 使用最佳版本 (评分: {best_score}/10)")
        return f"## {section_title}\n\n{best_content}"
    else:
        error_msg = f"章节起草失败，经过{max_retries}次尝试仍无法达到质量要求"
        print(f"  ❌ {error_msg}")
        return f"## {section_title}\n\n{error_msg}"


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

    # 大模型基于主题和初步检索结果，进行章节的规划
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

    # 3. 并行处理各章节（带质量控制）
    print("\nStep 3: 并行处理各章节（带质量控制和迭代改进）...")

    tasks = []
    for i, section in enumerate(sections):
        section_title = section.get("section_title")
        search_keywords = section.get("search_keywords")

        print(f"\n--- 准备处理章节 {i + 1}: {section_title} ---")

        # 使用带质量控制的章节处理函数
        task = process_section_with_quality_control(section_title, search_keywords)
        tasks.append(task)

    # 并行执行所有章节处理任务
    print(f"\n--- 并行处理 {len(tasks)} 个章节 ---")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    drafted_sections = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"## {sections[i].get('section_title')}\n\n章节处理异常: {result}"
            drafted_sections.append(error_msg)
            print(f"❌ 章节 {i + 1} 处理失败: {result}")
        else:
            drafted_sections.append(result)
            print(f"✓ 章节 {i + 1} 处理完成")

    # 4. 报告整合与最终输出
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    # 优化策略：如果内容过长，直接组装而不调用LLM
    if len(full_report_draft) > 15000:  # 字符数阈值
        print("  内容较长，采用直接组装方式...")

        # 生成简单的摘要
        summary = f"""## 摘要

本报告深入探讨了{research_title}，涵盖了{len(sections)}个关键方面。报告基于最新的行业资料和实践案例，为读者提供全面的分析和见解。
"""

        # 直接组装报告
        final_report = f"""# {research_title}

{summary}

---

{full_report_draft}

---

## 结论

通过本报告的研究，我们全面分析了{research_title}的各个方面。随着技术的不断发展，相关领域将继续演进，值得持续关注。
"""
        return final_report

    # 内容不长时，使用LLM整合（带超时重试）
    print("  使用AI整合报告...")

    # 简化prompt，减少token消耗
    final_prompt = f"""
    请为以下研究报告添加摘要和结论。

    **报告标题:** {research_title}

    **章节内容:**
    {full_report_draft[:10000]}...  # 限制长度

    **任务:**
    1. 在开头添加简洁的摘要（200字以内）
    2. 在末尾添加结论（300字以内）
    3. 保持Markdown格式

    直接输出完整报告，包含：摘要 + 原章节内容 + 结论
    """

    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"  尝试整合 ({attempt + 1}/{max_retries})...")
            final_report = await Runner.run(
                DeepResearchAgent,
                final_prompt,
            )
            print("  ✓ 报告整合成功")
            return final_report.final_output
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "RequestTimeOut" in error_msg:
                print(f"  ⚠ 请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print("  等待后重试...")
                    await asyncio.sleep(2)  # 等待2秒后重试
                    continue
            print(f"  ❌ 整合失败: {e}")

    # 所有尝试都失败，返回基础版本
    print("  使用基础组装方式...")
    summary = f"""## 摘要

本报告深入探讨了{research_title}，涵盖了{len(sections)}个关键方面。
"""

    final_report = f"""# {research_title}

{summary}

---

{full_report_draft}

---

## 结论

本报告全面分析了{research_title}的各个方面，为读者提供了深入的见解。
"""
    return final_report


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