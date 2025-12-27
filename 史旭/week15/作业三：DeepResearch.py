# DeepResearch 深度检索（深入研究）
# 主要流程：
# 1.使用Jina等搜索引擎获取 研究标题 相关内容，之后通过Agent生成相关章节信息
# 2.对每一个章节使用Jina等搜索引擎获取 章节 相关内容，之后通过Agent生成章节报告，同时判断章节报告是否合理（不合理 给出修改建议，重新生成）
# 3.整合所有章节报告，通过Agent生成 研究标题 完整的高质量报告
# 4.输出结果信息
# 检索规划 -》 内容抓取 -》 章节报告（判断是否合格） -》 章节整合 -》 最终结论
import asyncio, requests, json, urllib.parse
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, trace
from agents import set_default_openai_api, set_tracing_disabled

# 0.Agent环境配置
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
async_openai = AsyncOpenAI(
    api_key="sk-04ab3d7290e243dda1badc5a1d5ac858",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
openai_model = OpenAIChatCompletionsModel(
    model="qwen-max",
    openai_client=async_openai
)


# 1.Jina 检索API
async def jina_search(search_title: str):
    # 1.将 search_title 转换为 url 类型
    url_search = urllib.parse.quote(search_title)

    # 构建 jina API 调用参数
    jina_url = f"https://s.jina.ai/?q={url_search}"
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer jina_c442fd5ce1824914903e68e8040373323iAVpKD5VhBL14jVwBeAy3e3yrFX",
        "X-Respond-With": "no-content"
    }

    try:
        response = requests.get(jina_url, headers=headers)
        response.raise_for_status()

        # 转换格式
        res_json = json.loads(response.text)

        # 获取 转换后的数据信息
        res_content = []
        for item in res_json["data"]:
            res_content.append({
                "title": item["title"],
                "url": item["url"],
                "description": item["description"]
            })

        return res_content
    except requests.exceptions.RequestException as e:
        print(f"Jina RequestError 检索失败，检索标题：{search_title}，异常信息：{e}")
        return []
    except Exception as e:
        print(f"Jina ExceptionError 检索失败，检索标题：{search_title}，异常信息：{e}")
        return []


# 2.Jina 读取API（获取 URL 网页内容）
async def jina_read(read_info: dict):
    # 1.获取 参数信息
    title = read_info["title"],
    url = read_info["url"],
    description = read_info["description"]
    print(f"Jina Read -> url：{url}")

    # 构建 jina API 调用参数
    url = f"https://r.jina.ai/{url[0]}"
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer jina_c442fd5ce1824914903e68e8040373323iAVpKD5VhBL14jVwBeAy3e3yrFX",
        "X-Respond-With": "no-content",
        "X-Content-Type": "markdown"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 转换格式
        res_content = response.json().get("data", []).get("content", "无任何信息")

        return res_content
    except requests.exceptions.RequestException as e:
        print(f"Jina RequestError 抓取失败，抓取章节：{title}，异常信息：{e}")
        return "无任何信息"
    except Exception as e:
        print(f"Jina ExceptionError 抓取失败，抓取章节：{title}，异常信息：{e}")
        return "无任何信息"


# 3.创建 流程Agent
# ①章节规划 Agent
chapterPlanningIns = """
你是一名深度研究专家和项目经理。你的任务是协调整个研究项目，包括：
1. **研究规划 (生成大纲):** 根据用户提供的研究主题和初步搜索结果，生成一个详尽、逻辑严密、结构清晰的报告大纲。大纲必须以严格的 JSON 格式输出，用于指导后续的章节内容检索和起草工作。
2. **报告整合 (组装):** 在所有章节内容起草完成后，将它们整合在一起，形成一篇流畅、连贯、格式优美的最终研究报告。报告必须包括摘要、完整的章节内容、结论和引用来源列表。
"""
chapterPlanningAgent = Agent(
    name="Chapter Planning Agent",
    instructions=chapterPlanningIns,
    model=openai_model
)

# ②章节报告生成 Agent
chapterReportIns = """
你是一名专业的内容撰稿人。你的任务是将提供的原始网页抓取内容和搜索结果，根据指定的章节主题，撰写成一篇结构合理、重点突出、信息准确的报告章节。
你必须严格遵守以下规则：
1. **聚焦主题:** 严格围绕给定的 '章节主题' 进行撰写。
2. **信息来源:** 只能使用提供的 '原始网页内容' 和 '搜索结果摘要' 中的信息。
3. **格式:** 使用 Markdown 格式。
4. **引用:** 对于文中引用的关键事实和数据，必须在段落末尾用脚注或括号标记引用的来源 URL，例如 [来源: URL]。
"""
chapterReportAgent = Agent(
    name="Chapter Report Agent",
    instructions=chapterPlanningIns,
    model=openai_model
)

# ③章节内容修正 Agent
chapterReactIns = """
你是一名严谨的学术审稿人和内容质量审核专家。你的任务是对一篇由AI撰写的报告章节进行**合理性、完整性、准确性与合规性**的全面评估。

请严格按以下规则执行：
1. **评估维度**：
   - **主题聚焦性**：是否紧扣给定的“章节主题”？
   - **信息准确性**：是否仅使用提供的原始网页内容？有无虚构、夸大或错误信息？
   - **引用规范性**：所有关键事实是否标注了来源 URL（如 [来源: https://xxx]）？
   - **结构逻辑性**：段落是否连贯？有无自相矛盾或逻辑断裂？
   - **内容充分性**：是否覆盖了搜索结果中的核心要点？

2. **输出格式（必须严格遵守）**：
   - 如果章节**合理且高质量**，请直接输出：
     ```json
     {"is_valid": true, "feedback": "章节内容合理，无需修改。"}
     ```
   - 如果章节**存在任何问题**，请输出：
     ```json
     {
       "is_valid": false,
       "feedback": "具体问题描述",
       "suggestions": ["建议1", "建议2", "..."]
     }
     ```

3. **禁止行为**：
   - 不得添加未在原始材料中出现的信息。
   - 不得输出除上述 JSON 外的任何文字（包括解释、问候、Markdown）。
"""
chapterReactAgent = Agent(
    name="Chapter React Agent",
    instructions=chapterReactIns,
    model=openai_model
)

# ④所有章节总结 Agent
chapterSummaryIns = """
你是一名资深研究报告主编，擅长将分散的章节内容整合为一篇逻辑严密、语言流畅、学术规范的完整报告。

你的任务是：
1. **阅读并理解所有章节内容**（每个章节以 `## 章节主题 XXX` 开头）。
2. **撰写一篇完整的最终研究报告**，必须包含以下部分：
   - **标题**：使用原始研究主题作为主标题
   - **摘要**：200–300 字，概括研究背景、核心发现、主要结论与意义
   - **正文**：保留所有章节的标题和内容，确保段落衔接自然，避免重复
   - **结论**：总结全文核心观点，指出未来方向或实践价值
   - **参考文献**：汇总所有在正文中引用过的 URL（去重），按出现顺序或字母序列出

3. **格式要求**：
   - 使用标准 Markdown
   - 所有引用必须保留原始 `[来源: URL]` 格式
   - 不得添加任何未在输入章节中出现的新信息
   - 语言正式、客观、简洁，避免口语化

4. **禁止行为**：
   - 虚构数据、案例或引用
   - 删除或忽略任一输入章节
   - 输出除最终报告外的任何解释性文字
"""
chapterSummaryAgent = Agent(
    name="Chapter Summary Agent",
    instructions=chapterSummaryIns,
    model=openai_model
)


# 章节报告生成（异步方法，所有章节一同生成）
async def chapterRepostGenerator(idx, section):
    section_title = section["section_title"]
    search_keywords = section["search_keywords"]

    print(f"章节 {idx}，{section_title}，关键词：{search_keywords}")

    # 根据 section_title 检索相关内容信息
    section_title_jina = await jina_search(section_title)
    section_title_jina_res = json.dumps(section_title_jina, indent=2, ensure_ascii=False)
    print(f"章节 {idx}，{section_title} 检索结果：{section_title_jina_res}")

    # 根据 section_title_jina 中的 url，获取网页内容
    section_title_page_content = []
    for section_title_info in section_title_jina[:2]:
        section_title_read_res = await jina_read(section_title_info)

        section_title_page_content.append(
            f"URL: {section_title_info['url']}，内容：{section_title_read_res[:2000]}"
        )

    # 整合 抓取到的 页面内容
    chapterUrlContent = "\n\n".join(section_title_page_content)
    print(f"章节 {idx}，{section_title} 抓取结果：{chapterUrlContent}")

    # 根据 抓取到的章节内容，使用Agent生成 章节报告
    chapterReportPrompt = f"""
    **章节主题:**
    {section_title}

    **搜索结果摘要:**
    {section_title_jina_res}

    **原始网页内容 (请基于此内容撰写):**
    {chapterUrlContent}
    """
    try:
        # 章节报告
        chapterReportRes = await Runner.run(chapterReportAgent, chapterReportPrompt)
        print(f"\n\n章节报告 {idx}：{chapterReportRes.final_output}")

        # 判断章节报告是否合理
        chapterReactPrompt = f"""
        你将收到一个AI生成的报告章节，请根据审核规则进行评估。

        **章节主题:**
        {section_title}

        **原始网页内容摘要（用于事实核对）:**
        {section_title_jina_res}

        **完整的原始网页内容（节选）:**
        {chapterUrlContent}

        **待审核的章节报告:**
        {chapterReportRes.final_output}
        """
        # 章节报告 是否合理

        chapterReactRes = await Runner.run(chapterReactAgent, chapterReactPrompt)
        print(f"\n\n章节报告校验检查 {idx}：{chapterReactRes.final_output}")

        chapterReactResPy = json.loads(chapterReactRes.final_output.strip("```json").strip("```"))
        if chapterReactResPy != None and chapterReactResPy.get("is_valid", False):
            return f"## 章节主题 {section_title}\n\n{chapterReportRes.final_output}\n\n"
        else:
            return f"## 章节主题 {idx} {section_title}\n\n{chapterReportRes.final_output}\n\n{chapterReactResPy.get('suggestions', '无任何问题')}"

    except Exception as e:
        print(f"章节报告Prompt：{chapterReportPrompt}")
        print(f"\n\n章节 {idx} 报告生成or校验检查 Error：{e}")


# 4.DeepResearch 流程
async def deepresearch(query: str):
    # 1.检索规划
    print(f"DeepResearch 深度研究第一步：检索规划")
    print(f"1.检索标题：{query}")

    # 调用 jina 检索API
    query_jina_search_res = await jina_search(query)
    query_jina_search_res = json.dumps(query_jina_search_res, indent=2, ensure_ascii=False)
    print(f"检索结果：{query_jina_search_res}")

    # 使用Agent，对检索结果进行规划（生成 章节信息）
    chapterPlanningPrompt = f"""研究主题：{query}\n初步搜索摘要：{query_jina_search_res}\n"""
    chapterPlanningPrompt += """请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
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
    # 调用 章节规划 Agent
    chapterPlanningRes = await Runner.run(chapterPlanningAgent, chapterPlanningPrompt)

    # 章节规划
    print(f"\n\n章节规划信息：{chapterPlanningRes.final_output}")

    chapterPlanningRes = json.loads(chapterPlanningRes.final_output.strip("```json").strip("```"))

    # 2.章节内容生成
    print(f"2.正在生成章节内容：标题{query}")
    chapterReportFunction = []
    for idx, section in enumerate(chapterPlanningRes["sections"], start=1):
        # 获取 章节报告生成 对应的异步函数
        chapterReportFunction.append(chapterRepostGenerator(idx, section))

    # 异步调用（所有章节同时生成）
    chapterReportInfo = await asyncio.gather(*chapterReportFunction)

    # 3. 整合章节报告
    chapterReportAll = "\n\n".join(chapterReportInfo)
    print(f"3.章节信息已整合完毕：{chapterReportAll}")

    # 生成完整的 标题报告
    chapterSummaryPrompt = f"""
    请根据以下多个章节内容，整合生成一篇完整的深度研究报告。
    
    研究主题：{query}
    
    所有章节内容如下：
    {chapterReportAll}
    """
    chapterSummaryRes = await Runner.run(chapterSummaryAgent, chapterSummaryPrompt)

    print(f"\n\n4.**研究主题报告：\n{chapterSummaryRes.final_output}")

    with open(f"./{query}.md", "w", encoding="utf-8") as f:
        f.write(chapterSummaryRes.final_output)


if __name__ == '__main__':
    asyncio.run(deepresearch("Agentic AI在软件开发中的最新应用和挑战"))
