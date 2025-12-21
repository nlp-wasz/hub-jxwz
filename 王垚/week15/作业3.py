import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

# 配置环境变量
os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 新增数据结构 ---

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class QualityAssessment:
    """质量评估结果"""
    quality_score: float
    completeness_score: float
    coherence_score: float
    citation_score: float
    language_score: float
    improvement_suggestions: List[str]
    needs_regeneration: bool
    feedback: str

@dataclass
class ChapterTask:
    """章节任务数据结构"""
    section_info: Dict
    status: TaskStatus = TaskStatus.PENDING
    content: str = ""
    quality_score: float = 0.0
    react_round: int = 0
    feedback_history: List[str] = None
    retry_count: int = 0
    creation_time: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.feedback_history is None:
            self.feedback_history = []

    @property
    def section_title(self) -> str:
        return self.section_info.get("section_title", "未知章节")

class APIRateLimiter:
    """API调用频率限制器"""
    def __init__(self, max_calls_per_second: int = 10):
        self.calls = []
        self.max_calls = max_calls_per_second
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # 清理超过1秒的调用记录
            self.calls = [call_time for call_time in self.calls if now - call_time < 1.0]

            if len(self.calls) >= self.max_calls:
                sleep_time = 1.0 - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.calls.append(now)

class ResearchError(Exception):
    """研究过程基础异常"""
    pass

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

# 3.3. ReAct质量评估代理 (ReAct Quality Assessor)
react_system_prompt = """
你是专业的内容质量评估专家。采用ReAct（Reasoning and Acting）方法进行质量评估：

**Reason（推理）**: 深入分析章节内容的质量、完整性、准确性和逻辑性
**Act（行动）**: 提供具体的改进建议和优化方向
**Iterate（迭代）**: 通过多轮反馈确保内容达到最高质量标准

**评估维度：**
- 内容深度与广度 (权重: 25%) - 是否全面覆盖主题的关键方面
- 逻辑连贯性与结构 (权重: 25%) - 内容组织是否清晰，逻辑是否流畅
- 引用质量与准确性 (权重: 25%) - 信息来源的可靠性和相关性
- 语言表达与专业性 (权重: 25%) - 表达是否清晰、准确、专业

**质量要求：** 综合评分必须达到0.85以上才能通过评估。

请对提供的章节内容进行严格评估，并给出具体的改进建议。如果质量不达标，明确指出需要改进的方面。
"""
ReactAgent = Agent(
    "ReAct Quality Assessor",
    instructions=react_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


# --- 4. 并行研究管理器 ---

class ParallelResearchManager:
    """并行研究管理器 - 负责协调并发的章节生成和质量评估"""

    def __init__(self, max_concurrent: int = 8, max_react_rounds: int = 4, strict_error_handling: bool = True):
        self.max_concurrent = max_concurrent  # 激进并发策略
        self.max_react_rounds = max_react_rounds  # 严格质量评估轮数
        self.strict_error_handling = strict_error_handling  # 保守错误处理
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.api_limiter = APIRateLimiter(max_calls_per_second=10)
        self.tasks = []
        self.completed_tasks = []
        self.failed_tasks = []

    async def create_chapter_tasks(self, sections: List[Dict]) -> List[ChapterTask]:
        """创建章节任务列表"""
        tasks = []
        for section in sections:
            task = ChapterTask(section_info=section)
            tasks.append(task)
        return tasks

    async def process_single_chapter(self, chapter_task: ChapterTask) -> ChapterTask:
        """处理单个章节的完整流程：检索→抓取→起草→ReAct评估"""
        async with self.semaphore:  # 控制并发数量
            chapter_task.status = TaskStatus.RUNNING
            logger.info(f"开始处理章节: {chapter_task.section_title}")

            try:
                # 1. 精确检索
                await self.api_limiter.acquire()
                section_query = f"{chapter_task.section_title} 搜索关键词: {chapter_task.section_info.get('search_keywords', '')}"
                section_search_results_str = await async_search_jina(section_query)

                # 2. 抓取内容
                urls_to_crawl = []
                try:
                    search_results = json.loads(section_search_results_str)
                    urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
                except:
                    logger.warning(f"解析搜索结果失败: {chapter_task.section_title}")

                crawled_content = []
                chapter_task.sources = urls_to_crawl.copy()

                for url in urls_to_crawl:
                    await self.api_limiter.acquire()
                    content = await async_crawl_jina(url)
                    crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")

                raw_materials = "\n\n".join(crawled_content)

                # 3. 内容起草
                draft_prompt = f"""
                **章节主题:** {chapter_task.section_title}

                **搜索结果摘要:**
                {section_search_results_str[:3000]}... (仅用于参考要点)

                **原始网页内容 (请基于此内容撰写):**
                {raw_materials}

                请根据上述信息，撰写 {chapter_task.section_title} 这一章节的详细内容。
                """

                await self.api_limiter.acquire()
                section_draft = await Runner.run(DraftingAgent, draft_prompt)
                chapter_task.content = section_draft.final_output

                logger.info(f"章节初稿完成: {chapter_task.section_title}")

                # 4. ReAct质量评估循环
                chapter_task.content = await self.react_quality_cycle(chapter_task)

                chapter_task.status = TaskStatus.COMPLETED
                chapter_task.completion_time = time.time()
                logger.info(f"章节处理完成: {chapter_task.section_title}, 最终质量评分: {chapter_task.quality_score}")

                return chapter_task

            except Exception as e:
                chapter_task.status = TaskStatus.FAILED
                error_msg = f"章节处理失败: {e}"
                logger.error(f"{error_msg} - {chapter_task.section_title}")

                if self.strict_error_handling:
                    # 保守错误处理：抛出异常停止所有任务
                    await self.handle_critical_error(e)
                else:
                    # 记录失败但继续处理其他章节
                    chapter_task.content = f"## {chapter_task.section_title}\n\n{error_msg}"

                return chapter_task

    async def react_quality_cycle(self, chapter_task: ChapterTask) -> str:
        """ReAct质量评估循环 - 严格策略"""
        for round_num in range(self.max_react_rounds):
            # Reason: 评估当前内容质量
            assessment = await self.assess_content_quality(chapter_task.content, chapter_task.section_info)

            chapter_task.quality_score = assessment.quality_score

            # 质量达标则通过
            if assessment.quality_score >= 0.85:
                logger.info(f"章节 {chapter_task.section_title} 质量评估通过，评分: {assessment.quality_score}")
                break

            # 最后一轮仍未达标，记录警告
            if round_num == self.max_react_rounds - 1:
                logger.warning(f"章节 {chapter_task.section_title} 达到最大评估轮次仍未达标，最终评分: {assessment.quality_score}")
                break

            # Act: 生成改进建议并重新起草
            improvement_prompt = self.generate_improvement_prompt(assessment, chapter_task)

            await self.api_limiter.acquire()
            section_draft = await Runner.run(DraftingAgent, improvement_prompt)
            chapter_task.content = section_draft.final_output
            chapter_task.react_round += 1

            # 记录反馈历史
            chapter_task.feedback_history.append(assessment.feedback)

            # 添加间隔避免API过载
            await asyncio.sleep(0.5)

        return chapter_task.content

    async def assess_content_quality(self, content: str, section_info: Dict) -> QualityAssessment:
        """评估章节内容质量"""
        assessment_prompt = f"""
        请对以下章节内容进行质量评估：

        **章节主题:** {section_info.get('section_title', '未知')}

        **章节内容:**
        {content[:4000]}... (内容截断用于评估)

        请按照以下格式进行评估（只需要JSON，不要其他内容）：
        {{
            "quality_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "coherence_score": 0.0-1.0,
            "citation_score": 0.0-1.0,
            "language_score": 0.0-1.0,
            "improvement_suggestions": ["建议1", "建议2"],
            "needs_regeneration": true/false,
            "feedback": "详细的评估反馈"
        }}
        """

        try:
            await self.api_limiter.acquire()
            response = await Runner.run(ReactAgent, assessment_prompt)
            assessment_data = json.loads(response.final_output.strip("```json").strip("```"))

            return QualityAssessment(
                quality_score=assessment_data.get("quality_score", 0.0),
                completeness_score=assessment_data.get("completeness_score", 0.0),
                coherence_score=assessment_data.get("coherence_score", 0.0),
                citation_score=assessment_data.get("citation_score", 0.0),
                language_score=assessment_data.get("language_score", 0.0),
                improvement_suggestions=assessment_data.get("improvement_suggestions", []),
                needs_regeneration=assessment_data.get("needs_regeneration", False),
                feedback=assessment_data.get("feedback", "无反馈")
            )
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            # 返回默认评估结果
            return QualityAssessment(
                quality_score=0.5,
                completeness_score=0.5,
                coherence_score=0.5,
                citation_score=0.5,
                language_score=0.5,
                improvement_suggestions=["重新生成内容"],
                needs_regeneration=True,
                feedback=f"评估过程出错: {e}"
            )

    def generate_improvement_prompt(self, assessment: QualityAssessment, chapter_task: ChapterTask) -> str:
        """生成改进提示词"""
        improvements_text = "\n".join([f"- {suggestion}" for suggestion in assessment.improvement_suggestions])

        return f"""
        请根据以下质量评估反馈，改进章节内容：

        **章节主题:** {chapter_task.section_title}

        **当前内容:**
        {chapter_task.content}

        **质量评估反馈:**
        {assessment.feedback}

        **改进建议:**
        {improvements_text}

        **当前质量评分:** {assessment.quality_score}

        请基于上述反馈重新撰写/改进章节内容，确保质量评分达到0.85以上。
        """

    async def handle_critical_error(self, error: Exception):
        """保守错误处理：遇到问题时暂停所有任务"""
        logger.error(f"检测到严重错误: {error}")
        logger.info("暂停所有并行任务，等待问题解决...")

        # 取消所有运行中的任务
        for task in asyncio.all_tasks():
            if task != asyncio.current_task() and not task.done():
                task.cancel()

        # 等待任务清理
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)

        raise ResearchError(f"研究过程因严重错误而暂停: {error}")

    async def monitor_progress(self):
        """实时监控任务执行进度"""
        while not self.all_tasks_completed():
            pending = len([t for t in self.tasks if t.status == TaskStatus.PENDING])
            running = len([t for t in self.tasks if t.status == TaskStatus.RUNNING])
            completed = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
            failed = len([t for t in self.tasks if t.status == TaskStatus.FAILED])

            progress_info = f"任务进度 - 待处理: {pending}, 运行中: {running}, 已完成: {completed}, 失败: {failed}"
            logger.info(progress_info)

            await asyncio.sleep(5)  # 每5秒更新一次进度

    def all_tasks_completed(self) -> bool:
        """检查所有任务是否已完成"""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for task in self.tasks)

    async def execute_parallel_research(self, sections: List[Dict]) -> List[ChapterTask]:
        """执行并行研究流程"""
        logger.info(f"开始并行研究，共{len(sections)}个章节，最大并发数: {self.max_concurrent}")

        # 创建章节任务
        self.tasks = await self.create_chapter_tasks(sections)

        # 启动进度监控
        monitor_task = asyncio.create_task(self.monitor_progress())

        try:
            # 创建并行任务
            processing_tasks = []
            for chapter_task in self.tasks:
                task = asyncio.create_task(self.process_single_chapter(chapter_task))
                processing_tasks.append(task)

            # 等待所有任务完成
            completed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # 分类任务结果
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    self.tasks[i].status = TaskStatus.FAILED
                    logger.error(f"任务异常: {result}")
                else:
                    self.tasks[i] = result

            # 统计结果
            self.completed_tasks = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
            self.failed_tasks = [t for t in self.tasks if t.status == TaskStatus.FAILED]

            logger.info(f"并行研究完成 - 成功: {len(self.completed_tasks)}, 失败: {len(self.failed_tasks)}")

            return self.tasks

        finally:
            # 取消进度监控
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

# --- 5. 深度研究核心流程 ---

async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    升级版深度研究流程：规划 -> 并行检索+ReAct评估 -> 整合。
    支持多章节并行处理和质量反馈优化。
    """
    logger.info(f"开始深度研究: {query}")

    # 1. 初步检索
    logger.info("Step 1: 进行初步检索...")
    try:
        initial_search_results_str = await async_search_jina(query)
        logger.info(f"初步检索完成，结果长度: {len(initial_search_results_str)}")
    except Exception as e:
        logger.error(f"初步检索失败: {e}")
        raise ResearchError(f"初步检索失败: {e}")

    # 2. 生成研究大纲
    logger.info("Step 2: 基于初步结果生成研究大纲...")

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
        outline_json = json.loads(outline_response.final_output.strip("```json").strip("```"))
        logger.info("研究大纲生成成功")

    except Exception as e:
        logger.error(f"大纲生成失败: {e}，使用默认结构")
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

    logger.info(f"报告标题: {research_title}")
    logger.info(f"规划了 {len(sections)} 个章节")

    # 3. 并行章节处理 + ReAct质量评估
    logger.info("Step 3: 启动并行章节处理和质量评估...")

    # 初始化并行研究管理器（使用激进策略参数）
    manager = ParallelResearchManager(
        max_concurrent=8,  # 激进并发策略
        max_react_rounds=4,  # 严格质量评估
        strict_error_handling=True  # 保守错误处理
    )

    try:
        # 执行并行研究
        chapter_tasks = await manager.execute_parallel_research(sections)

        # 检查是否有失败的章节
        if manager.failed_tasks and manager.strict_error_handling:
            failed_titles = [task.section_title for task in manager.failed_tasks]
            raise ResearchError(f"以下章节处理失败: {failed_titles}")

        # 收集完成的章节内容
        drafted_sections = []
        all_sources = []

        for task in chapter_tasks:
            if task.status == TaskStatus.COMPLETED:
                drafted_sections.append(f"## {task.section_title}\n\n{task.content}")
                all_sources.extend(task.sources)
                logger.info(f"章节 {task.section_title} 完成，质量评分: {task.quality_score:.2f}, ReAct轮次: {task.react_round}")
            else:
                # 即使有失败章节，在非严格模式下也包含错误信息
                error_content = f"## {task.section_title}\n\n章节处理失败: {task.content}"
                drafted_sections.append(error_content)
                logger.warning(f"包含失败章节: {task.section_title}")

    except Exception as e:
        logger.error(f"并行章节处理失败: {e}")
        raise ResearchError(f"并行章节处理失败: {e}")

    # 4. 报告整合与最终输出
    logger.info("Step 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    # 收集所有引用来源
    unique_sources = list(set(all_sources))

    final_prompt = f"""
    请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

    **报告标题:** {research_title}

    **已起草的章节内容:**
    {full_report_draft}

    **任务要求:**
    1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
    2. 保持各章节之间的连贯性和逻辑性。
    3. 在报告末尾添加一个**【结论与展望】**部分（如果大纲中没有）。
    4. 添加一个**【引用来源】**列表，包含以下URL: {unique_sources}
    5. 整体报告必须格式优美，使用 Markdown 格式。
    6. 确保报告的专业性和学术性。
    """

    try:
        final_report = await Runner.run(
            DeepResearchAgent,
            final_prompt,
        )
        logger.info("最终报告整合完成")
        return final_report.final_output
    except Exception as e:
        logger.error(f"最终报告整合失败: {e}")
        return f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"

async def test_parallel_system():
    """测试新并行系统的基本功能"""
    logger.info("=== 开始测试并行深度研究系统 ===")

    try:
        # 使用较小的测试主题
        test_topic = "Python异步编程的优势和应用场景"

        # 限制章节数量以便测试
        final_report = await deep_research(test_topic, max_sections=3)

        logger.info("=== 测试完成 ===")
        print("\n" + "="*80)
        print("生成的报告:")
        print("="*80)
        print(final_report)
        print("="*80)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"测试过程中出现错误: {e}")

async def main():
    """主函数 - 可选择运行测试或实际研究"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行测试模式
        await test_parallel_system()
    else:
        # 运行实际研究
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