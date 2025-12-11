# AI大模型应用：NLP与大模型

2025年

内部资料， 请勿外传

# 文档解析工具：PaddleOCR

PaddleOCR 是基于飞桨（PaddlePaddle）深度学习开源框架的文字识别开发套件，是一套丰富、领先、实用的 OCR（光学字符识别）工具库。

PP-OCRv5：精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。

PaddleOCR-VL：超紧凑视觉语言模型 (VLM) 文档解析，支持超过 100 种语言，在识别复杂元素如文本、表格、公式和图表方面表现出色。

PP-StructureV3：复杂文档解析，能够将复杂的 PDF和文档图像智能转换为保留原始结构的 Markdown 或JSON 文件，完美保持文档版式和层次结构。

PP-ChatOCRv4：智能信息抽取，从海量文档中精准提取关键信息，实现文档问答等功能。

![](images/c2ce2377aa7d22a263e0f8abe08017702a3ac523eee857a7680632189738cc95.jpg)

Text Detection Image Detection Table Detection Formula Recognition Chart Recognition

<table><tr><td>Method Type</td><td>Methods</td><td colspan="2">Edit↓</td></tr><tr><td rowspan="2"></td><td></td><td>EN</td><td>ZH</td></tr><tr><td>PP-StructureV3</td><td>0.145 0.166</td><td>0.206 0.310</td></tr><tr><td rowspan="5">Pipeline Tools</td><td>MinerU-1.3.11 (Wang et al., 2024) MinerU-0.9.3 (Wang et al., 2024)</td><td>0.150</td><td>0.357</td></tr><tr><td>Mathpix1</td><td>0.191</td><td>0.365</td></tr><tr><td>Pix2Text-1.1.2.3 (breezedeus, 2022)</td><td>0.320</td><td>0.528</td></tr><tr><td>Marker-1.2.3 (Paruchuri, 2023)</td><td>0.336</td><td></td></tr><tr><td>Unstructured-0.17.2 (Unstructured-IO, 2022)</td><td>0.586</td><td>0.556 0.716</td></tr><tr><td rowspan="6"></td><td>OpenParse-0.7.0 (Filimoa, 2024)</td><td>0.646</td><td>0.814</td></tr><tr><td>Docling-2.14.0 (Docling Team, 2024)</td><td>0.589</td><td>0.909</td></tr><tr><td>GOT-OCR2.0 (Wei et al., 2024)</td><td>0.287</td><td>0.411</td></tr><tr><td>Mistral OCR²</td><td>0.268</td><td>0.439</td></tr><tr><td>OLMOCR-sglang (Poznanski et al., 2025)</td><td>0.326</td><td>0.469</td></tr><tr><td>SmolDocling-256M_transformer (Nassar et al., 2025) Nougat (Blecher et al., 2023)</td><td>0.493</td><td>0.816</td></tr><tr><td rowspan="4">General VLMs</td><td>Gemini2.5-Pro3</td><td>0.452 0.148</td><td>0.973 0.212</td></tr><tr><td>Gemini2.0-flash4</td><td></td><td></td></tr><tr><td>Qwen2.5-VL-72B (Yang et al., 2024)</td><td>0.191</td><td>0.264</td></tr><tr><td>GPT-405</td><td>0.214</td><td>0.261</td></tr><tr><td></td><td>InternVL2-76B (Chen et al., 2024)</td><td>0.233 0.440</td><td>0.399 0.443</td></tr></table>

Text Recognition Image Recognition Table Recognition Formula Recognition Chart Recognition

Text Line Orientation Classification Layout Analysis

# 文档解析工具：Qwen-VL OCR

通义千问OCR 是专用于文字提取的视觉理解模型，可从各类图像（如扫描文档、表格、票据等）中提取文本或解析结构化数据，支持识别多种语言，并能通过特定任务指令实现信息抽取、表格解析、公式识别等高级功能。

![](images/f9e5e7571dafbc9057c68e3638a232af2e2dc29a901a6ef325971064bf3f3390.jpg)

# 文档解析工具： MinerU

MinerU 是一款开源的文档处理工具，专注于将 PDF 文档高效地解析并转化为结构化的Markdown 格式。

结构化排版： 转换后的内容符合人类的阅读顺序，并最大限度地保留了原文档的结构和格式。

图像与表格提取： 能够提取 PDF 中的图像和表格，并在生成的 Markdown 文件中进行展示。

公式转换 (LaTeX)： 自动将 PDF 中的数学公式识别并转换为标准的 LaTeX 格式，方便在支持 LaTeX 的环境中显示和编辑。

![](images/402bfecb577d3fd19442b66fc66127aa6b3ec5f67429cc79f02ae08d89ab5dfb.jpg)

# 文档解析工具：DeepSeek-OCR

DeepEncoder（视觉编码器）： 负将二维文本图像映射成数量大大减少的视觉 tokens。窗口注意力 和 全局注意力编码器 的串行连接，以高效处理高分辨率输入。卷积压缩器 在进入全局注意力之前大幅减少视觉 tokens的数量。

DeepSeek3B-MoE-A570M（解码器）： 紧凑的 专家混合（MoE） 架构模型，它接收压缩后的视觉 tokens，并将其解码回原始的文本信息。MoE 设计使其能够高效推理并保持高准确率。

![](images/843517f2485cb6ad4103448a0d50c28a47297b4cb96ace088364682ec470fab8.jpg)

<table><tr><td rowspan="2">Model</td><td rowspan="2">Tokens</td><td colspan="3">English</td><td colspan="4">Chinese</td></tr><tr><td colspan="3">overall textformula tableorder</td><td>overall</td><td></td><td></td><td>textformula tableorder</td></tr><tr><td colspan="9">PiplineModels</td></tr><tr><td>Dolphin [11]</td><td></td><td>0.356 0.352</td><td>0.465</td><td>0.2580.35</td><td>0.44</td><td>0.44</td><td>0.604（</td><td>0.367 0.351</td></tr><tr><td>Marker[1]</td><td></td><td>0.296 0.085</td><td>0.374</td><td>0.609 0.116</td><td>0.497</td><td>0.293</td><td>0.688</td><td>0.678 0.329</td></tr><tr><td>Mathpix [2]</td><td></td><td>0.191 0.105</td><td>0.306</td><td>0.2430.108</td><td>0.364</td><td>0.381</td><td>0.454</td><td>0.32 0.30</td></tr><tr><td>MinerU-2.1.1[34]</td><td>=</td><td>0.162</td><td>0.072 0.313</td><td>0.166 0.097</td><td>0.244</td><td>0.111</td><td>0.581</td><td>0.150.136</td></tr><tr><td>MonkeyOCR-1.2B[18]</td><td></td><td>0.154</td><td>0.062 0.295</td><td>0.1640.094</td><td>0.263</td><td>0.179</td><td>0.464</td><td>0.168 0.243</td></tr><tr><td>PPstructure-v3 [9]</td><td></td><td>0.152 0.073</td><td>0.295</td><td>0.1620.077</td><td>0.223</td><td>0.136</td><td>0.535</td><td>0.111 0.11</td></tr><tr><td colspan="9">End-to-endModels</td></tr><tr><td>Nougat [6]</td><td>2352</td><td>0.452</td><td>0.365 0.488</td><td>0.572 0.382</td><td>0.973</td><td>0.998</td><td>0.941</td><td>1.000.954</td></tr><tr><td>SmolDocling [25]</td><td>392</td><td>0.493</td><td>0.262 0.753</td><td>0.7290.227</td><td>0.816</td><td>0.838</td><td>0.997</td><td>0.907 0.522</td></tr><tr><td>InternVL2-76B[8]</td><td>6790</td><td>0.44 0.353</td><td>0.543</td><td>0.547 0.317</td><td>0.443</td><td>0.29</td><td>0.701</td><td>0.5550.228</td></tr><tr><td>Qwen2.5-VL-7B [5]</td><td>3949</td><td>0.316 0.151</td><td>0.376</td><td>0.5980.138</td><td>0.399</td><td>0.243</td><td>0.5</td><td>0.627 0.226</td></tr><tr><td>OLMOCR [28]</td><td>3949</td><td>0.326 0.097</td><td>0.455</td><td>0.608 0.145</td><td>0.469</td><td>0.293</td><td>0.655</td><td>0.652 0.277</td></tr><tr><td>GOT-OCR2.0 [38]</td><td>256</td><td>0.287 0.189</td><td>0.360</td><td>0.459 0.141</td><td>0.411</td><td>0.315</td><td>0.528</td><td>0.52 0.28</td></tr><tr><td>OCRFlux-3B[3]</td><td>3949</td><td>0.238 0.112</td><td>0.447</td><td>0.269 0.126</td><td>0.349</td><td>0.256</td><td>0.716</td><td>0.162 0.263</td></tr><tr><td>GPT4o [26]</td><td>-</td><td>0.233 0.144</td><td>0.425</td><td>0.2340.128</td><td>0.399</td><td>0.409</td><td>0.606</td><td>0.3290.251</td></tr><tr><td>InternVL3-78B[42]</td><td>6790</td><td>0.218 0.117</td><td>0.38</td><td>0.279 0.095</td><td>0.296</td><td>0.21</td><td>0.533</td><td>0.282 0.161</td></tr><tr><td>Qwen2.5-VL-72B[5]</td><td>3949</td><td>0.214 0.092</td><td>0.315</td><td>0.341 0.106</td><td>0.261</td><td>0.18</td><td>0.434</td><td>0.262 0.168</td></tr><tr><td>dots.ocr [30]</td><td>3949</td><td>0.182 0.137</td><td>0.320</td><td>0.166 0.182</td><td>0.261</td><td>0.229</td><td>0.468</td><td>0.160 0.261</td></tr><tr><td>Gemini2.5-Pro [4]</td><td>=</td><td>0.148 0.055</td><td>0.356</td><td>0.130.049</td><td>0.212</td><td>0.168</td><td>0.439</td><td>0.119 0.121</td></tr><tr><td>MinerU2.0[34]</td><td>6790</td><td>0.133 0.045</td><td>0.273</td><td>0.15 0.066</td><td>0.238</td><td>0.115</td><td>0.506</td><td>0.209 0.122</td></tr><tr><td>dots.ocr+200dpi [30]</td><td>5545</td><td>0.125 50.032</td><td>0.329</td><td>0.0990.04</td><td>0.16</td><td>0.066（</td><td>0.416</td><td>0.0920.067</td></tr><tr><td colspan="9">DeepSeek-OCR (end2end)</td></tr><tr><td>Tiny</td><td>64</td><td>0.386</td><td>0.373 0.469</td><td>0.4220.283</td><td>0.361</td><td>0.307</td><td>0.635</td><td>0.2660.236</td></tr><tr><td>Small</td><td>100</td><td>0.221</td><td>0.142 0.373</td><td>0.2420.125</td><td>0.284</td><td>0.24</td><td>0.53</td><td>0.1590.205</td></tr><tr><td>Base</td><td>256(182)</td><td>0.137</td><td>0.267</td><td>0.1630.064</td><td>0.24</td><td>0.205</td><td>0.474</td><td>0.1 0.181</td></tr><tr><td>Large</td><td>400(285)</td><td>0.138</td><td>0.054 0.054</td><td>0.152 0.067</td><td>0.208</td><td>0.143</td><td>0.461</td><td>0.104 0.123</td></tr><tr><td>Gundam</td><td>795</td><td>0.127</td><td>0.277 0.043 0.269</td><td>0.134 0.062</td><td>0.181</td><td>0.097</td><td>0.432</td><td>0.089 0.103</td></tr><tr><td>Gundam-Mt200dpi</td><td>1853</td><td>0.123</td><td>0.049 0.242</td><td>0.147 0.056</td><td>0.157</td><td>0.087</td><td>0.377</td><td>0.080.085</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/16c98f338046622567119f5bc0ca6b4a0d695759989242df43fd16b831c0cc00.jpg)

(b) Performance on Omnidocbench

# 文档解析方法：HunyuanOCR

HunyuanOCR在一个轻量级框架内实现了对 OCR 核心能力的全面支持，包括 文本检测与识别 (Spotting)、文档解析 (Parsing)、信息抽取 (IE)、视觉问答 (VQA) 和图像文本翻译 (Translation)。

基于致密架构的 Hunyuan-0.5B 模型，引入 XD-RoPE，建立 1D 文本序列、2D 页面布局和 3D 时空信息的原生对齐机制，增强了模型处理复杂版面和文档逻辑推理的能力。

![](images/90941fff1c577861d2bb89c7cf6fed2c688a4103ba3fd4ab63612bd92cfa1548.jpg)

Table 1: Performance comparison of different VLMs and OCR systems across multiple tasks. $\scriptstyle { \hat { \alpha } }$ indicates Supported and High-Performing, $\textcircled { \scriptsize { \scriptsize { \sim } } }$ indicates Supported with Moderate Performance, and $\mathbf { a }$ indicates Supported but Underperforming. Otherwise,itis Not Supported.   

<table><tr><td rowspan="2">Model Type</td><td rowspan="2">Inference Type</td><td rowspan="2">Model Name</td><td rowspan="2">Deploytment Cost</td><td colspan="4">Task</td></tr><tr><td>Spotting Parsing Text-VQA IE Translation</td><td></td><td></td><td></td></tr><tr><td rowspan="5">Casecade Pipeline</td><td rowspan="5">Multi-Step</td><td>PaddleOCR-V5</td><td>low</td><td>C</td><td>=</td><td></td><td></td></tr><tr><td>BaiduOCR</td><td>low</td><td>中</td><td></td><td></td><td></td></tr><tr><td>Marker-1.8.2</td><td>low</td><td>☆</td><td></td><td></td><td></td></tr><tr><td>PP-ChatOCR</td><td>medium</td><td></td><td></td><td>☆</td><td></td></tr><tr><td>PP-DocTranslation</td><td>high</td><td></td><td></td><td></td><td>S</td></tr><tr><td rowspan="3">Specializedl VLMs (Modular)</td><td rowspan="3">two-stage</td><td>MonkeyOCR-pro-3B</td><td>medium</td><td></td><td>中</td><td></td><td></td></tr><tr><td>MinerU2.5</td><td>low</td><td></td><td>中</td><td></td><td></td></tr><tr><td>PaddleOCR-VL</td><td>low</td><td>1 中</td><td>=</td><td></td><td></td></tr><tr><td rowspan="3">General VLMs</td><td rowspan="3">One-Step</td><td>Gemini-2.5-Pro</td><td>high</td><td>☆</td><td>众</td><td>中</td><td>中</td></tr><tr><td>Seed-1.6-Vision</td><td>high</td><td>C</td><td>? 中</td><td>☆</td><td>中</td></tr><tr><td>Qwen3-VL-235B-Instruct</td><td>high</td><td>C </td><td>中</td><td></td><td>中</td></tr><tr><td rowspan="4">Specialized VLMs (End2End)</td><td rowspan="4">One-Step</td><td>Mistral-OCR</td><td>medium</td><td>=</td><td>C =</td><td>=</td><td>=</td></tr><tr><td>Deepseek-OCR</td><td>medium</td><td></td><td>S ☆ 中</td><td>☆</td><td></td></tr><tr><td>dots.ocr</td><td>medium</td><td>=</td><td></td><td></td><td></td></tr><tr><td>HunyuanOCR</td><td>low</td><td>中 ☆</td><td>中</td><td>中</td><td>中</td></tr></table>

Playwright 是由 Microsoft 团队开发的较新的开源框架，旨在解决现代 Web 应用（如单页应用 SPA）的测试挑战。

支持所有主流浏览器引擎： Chromium(Chrome/Edge) 、 Firefox 和 WebKit(Safari)。

跨平台 ： 支 持 Windows 、 Linux 和macOS 上的测试。

Playwright包括 Codegen（通过录制生成代码）、Playwright Inspector（实时调试）和Trace Viewer（详细的测试失败分析），可以可以轻松模拟和修改网络请求和响应，支持模拟各种移动设备的视口和用户代理。

# LangChain Web Browsing Agent

![](images/ea963fb3444a6791011b31a3e919b750d9a2834e1e17a4070bfee1688d241bb6.jpg)

![](images/9de3b38a67461a33003137ba82e79240048c70f8a1f2748dc3f1334d9d3f7086.jpg)

# Agent与Computer Use

Manus是一款由原中国团队 Monica 开发的通用人工智能代理产品（AI Agent）。与传统的聊天机器人不同，Manus 的突出特点是其自律性和多模态任务执行能力：

自律任务执行： 能够自主规划和执行多步骤任务，无需用户持续指导。它可以将任务分解为待办事项列表，并执行这些子任务以交付最终结果。

跨越思考到交付： 其他 AI 工具可能止步于头脑风暴，而 Manus 则能将想法付诸实践。

异步工作： 在云端异步工作，用户可以关闭设备，Manus 完成任务后会发出通知。

![](images/926523418430e056e3bf1dc7fc655d16297c4cd3f7b9a98ab6c7460bcc623f4f.jpg)

# 互联网搜索与网页解析

Jina AI 是一家专注于多模态人工智能和神经搜索技术的公司，致力于让开发者和企业能够轻松构建和部署先进的 AI 应用。它的核心目标是解决复杂数据（文本、图像、音频、视频等）的搜索、理解和处理问题。

<table><tr><td rowspan=1 colspan=1>Embeddings</td><td rowspan=1 colspan=1>Rerankers</td><td rowspan=1 colspan=1>ReaderLMs</td></tr><tr><td rowspan=1 colspan=1>jina-code-embeddings-1.5b (2025-09-01)HE|ArXiv| API Blog</td><td rowspan=1 colspan=1>jina-reranker-v3 (2025-10-03)HE|ArXiv |API| Blog</td><td rowspan=1 colspan=1>ReaderLM-v2 (2025-01-16)HF|ArXiv|API| Blog</td></tr><tr><td rowspan=1 colspan=1>jina-code-embeddings-0.5b (2025-09-01)HE|ArXiv|API| Blog</td><td rowspan=1 colspan=1> jina-reranker-m0 (2025-04-08)HE|AP| Blog</td><td rowspan=1 colspan=1>reader-lm-1.5b (2024-08-11)HF|Blog</td></tr><tr><td rowspan=1 colspan=1>jina-embeddings-v4 (2025-06-24)HE|ArXiv|API| Blog</td><td rowspan=1 colspan=1> jina-reranker-v2-base-multilingual (2024-06-25)HE|API| Blog</td><td rowspan=1 colspan=1>reader-lm-0.5b (2024-08-11)HF|Blog</td></tr><tr><td rowspan=1 colspan=1>jina-clip-v2 (2024-11-05)HF|ArXiv|API Blog</td><td rowspan=1 colspan=1>jina-reranker-v1-turbo-en (2024-04-18)HE|API| Blog</td><td rowspan=1 colspan=1>Reader API(2024-04-01)API</td></tr><tr><td rowspan=1 colspan=1>jina-embeddings-v3 (2024-09-18)HE|ArXiv|API| Blog</td><td rowspan=1 colspan=1>jina-reranker-v1-tiny-en (2024-04-18)HE|API| Blog</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1> jina-colbert-v2 (2024-08-31)HE|ArXiv|API Blog</td><td rowspan=1 colspan=1>jina-reranker-v1-base-en (2024-02-29)HF|API Blog</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>jina-clip-v1 (2024-06-05)HE|ArXiv|API| Blog</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

<table><tr><td>米</td><td>读取器API</td><td>https://r.jina.ai</td><td>将 URL转换为大模型友好文本</td></tr><tr><td>*</td><td>读取器API</td><td>https://s.jina.ai</td><td>搜索网络并将结果转换为大模型友好文本</td></tr><tr><td>米</td><td>深度搜索</td><td>https://deepsearch.jina.ai/v1/chat/completions</td><td>推理、搜索和迭代以找到最佳答案</td></tr><tr><td></td><td>向量模型API</td><td>https://api.jina.ai/vl/embeddings</td><td>将文本/图片转为定长向量</td></tr><tr><td>米</td><td>重排器API</td><td>https://api.jina.ai/vl/rerank</td><td>按查询对文档进行精排</td></tr><tr><td>開</td><td>分类器API</td><td>https://api.jina.ai/vl/train</td><td>使用训练样本训练分类器</td></tr><tr><td>開</td><td>分类器API (零样本)</td><td>https://api.jina.ai/vl/classify</td><td>使用零样本分类对输入进行分类</td></tr><tr><td>開</td><td>分类器API (少量样本)</td><td>https://api.jina.ai/v1/classify</td><td>使用经过训练的少样本分类器对输入进行分</td></tr><tr><td></td><td>切分器API</td><td>https://api.jina.ai/vl/segment</td><td>对长文本进行分词分句</td></tr></table>

# 互联网搜索与网页解析

SerpApi Documentation Integrations Features Pricing Use cases Resources

# APls

# Explore allsearch engines we support or check out our complete APl list.

# rch API

m our fast,easy，nd c

ocation

as,United States

# rch_metadata":{

Bing Images API   
Bing Copilot API   
Bing Search API   
DuckDuckGo Search API   
DuckDuckGo Light APl   
eeBay Search API   
Facebook Profile APl   
N Naver Search API   
· OpenTable Reviews API   
The Home Depot Search APl   
Tripadvisor Search API   
\*Walmart Search APl   
yYahoo! Search API   
Y Yandex Search APl   
Yelp Search API   
DYouTube Search API   
Extra APls   
Status and Error Codes   
G Google Search API   
G Google Light Search APl   
0 Google Ads Transparency APl   
Google Al Mode API   
Google Al Overview APl   
Google Autocomplete APl   
曲Google Events APl   
Google Forums APl   
Google Finance APl   
Google Flights API   
Google Hotels API   
Google Images APl   
Google Images Light APl   
Google Immersive Product APl   
Google Jobs API   
Google Lens API   
Google Local APl   
Google Local Services APl   
Google Maps API

★Google Maps Reviews APl 国Google News API 国Google News Light APl Google Patents APl Google Play Store APl Google Related Questions AP Google Reverse Image APl Google Scholar APl Google Shopping APl Google Shopping Light APl G Google Light Fast API Google Short Videos APl Google Travel Explore APl Google Trends API GoogleVideos APl Google Videos Light APl a Amazon Search APl $\textcircled{4}$ Apple App Store APl Baidu Search API

"6165916694c6c7025deef5ab" tatus":"Success" son_endpoint":"https://serpapi reated_at":"2021-10-1213:45: rocessed_at": "2021-10-12 13:45 bogle_url":"https://www.googl aw_html_file":"https://serpapi otal_time_taken":1.85

rch_parameters":   
ngine":"google", :"Coffee",   
ocation_requested":"Austin,Te   
ocation_used":"Austin,Texas,Ur   
oogle_domain":"google.com", "： "en" . "us"   
evice":"desktop"

ch_information":{ rganic_results_state":"Result

# google search

"knowledgeGraph" : "title": "Google Search", "type":"Website", "website": "https://google.com/", "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmIN-ig "description": "Google Search is a search engine provided and operated by "descriptionSource":"Wikipedia", "descriptionLink": "https://en.wikipedia.org/wiki/Google_Search" "attributes": "Written in": "Python，C，and C++", "Category": "Search engine", "Date launched": "1998" }

# https://serper.dev/

# 联网搜索

$\textcircled{1}$ 智谱AI为开发者提供全系列AI搜索工具，覆盖基础检索(Web Search API)、问答增强(Web Searchin Chat)、搜索智能体（Search Agent）三大服务，基于统一API接口集成自研引擎及第三方服务（搜狗/夸克)，提供从原始网页数据抓取、搜索结果与LLM生成融合、到多轮对话上下文管理的全链路能力，助力开发者以更低成本 构建可信、实时、可溯源的AI应用。

查看产品价格. 查看您的 API Key

# AI Search、DeepSearch 和 DeepResearch

AI Search / DeepSearch是指利用自然语言处理来提升搜索体验并提供更准确、相关和个性化的搜索结果的系统。

对话式搜索：通过生成式AI技术，AI Search可以像与智能朋友对话一样，直接回答用户的问题，提供简洁明了的答案。智能搜索优化：优化搜索结果的准确性和相关性，能够处理复杂的查询并快速提供最相关的信息。

DeepResearch是一款基于深度学习和智能推理技术的科研辅助工具，旨在提升信息搜索、分析和整合的效率。

多步骤推理与自动化搜索：能够通过联网搜索、数据分析和内容生成，快速完成复杂的研究任务。高效的信息整合与报告生成：能够整合多种信息来源，提炼关键信息，并生成高质量的技术报告或实验设计方案。

![](images/1cb01b8db2c38a3679cfa67eb34c94c87275e058eae7e34f163d4f17f6c7f669.jpg)

# AI Search、DeepSearch 和 DeepResearch

Tongyi DeepResearch 是阿里巴巴通义实验室开发的一款开源深度研究智能体，专为长周期、深层次的信息检索和研究任务设计。

ReAct 架构：基于 ReAct 框架，将推理（Thought）和行动（Action）交替进行，形成轨迹。  
上下文管理：采用动态上下文管理机制，通过马尔可夫状态重构，使代理在有限的上下文窗口内保持一致的推理能力。

Table 1: Performance comparison on various benchmarks.   

<table><tr><td> Benchmarks</td><td>Humanity&#x27;s Last Exam</td><td>Browse Comp</td><td>Browse Comp-ZH</td><td>GAIA</td><td>xbench DeepSearch</td><td>WebWalker QA</td><td>FRAMES</td></tr><tr><td colspan="8">LLM-based ReAct Agent</td></tr><tr><td>GLM 4.5</td><td>21.2</td><td>26.4</td><td>37.5</td><td>66.0</td><td>70.0</td><td>65.6</td><td>78.9</td></tr><tr><td>Kimi K2</td><td>18.1</td><td>14.1</td><td>28.8</td><td>57.7</td><td>50.0</td><td>63.0</td><td>72.0</td></tr><tr><td>DeepSeek-V3.1</td><td>29.8</td><td>30.0</td><td>49.2</td><td>63.1</td><td>71.0</td><td>61.2</td><td>83.7</td></tr><tr><td>Claude-4-Sonnet</td><td>20.3</td><td>12.2</td><td>29.1</td><td>68.3</td><td>65.0</td><td>61.7</td><td>80.7</td></tr><tr><td>OpenAI o3</td><td>24.9</td><td>49.7</td><td>58.1</td><td>1</td><td>67.0</td><td>71.7</td><td>84.0</td></tr><tr><td>OpenAI o4-mini</td><td>17.7</td><td>28.3</td><td>1</td><td>60.0</td><td>1</td><td>1</td><td>1</td></tr><tr><td colspan="8">DeepResearch Agent</td></tr><tr><td>OpenAI DeepResearch</td><td>26.6</td><td>51.5</td><td>42.9</td><td>67.4</td><td>1</td><td></td><td></td></tr><tr><td>Gemini DeepResearch</td><td>26.9</td><td>1</td><td>1</td><td>1</td><td>1</td><td></td><td>1</td></tr><tr><td>Kimi Researcher</td><td>26.9</td><td>1</td><td>1</td><td>1</td><td>69.0</td><td>一</td><td>78.8</td></tr><tr><td> Tongyi DeepResearch (30B-A3B)</td><td> 32.9</td><td> 43.4</td><td> 46.7</td><td>70.9</td><td>75.0</td><td> 72.2</td><td>90.6</td></tr></table>

![](images/aed780cc09cfd26c9bfcc758e86fcfffa1788ebc8e99b1b239f5d5b9a9e514f1.jpg)

# AI Search、DeepSearch 和 DeepResearch

DeepResearch 是一种基于人工智能技术的深度研究工具，它通过整合大型语言模型（LLM）、信息检索和自动化推理等技术，实现研究流程的自动化和增强。

智能知识发现：能够跨异构数据源自动执行文献检索、假设生成和模式识别。  
端到端流程自动化：将实验设计、数据采集、分析和结果解释整合为统一的 AI 驱动流程。  
协同智能增强：通过自然语言接口、可视化与动态知识表示促进人机协作。

![](images/4b1b5840f872fac4dc08dd8935bfaa639bcbc6a0442972e9b96498a8da3a6b6d.jpg)