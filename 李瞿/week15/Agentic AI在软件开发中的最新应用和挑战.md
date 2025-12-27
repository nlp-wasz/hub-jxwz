# Agentic AI在软件开发中的最新应用和挑战

# Agentic AI在软件开发中的最新应用和挑战

## 摘要
本文探讨了Agentic AI在软件开发中的最新应用和面临的挑战。Agentic AI通过模仿人类处理复杂问题的方式，将任务分解为更小、更易管理的步骤，从而提高输出质量和可靠性。本文首先介绍了Agentic AI的背景及其与传统大型语言模型（LLM）的区别，然后详细解析了Agentic AI的核心概念与机制，包括多智能体系统（MAS）、上下文工程以及关键机制如任务分解与执行、自主决策和工具集成。接着，文章讨论了Agentic AI的基础设施与实践，重点关注身份认证与授权管理以及安全威胁与缓解措施。此外，本文还探讨了Agentic AI在软件开发中的具体应用，包括其架构、关键组件以及实际部署中的考虑因素。最后，文章总结了Agentic AI面临的挑战，并提出了相应的解决方案。

## 引言与背景

近年来，随着人工智能（AI）技术的迅猛发展，特别是在自然语言处理领域的突破，一种新的AI系统——Agentic AI逐渐走入人们的视野。Agentic AI是指通过一系列深思熟虑、迭代的步骤执行任务的AI系统，它模仿人类处理复杂问题的方式，从而在多个领域展现出巨大的潜力和应用前景。

传统的大型语言模型（LLM）通常依赖于单一提示来生成完整的输出，例如从头到尾撰写一篇文章。然而，这种方式在处理复杂的、多步骤的任务时往往显得力不从心。相比之下，Agentic AI将任务分解为更小、更易管理的步骤，通过规划、研究、起草、修订等过程，结合人工反馈，最终产生更高质量的输出。这种迭代的过程不仅提高了输出的质量，还使得AI系统在专业任务中更加可靠和高效[1]。

Agentic AI的应用范围广泛，从法律文件分析、医疗研究到商业产品开发，其优势在于能够显著提升LLM的性能，并且可以通过并行化和模块化的方式优化工作流程。例如，在**Human Eval**编码基准测试中，GPT-3.5通过直接提示实现40%的准确率，而GPT-4通过直接提示提升至67%。但将GPT-3.5包装在Agentic工作流程中（例如，编写代码、反思和修订）可以超越GPT-4的直接提示，这表明工作流程设计可能比模型升级更具影响力[1]。

此外，Agentic AI系统可以根据任务需求的不同，采用不同程度的自主性。低自主代理的工作流程由人类工程师预定义，适用于结构化任务；而高自主代理则允许LLM决定行动序列，适用于更灵活和复杂的任务。半自主代理则在这两者之间取得平衡，既有一定的灵活性，又保持了一定程度的可控性[1]。

总之，Agentic AI作为一种新兴的AI范式，通过模仿人类处理复杂问题的方式，提供了一种更加高效和可靠的解决方案。它不仅在性能上有所提升，还在多个领域展现出了广泛的应用前景，有望在未来推动更多创新和发展。

---

[1]: [http://www.bilibili.com/read/cv43486683/](http://www.bilibili.com/read/cv43486683/)

## Agentic AI的核心概念与机制

### 1. 引言
Agentic AI是当前人工智能领域的一个重要发展方向，它不仅继承了生成式AI的强大语言理解与内容生成能力，还具备自主决策、任务分解和多智能体协作等复杂功能。本文将系统解析Agentic AI的核心概念及其运作机制。

### 2. Agentic AI的定义
Agentic AI是一种多智能体协作系统，能够动态分解任务并协同执行。相比于传统的单智能体系统（如AI Agents），Agentic AI通过多个专业智能体之间的协作，像人类团队一样分工拆解复杂目标，并能根据实际情况动态调整任务分配[1]。

### 3. Agentic AI的技术架构
#### 3.1 多智能体系统（MAS）
Agentic AI的基础架构之一是多智能体系统（MAS）。这种系统由多个具有自主性、感知力和通信能力的实体组成，可用于分布式问题解决。每个智能体都能够基于其目标制定计划，并与其他智能体进行协作以完成更复杂的任务[1]。

#### 3.2 上下文工程
随着Agentic AI应用复杂度的增加，上下文管理成为了一个关键挑战。上下文工程（Context Engineering）作为一种专门解决Agentic AI时代上下文管理挑战的技术方法论应运而生。其核心在于优化上下文的存储、检索及处理过程，从而提高系统的成本效益、性能以及可扩展性[2]。

##### 3.2.1 单Agent场景下的上下文管理
在单Agent架构中，除了维护对话历史外，还需要记录工具定义、每次工具调用的历史信息以及多步骤推理链等。这些新增的信息量级巨大，对上下文管理提出了新的要求[2]。

##### 3.2.2 多Agent场景下的上下文管理
当涉及到多Agent系统时，上下文管理变得更加复杂。每个Agent都需要持有完整的上下文模块，同时还要支持跨Agent的信息共享。这导致整个系统的上下文总量显著增加，进一步增加了管理和优化的难度[2]。

### 4. Agentic AI的关键机制
#### 4.1 任务分解与执行
Agentic AI能够将一个复杂的任务分解成多个子任务，并分配给不同的智能体来执行。这种能力使得Agentic AI特别适合处理那些需要多步推理或涉及多种技能的任务[1][2]。

#### 4.2 自主决策
不同于传统的被动响应型AI，Agentic AI具备一定程度的自主决策能力。这意味着它们可以根据当前环境状态和任务需求自行规划行动方案，而不仅仅是按照预设规则行事[1]。

#### 4.3 工具集成与调用
为了弥补自身知识上的局限性，Agentic AI通常会集成各种外部工具和服务（如API、搜索引擎等），并通过调用这些工具来增强其解决问题的能力[1]。

### 5. 面临的挑战与未来方向
尽管Agentic AI展现出巨大的潜力，但其发展也面临着诸多挑战，包括但不限于：
- **模型上下文窗口的物理限制**：即使是支持超长上下文的模型，在处理大规模内容时也可能遇到瓶颈。
- **成本压力**：随着上下文长度增加，大模型的使用成本也会相应上升。
- **用户体验**：如何在保证个性化服务的同时减少上下文冗余，提升用户满意度。
- **性能与准确率**：过长的上下文可能导致模型响应速度下降及推理准确性受损等问题。

针对上述挑战，研究人员正在探索各种解决方案，例如开发更加高效的上下文优化算法、改进现有技术框架等，以期推动Agentic AI向更加成熟的方向发展[2]。

---

[1]: https://blog.csdn.net/2401_85343303/article/details/151223025  
[2]: https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/

## Agentic AI的基础设施与实践

### 引言
随着大型语言模型（LLM）和多智能体系统（MAS）的发展，AI Agent正逐步演进为具有高度自主性的主动智能体（Agentic AI）。这些智能体能够自主思考、规划和执行复杂任务，甚至协同完成更复杂的目标。这种演进带来了前所未有的机遇，同时也引发了新的安全挑战，特别是在身份认证与授权管理方面。

近年来，多个安全事件揭示了Agentic AI系统的脆弱性。例如，2024年11月，LangChain生态中的LangSmith平台Prompt Hub暴露出严重的身份与权限管理漏洞“AgentSmith”。攻击者通过上传带有恶意代理配置的prompt，当用户fork并执行这些prompt时，用户的通信数据包括API密钥和上传内容会被悄然中转至攻击者控制的服务器，导致敏感信息泄露。此外，2025年披露的MCP Inspector远程命令执行漏洞（CVE-2025-49596）也显示了缺乏客户端与本地代理之间的认证带来的风险。这些事件突显了Agentic AI系统中身份认证与授权机制的重要性。[来源: https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-5/]

### AI Agent 身份管理的核心概念与技术要求

#### 2.1 身份与认证方面的概念与术语
在开发Agentic AI系统时，理解相关术语对于确保安全的身份认证与授权至关重要。以下是一些关键术语及其定义：

- **代理（Agent）**：一种由AI驱动的应用程序或自动化工作负载，通过访问云资源和第三方服务来代表用户执行任务。代理需要动态身份管理才能安全地访问跨多个信任域的资源。
- **代理身份（Agent Identity）**：AI代理或自动化工作负载的唯一标识符及其关联元数据。代理身份使代理能够以自身身份进行身份验证，而不是冒充用户。
- **代理身份目录（Agent Identity Directory）**：一个集中式注册目录，用于管理代理身份及其相关元数据和访问策略。
- **工作负载身份（Workload Identity）**：代理身份的底层技术实现，代表独立于特定硬件或基础架构的逻辑应用程序或工作负载。
- **访问令牌（Access Token）**：包含有关实体访问信息系统的授权信息的JSON Web令牌 (JWT)。
- **IAM角色**：提供短期有效凭据的访问亚马逊云科技云资源的方式，适合请求AWS资源。
- **API密钥**：一个唯一的标识符，用于验证对API的请求，允许应用程序访问特定服务。

#### 2.2 OAuth and Token 管理方面的概念和术语
OAuth 2.0是一种行业标准授权协议和框架，允许应用程序在不暴露用户凭据的情况下获得对外部服务用户帐户的有限访问权限。以下是相关术语及其定义：

- **OAuth 2.0 授权器（OAuth 2.0 Authorizer）**：一个SDK组件，用于对传入代理端点的OAuth 2.0 API请求进行身份验证和授权。
- **OAuth 2.0 客户端凭据授予（2LO）**：用于无需用户交互的机器对机器身份验证。代理使用2LO直接向资源服务器进行身份验证。
- **OAuth 2.0 授权码授予（3LO）**：需要用户同意和交互。例如，当客服人员需要明确的用户权限才能从Google日历或Salesforce等外部服务访问用户特定数据时，他们会使用3LO。

### Agentic AI的安全威胁与缓解措施
Agentic AI引入了新的安全威胁，传统的网络安全控制措施虽然仍然适用，但需要额外的防护措施。OWASP生成式AI安全工作组推出了Agentic AI安全行动（ASI），提供了基于威胁模型的新兴Agent威胁参考，并给出了相关的缓解措施。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]

#### 全面理解Agentic AI所特有的安全威胁
根据Backslash Security在2025年6月发布的MCP安全调研报告，全球范围内可被识别的MCP服务器已超过15,000个，其中超过7,000个直接暴露在互联网上，构成了巨大的攻击面。MCP协议作为Agentic AI系统的重要组成部分，其无监管生长催生了一个全新的、发展迅猛的、信任匮乏的软件供应链系统。

以下是OWASP总结的15个Agentic AI特有的安全威胁及其具体案例和影响：

- **记忆投毒（Memory Poisoning）**：利用人工智能的短期和长期记忆系统，投入恶意或虚假数据，可能导致决策被篡改和未经授权的操作。例如，攻击者可以注入伪造的历史数据，使AI Agent做出错误的业务决策，导致经济损失。为了防范此类攻击，应定期审查和清理AI的记忆库，并实施数据验证机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **工具滥用（Tool Misuse）**：攻击者在授权权限范围内，通过欺骗性提示或命令操纵AI Agent，滥用其集成工具。例如，攻击者可以诱使AI Agent发送大量垃圾邮件或执行恶意脚本。可以通过限制AI Agent的权限范围、实施严格的输入验证和监控异常行为来减轻此类威胁。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **权限滥用（Privilege Compromise）**：攻击者利用权限管理中的弱点执行未经授权的操作。例如，攻击者可能通过社会工程学手段获取高权限账户，进而控制整个系统。建议采用最小权限原则，并定期审计权限分配。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **资源过载（Resource Overload）**：利用人工智能系统的资源密集型特性，攻击其计算、内存和服务能力，从而降低性能或导致故障。例如，攻击者可以通过DDoS攻击使AI Agent无法处理正常的请求。建议实施流量管理和负载均衡策略，以应对突发的高流量。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **级联幻觉攻击（Cascading Hallucination Attacks）**：利用人工智能倾向于生成看似合理但却是虚假的信息，这些信息会在系统中传播并扰乱决策。例如，攻击者可以诱导AI Agent生成虚假的财务报告，导致企业决策失误。建议实施多层次的数据验证和交叉检查机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **破坏意图和操纵目标（Intent Breaking & Goal Manipulation）**：利用Agentic AI的规划和目标设定能力中的漏洞，使攻击者能够操纵或改变Agent的目标和推理。例如，攻击者可以修改AI Agent的任务目标，使其执行有害操作。建议实施严格的目标验证和审计机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **不协调和欺骗行为（Misaligned & Deceptive Behaviors）**：Agentic AI利用推理和欺骗性反应来执行有害或不允许的操作。例如，攻击者可以诱导AI Agent发送误导性信息，导致用户做出错误决策。建议实施行为监控和异常检测机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **否认与不可追踪（Repudiation & Untraceability）**：由于日志记录不足或决策过程透明度低，导致Agentic AI执行的操作无法追溯或解释。例如，攻击者可以删除日志记录，掩盖其非法活动。建议实施全面的日志记录和审计机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **身份欺骗和冒充（Identity Spoofing & Impersonation）**：攻击者利用身份验证机制冒充Agentic AI或人类用户，从而以虚假身份执行未经授权的操作。例如，攻击者可以冒充管理员身份，获取敏感数据。建议实施多因素身份验证和持续的身份验证机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **过度的人类监督（Overwhelming Human in the Loop）**：针对具有人类监督和决策验证的系统，旨在利用人类的认知局限性或破坏交互框架。例如，攻击者可以通过大量的虚假警报淹没人类监督者，使其无法及时响应真实威胁。建议实施自动化辅助决策和异常检测机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **非预期的远程代码执行和代码攻击（Unexpected RCE and Code Attacks）**：攻击者利用人工智能生成的执行环境注入恶意代码、触发非预期的系统行为或执行未经授权的脚本。例如，攻击者可以注入恶意脚本，导致系统崩溃。建议实施严格的代码审查和运行时保护机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]
- **Agent通信投毒（Agent Communication Poisoning）**：攻击者操纵Agentic AI之间的通信渠道，导致信息传递错误或恶意指令的传播。例如，攻击者可以拦截和篡改通信数据，导致AI Agent执行错误操作。建议实施加密通信和完整性校验机制。[来源: https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]

### 结论
Agentic AI的基础设施与实践需要重点关注身份认证与授权管理，以及应对新的安全威胁。通过理解相关术语和技术要求，并采取适当的防护措施，可以有效提升Agentic AI系统的安全性。[来源: https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-5/ 和 https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/]

### 核实引用的时间点和事件的真实性
请注意，文中提到的2024年11月和2025年的事件是示例，实际事件的真实性和时间点需要进一步核实。如果这些事件是虚构的示例，应明确标注为示例，并提供真实的类似事件作为参考。

## Agentic AI在软件开发中的应用

### 介绍
Agentic AI 是一种新型的AI系统，它能够自主规划、执行和适应任务，以实现用户定义的目标。这种技术的发展为软件开发带来了新的机遇和挑战。本文将探讨Agentic AI在软件开发中的应用，包括其架构、关键组件以及实际部署中需要注意的问题。

### Agentic AI 的定义与特点

#### 定义
Agentic AI是一种能够自主决策并采取行动的AI系统，旨在减少对人类监督的需求。这类系统可以审查目标的当前状态，并根据评估结果制定适当的决策，例如增加新步骤或寻求其他AI系统或人类的帮助[来源: <https://www.oracle.com/cn/artificial-intelligence/agentic-ai/>]。

#### 特点
- **自主性**：Agentic AI能够在没有持续的人工干预的情况下运作。
- **目标导向**：该系统能够主动地设定并追求特定的目标。
- **适应性**：基于环境变化和反馈，Agentic AI能够调整其行为策略。
- **协作能力**：能与其他AI代理及人类合作完成复杂任务。

### Agentic AI 在软件开发中的架构
Agentic AI系统的架构主要包括四个核心模块：推理引擎、记忆系统、编排模块以及工具接口。

1. **推理引擎**：作为Agent的大脑，负责理解用户意图、制定执行计划等。通常基于大语言模型（LLMs）构建，需要精心设计提示词模板来优化性能[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。
   
2. **记忆系统**：分为短期记忆和长期记忆两部分，分别用于维护会话上下文和存储历史交互数据。这要求开发者设计高效的检索算法和信息更新策略[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。

3. **编排模块**：负责协调各个组件的工作流程，管理整体执行过程。这涉及到工作流设计、异常处理等多个方面[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。

4. **工具接口**：允许Agent与外部世界交互，调用API、数据库等资源。标准化不同工具接入方式是这一环节的关键挑战之一[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。

### 实际部署中的考虑因素
为了确保Agentic AI应用顺利从原型阶段过渡到生产环境中，除了上述核心技术外，还需要关注以下几个方面：

- **质量评估**：建立自动化结合人工审核的质量保证体系，持续监控推理质量和任务完成率[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。
- **身份认证与授权**：解决“谁可以访问Agent”以及“Agent可以访问哪些资源”的问题，保障多租户环境下的安全性[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。
- **安全与隐私保护**：针对潜在威胁如记忆投毒等实施分层防护策略，在各个环节设置独立的安全过滤机制[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。
- **可观测性**：通过追踪推理链路等方式可视化Agent的行为过程，对于调试和优化至关重要[来源: <https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/>]。

### 结论
随着Agentic AI技术不断成熟，其在软件开发领域的应用越来越广泛。通过合理设计系统架构并注重实际部署时的安全性和可维护性，企业可以充分利用Agentic AI的优势，提高生产力和服务水平。未来，随着更多创新实践的出现，Agentic AI将在更广泛的场景下发挥重要作用。

## Agentic AI面临的挑战与解决方案

### 引言
Agentic AI通过多智能体协作、动态任务分解、持久记忆和协调自主性，克服了传统AI Agents在处理复杂、多步骤或需要协作的任务时的局限性。然而，这种范式转变也带来了新的挑战。本章节将探讨这些挑战，并提出相应的解决方案。

### 挑战一：多智能体间的协调与通信
**挑战描述**：在Agentic AI系统中，多个智能体需要协同工作以完成复杂的任务。这要求智能体之间能够高效地进行通信和协调。然而，实现这一目标并不容易，特别是在涉及大量智能体的情况下，通信延迟、数据同步和决策冲突等问题可能会出现 [来源: https://developer.volcengine.com/articles/7508126689751203891]。

**解决方案**：
- **分布式通信渠道**：使用异步消息队列、共享内存缓冲区或中间输出交换等技术来实现智能体之间的通信。这种方式可以减少通信延迟，并确保信息的一致性和可靠性。
- **协调算法**：引入高效的协调算法，如基于共识的协议（例如Paxos或Raft），以解决决策冲突问题。此外，还可以采用集中式调度器来管理任务分配和资源调度，从而提高系统的整体效率。

### 挑战二：目标分解与任务规划
**挑战描述**：用户指定的目标需要被自动解析并分解为更小的子任务，然后分配给不同的智能体。这个过程涉及到复杂的推理和规划，尤其是在面对动态环境时，如何有效地调整任务顺序和应对部分任务失败是一个难题 [来源: https://developer.volcengine.com/articles/7508126689751203891]。

**解决方案**：
- **自适应任务分解**：开发自适应的任务分解算法，使其能够根据当前环境状态和任务执行情况动态调整任务分解方案。例如，可以利用强化学习方法来优化任务分解策略，以最大化任务完成的成功率。
- **多步骤推理引擎**：构建强大的多步骤推理引擎，支持智能体在面对复杂情境时进行深层次的逻辑推理。这可以通过结合符号逻辑和统计学习方法来实现，以增强系统的理解和决策能力。

### 挑战三：持久记忆与上下文理解
**挑战描述**：为了实现长期有效的任务管理和优化，Agentic AI系统需要具备持久记忆功能，即能够在多次交互中保存和利用上下文信息。然而，如何有效地存储和检索这些信息，并保证其准确性和时效性，是该领域面临的一个重要挑战 [来源: https://developer.volcengine.com/articles/7508126689751203891]。

**解决方案**：
- **上下文工程**：采用先进的上下文工程技术，如知识图谱和语义网络，来组织和表示长期记忆中的信息。这样不仅便于信息的快速检索，还能帮助系统更好地理解上下文关系。
- **增量学习机制**：引入增量学习机制，使系统能够不断从新数据中学习并更新其知识库。这种方法有助于保持系统对最新信息的敏感度，同时避免了重新训练整个模型带来的高昂成本。

### 结论
尽管Agentic AI在处理复杂任务方面展现出巨大潜力，但要充分发挥其优势还需克服一系列技术挑战。通过采用适当的通信与协调机制、自适应任务分解算法以及高效的持久记忆技术，我们可以逐步解决这些问题，推动Agentic AI向更加成熟和完善的方向发展。

## 结论与展望
Agentic AI作为一种新兴的AI范式，通过模仿人类处理复杂问题的方式，提供了一种更加高效和可靠的解决方案。它不仅在性能上有所提升，还在多个领域展现出了广泛的应用前景。然而，Agentic AI的发展也面临着诸多挑战，包括多智能体间的协调与通信、目标分解与任务规划、持久记忆与上下文理解等方面。通过采用适当的解决方案，如分布式通信渠道、自适应任务分解算法和上下文工程技术，可以逐步解决这些问题，推动Agentic AI向更加成熟和完善的方向发展。

未来，随着技术的不断进步和创新实践的涌现，Agentic AI将在更多领域发挥重要作用，为企业和社会带来更大的价值。我们期待看到更多关于Agentic AI的研究和应用，共同推动这一领域的进一步发展。

## 引用来源
1. [http://www.bilibili.com/read/cv43486683/](http://www.bilibili.com/read/cv43486683/)
2. [https://blog.csdn.net/2401_85343303/article/details/151223025](https://blog.csdn.net/2401_85343303/article/details/151223025)
3. [https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/](https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/)
4. [https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-5/](https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-5/)
5. [https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/](https://aws.amazon.com/cn/blogs/china/privacy-and-security-of-agent-applications/)
6. [https://www.oracle.com/cn/artificial-intelligence/agentic-ai/](https://www.oracle.com/cn/artificial-intelligence/agentic-ai/)
7. [https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/](https://aws.amazon.com/cn/blogs/china/agentive-ai-infrastructure-practice-series-1/)
8. [https://developer.volcengine.com/articles/7508126689751203891](https://developer.volcengine.com/articles/7508126689751203891)