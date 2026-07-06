# Agentic RAG 求职分析助手 — 设计 Spec

## 项目概述

上传简历和 JD，AI Agent 自主检索多数据源，完成岗位匹配度分析和模拟面试。

## 技术栈

| 层 | 选型 |
|---|---|
| 前端 | Vue 3 + Vite + TypeScript |
| 后端 | FastAPI + Uvicorn + SSE |
| Agent 框架 | LangChain + LangGraph |
| 向量数据库 | ChromaDB (嵌入式) |
| Embedding | DeepSeek Embedding API (1536维) |
| 文档解析 | PyMuPDF + python-docx |
| Web 搜索 | Tavily Search API |
| LLM | DeepSeek Chat + Reasoner |
| 部署 | Docker Compose (双容器) |

## 核心功能流程 (4步)

1. **上传简历 & JD** — 支持 PDF/DOCX 文件上传 + 纯文本粘贴，系统自动解析提取结构化信息
2. **岗位匹配度分析** — Agent 检索简历+JD+Web(公司背景+面经)，输出匹配度评分、技能 GAP、改进建议、准备清单
3. **模拟面试** — Agent 基于 JD 要求+简历缺口+公司面试风格出题，逐题发问，AI 点评每道回答
4. **面试报告 & 改进计划** — 整体评分、逐题反馈、薄弱环节分析、后续复习计划

## 系统架构

四层架构：
- Vue 3 前端 (Port 5173) -> FastAPI 后端 (Port 8000) -> LangChain Agent 决策层 -> ChromaDB 检索+数据层

Agent 决策循环：理解意图 -> 规划检索 -> 并行检索(向量+关键词) -> 融合排序(RRF) -> Rerank -> 反思评估(不够则循环) -> 生成回答

混合检索流程：向量 Top20 + BM25 关键词 -> RRF 融合去重 -> BGE-Reranker 精排 Top5 -> 送入 LLM

## Agent 设计 (LangGraph 状态图)

状态节点：classify -> plan -> search -> fuse -> reflect -> generate

### 5 个工具
1. **search_resume** — 向量检索简历知识库
2. **search_jd** — 向量+关键词检索 JD 知识库
3. **web_search** — Tavily 搜索互联网信息(公司背景/面经/行业趋势)
4. **compare_skills** — 对比简历技能 vs JD 要求，输出结构化差异
5. **generate_question** — 基于技能缺口生成面试题

### 反思机制 4 维度
- 覆盖度：检索结果是否覆盖所有子问题？不满足 -> 二次检索
- 一致性：不同来源有无矛盾？有矛盾 -> 标注、优先权威来源
- 时效性：信息是否过时？过时 -> 加时间过滤重新搜索
- 充分性：现有信息足够高质量回答吗？不够 -> 换关键词/数据源

## 数据模型

### ChromaDB — 4 个 Collection
| Collection | 内容 | 检索方式 |
|---|---|---|
| resume_chunks | 简历按节切块 | 向量 + 关键词 |
| jd_chunks | JD 语义切块 | 向量 + 关键词 |
| interview_exp | 面经片段(Web搜索缓存) | 向量 + 元数据过滤 |
| company_info | 公司背景(Web搜索缓存) | 向量 + 元数据过滤 |

### 分块策略
- 简历：按节切分(个人信息/技能/经历/项目/教育)，~500 tokens，无重叠
- JD/面经：语义分块 + 滑动窗口，~800 tokens，100 tokens 重叠

## API 设计 (8 个端点)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/resume/upload | 上传简历(文件或文本) |
| POST | /api/jd/upload | 上传 JD(文件或文本) |
| POST | /api/match | 启动匹配度分析(异步) |
| GET | /api/match/{task_id}/stream | SSE 流式获取分析进度 |
| GET | /api/match/{task_id}/report | 获取匹配度报告 JSON |
| POST | /api/interview/start | 开始模拟面试 |
| POST | /api/interview/{session_id}/answer | 提交回答，获取点评+下一题 |
| GET | /api/interview/{session_id}/report | 获取面试报告 |

### SSE 事件类型
- `progress` — 阶段进度 (parsing/indexing/analyzing)
- `thinking` — Agent 实时动作 (search/reflect/generate)，透明化思考过程
- `result` — 最终结构化结果
- `done` — 流结束

## 前端页面

4 个路由：
- `/` — 上传页：拖拽上传 + 文本粘贴，简历和 JD 双输入区
- `/match/:taskId` — 匹配度报告：评分圆环、技能缺口、公司背景、面经摘要
- `/interview/:sessionId` — 模拟面试：对话式界面，题目在左、回答区在右
- `/report/:sessionId` — 面试报告：总分、逐题反馈、薄弱项、复习计划

## 前端组件树

```
App.vue
+-- AppHeader.vue (Logo + 导航)
+-- router-view
    +-- UploadPage.vue
    |   +-- FileDropZone.vue (拖拽上传)
    |   +-- TextPasteArea.vue (文本粘贴)
    +-- MatchReport.vue
    |   +-- ScoreCircle.vue (评分圆环)
    |   +-- SkillGapChart.vue (技能缺口雷达图)
    |   +-- InfoCard.vue x4 (公司背景/面经/建议/缺口)
    |   +-- ThinkingStream.vue (Agent 思考过程实时展示)
    +-- InterviewPage.vue
    |   +-- ChatBubble.vue (对话气泡)
    |   +-- QuestionCard.vue (当前题目)
    |   +-- AnswerInput.vue (回答输入+发送/跳过)
    |   +-- FeedbackCard.vue (AI 点评)
    +-- InterviewReport.vue
        +-- ScoreSummary.vue (总分)
        +-- QuestionReview.vue x N (逐题回顾)
        +-- WeakAreaAnalysis.vue (薄弱项)
        +-- StudyPlan.vue (复习计划)
```

## 非功能性要求

- 单用户，无登录/注册
- 专业 UI 风格，面试场景严肃正式
- SSE 流式输出，避免长时间等待
- ChromaDB 持久化到本地磁盘
- 简历和 JD 数据本地存储，不上传第三方
