<div align="center">

# 🔍 AlphaLens

**多智能体金融情报分析系统**

*输入一只股票，获取结构化风险报告。*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-编排引擎-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-前端-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md)

</div>

---

## 什么是 AlphaLens？

AlphaLens 使用 [LangGraph](https://github.com/langchain-ai/langgraph) 编排 **6 个专业 AI Agent**，从**社交媒体情绪**、**SEC 财报**、**量化指标**三个维度分析投资风险，交叉验证后生成结构化风险报告。

> 基于 LangGraph 构建的多智能体系统，调度 6 个专业 AI Agent，支持循环图路由、Pydantic 结构化输出、SEC 财报分析，将金融信号综合为风险报告，通过 LangSmith 追踪全链路。

<img width="1159" height="711" alt="Screenshot 2026-02-28 at 19 31 52" src="https://github.com/user-attachments/assets/9a583917-770c-44bb-bf07-f5bf2f5a82ee" />

### 核心亮点

- 🔄 **循环图** — Truth Checker 发现冲突时可将 Agent 送回重新调查
- ⚡ **并行扇出** — 3 个数据 Agent 同时执行，在 Truth Checker 处汇聚
- 🧩 **结构化输出** — 每个 Agent 返回 Pydantic 校验的 JSON Schema
- 🧠 **智能股票识别** — 支持自然语言输入（"苹果"、"最大的电动车公司"）→ 自动解析为股票代码
- 🖥️ **Streamlit 界面** — 实时流式日志、风险仪表盘、指标卡片、可下载报告
- 📊 **LangSmith 追踪** — LangSmith 仪表盘中完整展示执行链路
- 💰 **零成本** — 所有 API 均有慷慨的免费额度

<img width="1044" height="808" alt="Screenshot 2026-02-28 at 19 34 36" src="https://github.com/user-attachments/assets/2deac904-47d5-407e-8232-db1d7eae256c" />

---

## 系统架构

```
用户输入 (自然语言)
         │
         ▼
   ┌─────────────┐
   │  Supervisor  │ ← LLM 智能解析股票代码
   └──────┬──────┘
          │
    ┌─────┼─────┐          并行扇出
    ▼     ▼     ▼
┌──────┐┌──────┐┌──────┐
│情绪   ││ 量化 ││  SEC │
│侦察   ││ 分析  ││ 审计 │
│Scout ││Quant ││Audit │
└──┬───┘└──┬───┘└──┬───┘
   │       │       │
   └───────┼───────┘       扇入汇聚
           ▼
   ┌──────────────┐
   │ Truth Checker │ ← 交叉验证
   └──────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  有冲突？    无冲突
    │           │
    ▼           ▼
循环回到     ┌──────────┐
Supervisor  │ 报告生成  │
            │Generator │
            └──────────┘
```

---

## Agent 一览

| Agent | 角色 | 数据源 |
|:------|:-----|:-------|
| 🧠 **Supervisor** | 解析用户意图 → 股票代码，分发任务，管理重调查轮次 | Gemini LLM |
| 🔍 **Sentiment Scout** | 分析实时社交媒体情绪和新闻报道 | Grok `x_search` + Tavily |
| 📊 **Market Quant** | 计算价格走势、波动率、夏普比率、RSI-14、成交量趋势 | yfinance |
| ⚖️ **SEC Auditor** | 拉取 SEC EDGAR 财报 (10-K / 20-F)，提取风险因素和红旗 | SEC EDGAR + Tavily |
| 🔎 **Truth Checker** | 交叉验证 3 个 Agent 的报告，识别矛盾，给出行动建议 | Gemini LLM |
| 📋 **Report Generator** | 综合所有发现，生成 1-10 风险评级 | Gemini LLM |

---

## 快速开始

### 1. 克隆 & 安装

```bash
git clone https://github.com/YOUR_USERNAME/AlphaLens.git
cd AlphaLens
pip install -r requirements.txt
```

### 2. 配置 API Keys

在项目根目录创建 `.env` 文件：

```env
GOOGLE_API_KEY=你的_gemini_api_key
XAI_API_KEY=你的_grok_api_key
TAVILY_API_KEY=你的_tavily_api_key
LANGCHAIN_API_KEY=你的_langsmith_api_key   # 可选，用于链路追踪
```

### 3. 运行

**Web 界面（推荐）：**

```bash
streamlit run app_ui.py
```

**命令行模式：**

```bash
python main.py
```

---

## 项目结构

```
AlphaLens/
├── app_ui.py                 # Streamlit Web 界面
├── main.py                   # 命令行入口
├── requirements.txt
├── .env                      # API Keys（不提交到 Git）
│
├── app/
│   ├── config.py             # LLM 配置（Gemini 2.5 Flash）
│   ├── state.py              # Pydantic Schema + LangGraph 状态定义
│   ├── graph.py              # 图构建器 + 路由逻辑
│   ├── reporter.py           # Markdown 报告生成工具
│   │
│   └── agents/
│       ├── supervisor.py     # 任务分发 + 股票代码解析
│       ├── sentiment.py      # X 搜索 + Tavily 新闻情绪
│       ├── market_quant.py   # yfinance 量化分析
│       ├── sec_auditor.py    # SEC EDGAR 财报分析
│       ├── truth_checker.py  # 交叉验证 + 冲突检测
│       └── reporter.py       # 最终报告生成（LLM）
│
├── tests/
│   ├── test_*.py             # 单元测试（路由、错误处理、Pydantic 校验）
│   └── eval/                 # 三层评估框架
│       ├── eval_schemas.py   # LLM 评审 Pydantic Schema
│       ├── test_agent_quality.py      # 第一层：LLM-as-Judge 逐 Agent 评估
│       ├── test_truth_checker_eval.py # 第二层：冲突检测准确度
│       └── test_e2e_regression.py     # 第三层：全流程回归测试
```

---

## 界面预览

Streamlit 界面提供：

- **侧边栏** — 查询输入、API Key 管理、开始分析按钮
- **状态面板** — 实时流式日志，按 Agent 着色
- **风险仪表盘** — 1-10 风险评分 + 彩色进度条
- **指标卡片** — 三列布局展示 Market / Sentiment / SEC 数据
- **详情标签页** — 完整报告、Agent 推理链路、冲突分析
- **下载按钮** — 导出完整 Markdown 报告

---

## 测试与评估

AlphaLens 包含三层评估框架，衡量 Agent 输出质量，而非仅测试功能正确性。

```bash
# 运行全部评估测试（需要 API key + 足够的 Gemini 配额）
pytest tests/eval/ -v -s -m slow

# 仅运行单元测试（无 API 调用）
pytest tests/ --ignore=tests/eval -v
```

| 层级 | 测试内容 | 方法 |
|:-----|:---------|:-----|
| **Agent 质量评估** | 每个 Agent 输出的忠实度、完整度、合理度 | LLM-as-Judge（真实 API 调用 → Gemini 1-5 打分）|
| **Truth Checker 评估** | 已知场景下的冲突检测准确度 | Mock 报告 + 真实 LLM 交叉验证 |
| **端到端回归** | 代表性股票（AAPL、TSLA）全流程测试 | Pipeline 完整性 + LLM-as-Judge 评估最终报告 |

---

## 关键技术决策

| 决策 | 原因 |
|:-----|:-----|
| **LangGraph** 而非 CrewAI/AutoGen | 原生支持循环图、条件路由、并行扇出/扇入 |
| **Gemini 2.5 Flash** | 每天 1M 免费 token，结构化输出能力强 |
| **Grok `x_search`** | 唯一能原生访问 X/Twitter 实时数据的 API |
| **Pydantic 结构化输出** | Agent 间通信类型安全，无解析失败风险 |
| **`graph.stream()`** | 逐节点更新 UI，避免线程上下文问题 |
| **RSI-14（Wilder 法）** | 业界标准动量指标，非简单移动平均 |
| **20-F 支持** | 兼容外国上市公司（ADR 如 BABA、TSM） |

---

## 图路由工作原理

1. **Supervisor** 通过 LLM 将用户输入解析为股票代码。无法识别 → 图优雅退出。
2. **扇出**：Sentiment Scout、Market Quant、SEC Auditor 并行执行。
3. **扇入**：三份报告汇聚到 Truth Checker。
4. **Truth Checker** 评估一致性：
   - `consistent` / `minor_conflicts` → 进入 Report Generator
   - `needs_more_data`（仅第 1 轮）→ 循环回 Supervisor 进行第 2 轮调查
5. **Report Generator** 综合生成 1–10 风险评级。
6. Market Quant 在第 2 轮跳过重复计算（分钟级时间内 yfinance 数据不会变化）。

<img width="996" height="627" alt="Screenshot 2026-02-28 at 19 34 51" src="https://github.com/user-attachments/assets/3f2e7848-6ed7-4467-ad93-64bc169b87d3" />

---

## 许可证

MIT

---

<div align="center">

**基于 [LangGraph](https://github.com/langchain-ai/langgraph) · [Gemini](https://ai.google.dev/) · [Grok](https://x.ai/) · [Tavily](https://tavily.com/) 构建**

</div>
