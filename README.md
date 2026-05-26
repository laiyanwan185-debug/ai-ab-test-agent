# 🧪 A/B Test AI Agent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V3-4B32C3?logo=deepseek&logoColor=white)](https://platform.deepseek.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-6.0+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![SciPy](https://img.shields.io/badge/SciPy-1.17+-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

> 一款结合 **LLM 推理** 与 **严谨统计学归因（P-value）** 的 A/B Test 智能分析工具。
> 自然语言驱动，物理计算兜底——让 AI 替你跑显著性检验，数字绝不瞎编。

---

## ✨ 核心功能亮点

### 1. 全自动数据归因

用自然语言提问，Agent 自动编写并执行 Python 代码：

> *"B 组的转化率显著吗？"* → Agent 自动跑 Welch's t-test → 输出 P-value + 业务建议

### 2. 统计学防幻觉 — print-copy 策略

```
┌──────────┐     ┌────────────────┐     ┌──────────────┐
│  LLM 推理 │ ──▶ │ Python 物理计算 │ ──▶ │ Final Answer │
│  (规划)   │     │ (scipy + numpy) │     │ (原样复制)    │
└──────────┘     └────────────────┘     └──────────────┘
```

**所有数值由 Python `f-string` 生成，LLM 仅做格式转发**——杜绝 AI "拍脑袋" 编数字。

### 3. 可视化看板

- **Plotly 分组条形图** + SEM 误差线，一眼看出组间差异
- 可折叠的 **思维链（Thought Process）** 面板，完整展示 Agent 的推理路径
- **一键标准诊断**：侧边栏点击即可运行完整显著性分析

### 4. 报告导出

一键下载 **Markdown 格式** 分析报告，包含 AI 结论 + 数据摘要。

---

## 🏗️ 技术架构

```
┌──────────┐      ┌─────────────────────────────────────┐
│   User   │      │         LangChain Agent              │
│  (自然语言) │      │                                     │
│          │─────▶│  ┌──────────┐    ┌───────────────┐  │
│  提问/上传  │      │  │  ReAct   │───▶│ python_repl   │  │
│          │      │  │  Planner │    │   _ast        │  │
└──────────┘      │  └──────────┘    └───────┬───────┘  │
                  │                          │          │
                  │                   ┌──────▼───────┐  │
                  │                   │ scipy.stats  │  │
                  │                   │ ttest_ind()  │  │
                  │                   │ (Welch's)    │  │
                  │                   └──────┬───────┘  │
                  │                          │          │
                  │  ┌──────────┐    ┌──────▼───────┐  │
                  │  │  Final   │◀───│ f-string     │  │
                  │  │  Answer  │    │ print-copy   │  │
                  │  └──────────┘    └──────────────┘  │
                  └─────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Streamlit UI │
                  │  · 对话窗口    │
                  │  · Plotly 图表 │
                  │  · Markdown 导出│
                  └───────────────┘
```

| 层级 | 技术 | 职责 |
|---|---|---|
| **UI** | Streamlit | 对话窗口、文件上传、图表渲染、报告导出 |
| **编排** | LangChain ReAct Agent | 解析用户意图、规划代码执行步骤 |
| **计算** | Python REPL + SciPy | 物理执行统计检验，生成精确数值 |
| **可信层** | print-copy 策略 | Python f-string 生成完整中文结论，LLM 原样转发 |
| **模型** | DeepSeek Chat (OpenAI API) | 推理与规划 |

---

## 🚀 如何运行

### 前置要求

- Python 3.11+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）

### 第一步：配置密钥

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY="sk-your-api-key-here"
```

### 第二步：安装依赖

```bash
pip install -r requirements.txt
```

### 第三步：生成演示数据（可选）

```bash
python generate_ab_data.py
```

> 这会生成 `ab_test_data.csv`（500 行电商 A/B 测试数据，B 组转化率略优于 A 组）。

### 第四步：启动 Web 应用

```bash
streamlit run app.py
```

浏览器访问 **http://localhost:8501**。

---

## 📁 项目结构

```
ai-ab-test-agent/
├── app.py                   # Streamlit Web 主程序
├── agent_core.py            # LangChain Agent 核心（CLI 版本）
├── generate_ab_data.py      # 模拟 A/B 测试数据生成器
├── requirements.txt         # 依赖清单
├── .env                     # API 密钥（不入库）
├── ab_test_data.csv         # 演示数据
└── README.md
```

---

## 💡 使用场景

| 场景 | 示例问题 |
|---|---|
| **转化率分析** | "A 组和 B 组的转化率有没有显著差异？" |
| **GMV 归因** | "B 组的单日 GMV 是否显著高于 A 组？P-value 多少？" |
| **消耗效率** | "两组在消耗金额上的投资回报率谁更高？" |
| **自定义指标** | "帮我分析点击量的组间差异是否显著。" |

---

## 🛡️ 为什么数字不会瞎编？

传统的 LLM + Data 方案中，模型常常在看到代码输出后仍然**用自然语言改写数字**，导致偏差。

本项目采用 **print-copy 策略**：

1. **Prefix 严格约束** — 系统提示词强制 Agent 在一个 Python 代码块中完成全部计算和文本拼接
2. **Python f-string 生成结论** — 所有数值（均值、P-value）由 `scipy` 计算后通过 `f"{mean_a:.2f}"` 嵌入
3. **LLM 只做转发** — Final Answer 必须原样复制 Python 输出的【统计检验结论】

> 相当于：LLM 负责"解题思路"，Python 负责"对答案"，最终交给用户的是 Python 的答案而非 LLM 的答案。

---

<p align="center">
  <sub>Built with ❤️ using LangChain · DeepSeek · Streamlit · SciPy</sub>
</p>
