from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test AI Agent",
    page_icon="🧪",
    layout="wide",
)

# ── Paths & env ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── Constants ─────────────────────────────────────────────────────────────
STATISTICAL_QUESTION = (
    "请帮我分析，A组和B组在单日 GMV 上是否存在统计学显著差异？"
    "请给出具体的 P-value，并告诉我你的最终业务建议。"
)

PREFIX = (
    "你是一个严谨的大厂高级数据分析师。你的核心职责是用统计学为业务决策兜底。\n"
    "当用户询问 A/B 测试的差异、效果或显著性时，绝对不能自己口算或猜测。\n\n"
    "你必须编写一段 Python 代码，在一个代码块中完成以下全部操作：\n"
    "1. import scipy.stats as stats。\n"
    "2. 先按「日期」和「实验组别」对 df 做 groupby，对 GMV(元) 求和，得到每日每组的 GMV。\n"
    "   然后分别提取 A 组和 B 组的单日 GMV 数组。\n"
    "3. 使用 stats.ttest_ind(group_a, group_b, equal_var=False) 做 Welch's t-test。\n"
    "4. 计算均值 mean_a = group_a.mean(), mean_b = group_b.mean()。\n"
    "5. 根据 p_value 判断显著性（p < 0.05 为显著），生成业务建议文本。\n"
    "6. 用 Python 的 f-string 拼接出完整的中文结论文本，格式如下，并用一个 print() 一次性输出：\n\n"
    "   print(\n"
    "       f'【统计检验结论】\\n'\n"
    "       f'A组单日GMV均值：{{mean_a:.2f}} 元\\n'\n"
    "       f'B组单日GMV均值：{{mean_b:.2f}} 元\\n'\n"
    "       f'P-value：{{p_value:.4f}}\\n'\n"
    "       f'结论：差异{{\"显著\" if p_value < 0.05 else \"不显著\"}}\\n'\n"
    "       f'业务建议：{{advice}}'\n"
    "   )\n\n"
    "   其中 advice 变量需要根据结果动态生成。\n\n"
    "代码执行完毕后，观察终端打印出的【统计检验结论】文本。"
    "你的 Final Answer 只需原样复制打印输出的全部内容（包括中文），"
    "以 'Final Answer:' 开头。禁止修改任何一个数字，禁止使用占位符。\n\n"
    "注意：DataFrame `df` 已加载全部数据，绝对不要从预览行重建 DataFrame，"
    "直接使用 `df` 变量进行聚合和分析。\n"
)

ALLOWED_METRICS = ["曝光量", "点击量", "转化量", "消耗金额(元)", "GMV(元)"]

# ── Session state init ────────────────────────────────────────────────────
DEFAULTS = {
    "df": None,
    "chat_history": [],
    "last_thought_steps": [],
    "last_output": "",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Agent builder ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未在 .env 中找到 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
    )


def build_agent(df: pd.DataFrame) -> AgentExecutor:
    llm = _get_llm()
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=8,
        return_intermediate_steps=True,
        prefix=PREFIX,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def run_agent(agent: AgentExecutor, question: str) -> tuple[str, list]:
    """返回 (output_text, intermediate_steps)。"""
    prompt = (
        "你是一名严谨的大厂高级数据分析师。"
        "请基于已加载的 DataFrame `df`（包含全部数据）"
        "严格遵循系统提示词中的统计学流程进行分析。"
        f"问题：{question}"
    )
    result = agent.invoke({"input": prompt})
    return result["output"], result.get("intermediate_steps", [])


# ── Statistics helpers ────────────────────────────────────────────────────
def compute_ab_stats(df: pd.DataFrame, metric: str) -> dict:
    """按实验组别汇总指标的 mean / std / sem / count。"""
    a_vals = df[df["实验组别"] == "A"][metric]
    b_vals = df[df["实验组别"] == "B"][metric]
    for arr in (a_vals, b_vals):
        if arr.empty:
            raise ValueError(f"未找到 {metric} 的 A/B 分组数据")

    def _stats(s: pd.Series) -> dict:
        return {
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "sem": s.sem(ddof=1),
            "n": len(s),
        }

    from scipy import stats as sp_stats

    t_stat, p_value = sp_stats.ttest_ind(a_vals, b_vals, equal_var=False)
    return {"A": _stats(a_vals), "B": _stats(b_vals), "p_value": p_value, "t_stat": t_stat}


def build_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Plotly 分组条形图 + 误差线（SEM）。"""
    stats = compute_ab_stats(df, metric)
    groups = ["A 组（对照）", "B 组（实验）"]
    means = [stats["A"]["mean"], stats["B"]["mean"]]
    errors = [stats["A"]["sem"], stats["B"]["sem"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=means,
            error_y={"type": "data", "array": errors, "visible": True},
            marker={"color": ["#636EFA", "#EF553B"]},
            text=[f"{v:,.2f}" for v in means],
            textposition="outside",
            name=metric,
        )
    )
    fig.update_layout(
        title=f"{metric} — A/B 组均值对比（误差线 = SEM）",
        yaxis_title=metric,
        template="plotly_white",
        height=480,
        margin={"t": 60, "b": 40},
    )
    return fig


# ── Report generation ─────────────────────────────────────────────────────
def build_markdown_report(output: str, df: pd.DataFrame, chart_fig: go.Figure | None = None) -> str:
    """将 AI 结论 + 数据摘要组合为 Markdown 报告。"""
    lines = [
        f"# A/B Test 分析报告",
        f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据行数**：{len(df)}  |  **A 组**：{(df['实验组别']=='A').sum()} 行  |  **B 组**：{(df['实验组别']=='B').sum()} 行",
        "",
        "---",
        "## AI 分析结论",
        "",
        output,
        "",
        "---",
        "## 数据摘要",
        "",
    ]
    # Group summary table
    summary = (
        df.groupby("实验组别")[ALLOWED_METRICS]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    lines.append(summary.to_markdown() if hasattr(summary, "to_markdown") else str(summary))
    lines.append("")
    lines.append("> 本报告由 A/B Test AI Agent 自动生成")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

st.title("🧪 A/B Test AI Agent")
st.caption("上传你的 A/B 测试 CSV，用自然语言与统计学 Agent 对话。")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 数据来源")
    uploaded = st.file_uploader("上传 A/B 测试 CSV 文件", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required_cols = {"日期", "用户ID", "实验组别", "曝光量", "点击量", "转化量", "消耗金额(元)", "GMV(元)"}
            missing = required_cols - set(df.columns)
            if missing:
                st.error(f"CSV 缺少必要字段：{missing}")
                st.session_state.df = None
            else:
                st.session_state.df = df
                st.success(f"已加载 {len(df)} 行数据")
        except Exception as exc:
            st.error(f"读取失败：{exc}")
            st.session_state.df = None

    # Load demo data if available
    demo_path = BASE_DIR / "ab_test_data.csv"
    if demo_path.exists() and st.session_state.df is None:
        if st.button("📥 加载演示数据（ab_test_data.csv）"):
            st.session_state.df = pd.read_csv(demo_path)
            st.success(f"已加载演示数据 {len(st.session_state.df)} 行")
            st.rerun()

    st.divider()

    st.header("🚀 一键诊断")
    run_diag = st.button("运行标准显著性分析", type="primary", disabled=st.session_state.df is None)

    st.divider()

    st.header("📊 指标对比")
    metric_choice = st.selectbox("选择对比指标", ALLOWED_METRICS, index=4)

    st.divider()

    # Export
    if st.session_state.last_output:
        st.header("📄 导出报告")
        md_report = build_markdown_report(
            st.session_state.last_output,
            st.session_state.df,
        )
        st.download_button(
            "下载 Markdown 报告",
            data=md_report,
            file_name=f"ab_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

# ── Main area ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

# ── Left: Chat ────────────────────────────────────────────────────────────
with col_left:
    st.subheader("💬 数据分析对话")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("输入你的问题，例如：B 组的转化率显著吗？")

    if user_input and st.session_state.df is not None:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Agent 思考中..."):
                agent = build_agent(st.session_state.df)
                output, steps = run_agent(agent, user_input)

            st.session_state.last_output = output
            st.session_state.last_thought_steps = steps
            st.markdown(output)

            # Thought process expander
            if steps:
                with st.expander("🧠 查看 Agent 思维链（Thought Process）", expanded=False):
                    for i, (action, obs) in enumerate(steps, 1):
                        st.markdown(f"**Step {i} — `{action.tool}`**")
                        with st.container(border=True):
                            st.code(action.tool_input, language="python")
                        st.caption(f"↳ 输出 ({len(str(obs))} 字符)")
                        st.text(str(obs)[:2000])
                        st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": output})
        st.rerun()

    elif user_input and st.session_state.df is None:
        st.warning("请先从侧边栏上传 CSV 数据。")

# ── Right: Charts & diagnostics ───────────────────────────────────────────
with col_right:
    st.subheader("📊 数据可视化")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Metric comparison chart
        try:
            fig = build_comparison_chart(df, metric_choice)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"绘图失败：{exc}")

        # Quick stats table
        with st.expander("📋 快速统计摘要", expanded=False):
            st.dataframe(
                df.groupby("实验组别")[ALLOWED_METRICS]
                .agg(["mean", "std", "min", "max"])
                .round(2),
                use_container_width=True,
            )

    else:
        st.info("上传数据后，这里会展示 A/B 组指标对比图表。")

# ── Run diagnostics ───────────────────────────────────────────────────────
if run_diag and st.session_state.df is not None:
    with col_left:
        with st.chat_message("assistant"):
            with st.spinner("正在运行标准显著性分析（Welch's t-test）..."):
                agent = build_agent(st.session_state.df)
                output, steps = run_agent(agent, STATISTICAL_QUESTION)

            st.session_state.last_output = output
            st.session_state.last_thought_steps = steps

            st.markdown("### 🚀 标准显著性分析结果")
            st.markdown(output)

            if steps:
                with st.expander("🧠 查看 Agent 思维链（Thought Process）", expanded=True):
                    for i, (action, obs) in enumerate(steps, 1):
                        st.markdown(f"**Step {i} — `{action.tool}`**")
                        with st.container(border=True):
                            st.code(action.tool_input, language="python")
                        st.caption(f"↳ 输出")
                        st.text(str(obs)[:3000])
                        st.divider()

    st.rerun()
