from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Fix Windows console encoding for emoji / Unicode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ab_test_data.csv"
QUESTION = "请用中文告诉我，A组和B组各自的总曝光量和总GMV是多少？谁的投资回报率(ROI)更高？"

PREFIX = (
    "You are working with a pandas dataframe `df` in Python. "
    "The dataframe `df` contains the FULL dataset (500 rows). "
    "Do NOT recreate a dataframe from the sample shown — "
    "always query `df` directly because it already holds all the data. "
    "When you have the final answer, you MUST start your response with exactly "
    "the words 'Final Answer:' followed by the answer in Chinese. "
    "Use bullet points for clarity."
)


def build_agent():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未在 .env 中找到 DEEPSEEK_API_KEY，请先配置后再运行。")

    df = pd.read_csv(DATA_PATH)
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
    )

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=6,
        prefix=PREFIX,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
        },
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 ab_test_data.csv，请先运行 generate_ab_data.py。")

    agent = build_agent()
    prompt = (
        "你是一名电商数据分析师。请基于已加载的 DataFrame `df`（包含全部 500 行数据）进行计算。"
        "ROI = GMV(元) / 消耗金额(元)。"
        f"问题：{QUESTION}"
    )

    print("开始分析 A/B 测试数据...\n")
    result = agent.invoke({"input": prompt})

    print("\n" + "=" * 60)
    print("最终答案:")
    print("=" * 60 + "\n")
    print(result["output"])


if __name__ == "__main__":
    main()
