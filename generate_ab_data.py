from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("ab_test_data.csv")
RANDOM_SEED = 42
ROW_COUNT = 500


def build_dataset(row_count: int = ROW_COUNT) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2026-04-01", periods=30, freq="D")

    rows = []
    for idx in range(row_count):
        group = "A" if idx < row_count // 2 else "B"
        date = rng.choice(dates)
        user_id = f"U{100000 + idx}"

        impressions = int(rng.integers(180, 1200))

        if group == "A":
            ctr = float(np.clip(rng.normal(0.066, 0.008), 0.035, 0.095))
            cvr = float(np.clip(rng.normal(0.105, 0.014), 0.06, 0.16))
            cpc = float(np.clip(rng.normal(1.62, 0.14), 1.1, 2.0))
            aov = float(np.clip(rng.normal(168, 18), 120, 230))
        else:
            ctr = float(np.clip(rng.normal(0.068, 0.008), 0.038, 0.1))
            cvr = float(np.clip(rng.normal(0.118, 0.014), 0.075, 0.175))
            cpc = float(np.clip(rng.normal(1.66, 0.14), 1.15, 2.05))
            aov = float(np.clip(rng.normal(172, 18), 125, 235))

        clicks = min(impressions, max(0, int(rng.binomial(impressions, ctr))))
        conversions = min(clicks, max(0, int(rng.binomial(clicks, cvr))))

        spend = round(clicks * cpc * rng.uniform(0.96, 1.05), 2)
        gmv = round(conversions * aov * rng.uniform(0.97, 1.06), 2)

        rows.append(
            {
                "日期": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "用户ID": user_id,
                "实验组别": group,
                "曝光量": impressions,
                "点击量": clicks,
                "转化量": conversions,
                "消耗金额(元)": spend,
                "GMV(元)": gmv,
            }
        )

    df = pd.DataFrame(rows).sort_values(["日期", "实验组别", "用户ID"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("实验组别")[["曝光量", "点击量", "转化量", "消耗金额(元)", "GMV(元)"]]
        .sum()
        .assign(
            点击率=lambda x: x["点击量"] / x["曝光量"],
            转化率=lambda x: x["转化量"] / x["点击量"],
            ROI=lambda x: x["GMV(元)"] / x["消耗金额(元)"],
        )
    )

    print(f"已生成 {OUTPUT_PATH.resolve()}")
    print(f"数据行数: {len(df)}")
    print("\n分组汇总:")
    print(summary.round(4))


if __name__ == "__main__":
    main()
