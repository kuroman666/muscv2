import pandas as pd

INPUT_CSV = "master_gt.csv"
OUTPUT_SUMMARY_CSV = "summary_metrics.csv"


def calc_stats(df_sub: pd.DataFrame, batch_name: str) -> dict:
    """計算單一批次（或總計）的母體數、Overlap 與 Recall"""
    raw_total = len(df_sub)
    valid = df_sub[df_sub["engineer_confirmed"] == 1]
    rejected = raw_total - len(valid)

    total_gt = len(valid)
    f_only = len(valid[(valid["foundad_hit"] == 1) & (valid["muscv2_hit"] == 0)])
    m_only = len(valid[(valid["foundad_hit"] == 0) & (valid["muscv2_hit"] == 1)])
    overlap = len(valid[(valid["foundad_hit"] == 1) & (valid["muscv2_hit"] == 1)])

    f_detected = f_only + overlap
    m_detected = m_only + overlap

    f_recall = (f_detected / total_gt) if total_gt > 0 else 0.0
    m_recall = (m_detected / total_gt) if total_gt > 0 else 0.0

    return {
        "Batch": batch_name,
        "原候選總數": raw_total,
        "工程師退回(0)": rejected,
        "最新母體數(1)": total_gt,
        "FoundAD獨有": f_only,
        "MuSc-V2獨有": m_only,
        "Overlap重疊": overlap,
        "FoundAD抓到": f"{f_detected}/{total_gt}",
        "FoundAD_Recall": f"{f_recall:.2%}",
        "MuSc-V2抓到": f"{m_detected}/{total_gt}",
        "MuSc-V2_Recall": f"{m_recall:.2%}",
    }


def main():
    # 1. 讀取 master_gt.csv，並確保數值欄位型別正確（防呆：避免 Excel 存成字串或空值）
    df = pd.read_csv(INPUT_CSV)
    for col in ["foundad_hit", "muscv2_hit", "engineer_confirmed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    summary_rows = []

    # 2. 逐批重算 (W639, W642, W646...)
    for batch, group in df.groupby("batch", sort=True):
        summary_rows.append(calc_stats(group, batch))

    # 3. 加上跨批總計 (ALL)
    summary_rows.append(calc_stats(df, "ALL (總計)"))

    # 4. 印出結果並存成 CSV
    summary_df = pd.DataFrame(summary_rows)
    print("=" * 70)
    print("📊 最新瑕疵母體與 Recall 重算結果")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(f"\n【{row['Batch']}】")
        print(f"  母體更新: 原 {row['原候選總數']} 張 -> 退回 {row['工程師退回(0)']} 張 -> 最新母體 {row['最新母體數(1)']} 張")
        print(f"  組成拆解: FoundAD獨有 {row['FoundAD獨有']} | MuSc-V2獨有 {row['MuSc-V2獨有']} | Overlap {row['Overlap重疊']}")
        print(f"  FoundAD Recall : {row['FoundAD_Recall']} ({row['FoundAD抓到']})")
        print(f"  MuSc-V2 Recall : {row['MuSc-V2_Recall']} ({row['MuSc-V2抓到']})")

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"✅ 統計表已儲存至：{OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()