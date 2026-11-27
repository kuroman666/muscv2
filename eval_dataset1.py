import os
import pandas as pd

# ==========================================
# 1. 設定檔案路徑
# ==========================================
GT_CSV = "dataset1_gt.csv"                  # 欄位需包含: filename, gt_label (1=瑕疵, 0=正常)，可選: batch
PRED_CSV = "escape_ranking_report.csv"      # 1_infer_muscv2_no_gt.py 產出的報表
OUTPUT_DETAIL_CSV = "dataset1_eval_detail.csv"  # 輸出每張圖的對答案明細（方便查哪幾張判錯）


# ==========================================
# 2. 工具函式
# ==========================================
def clean_stem(filename: str) -> str:
    """只保留最乾淨的檔名（去掉路徑與副檔名），確保兩份 CSV 100% 對得上"""
    base = os.path.basename(str(filename).strip())
    return os.path.splitext(base)[0]


def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """計算混淆矩陣與各項分類指標"""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def print_report(title: str, th: float, m: dict):
    print(f"\n📌 【{title}】 Threshold >= {th:.4f}")
    print(f"  - Accuracy  : {m['Accuracy']:.2%}")
    print(f"  - Precision : {m['Precision']:.2%}  |  Recall : {m['Recall']:.2%}  |  F1 : {m['F1']:.4f}")
    print(f"  - 混淆矩陣   : TP={m['TP']}, FN={m['FN']} (漏抓) | TN={m['TN']}, FP={m['FP']} (誤報)")


# ==========================================
# 3. 主程式：讀取、合併、掃描最佳 Threshold
# ==========================================
def main():
    # 1. 讀取兩份 CSV
    gt_df = pd.read_csv(GT_CSV)
    pred_df = pd.read_csv(PRED_CSV)

    # 2. 產生統一的比對鑰匙 (clean_key)
    gt_df["clean_key"] = gt_df["filename"].apply(clean_stem)
    pred_df["clean_key"] = pred_df["Original_Filename"].apply(clean_stem)

    # 確保型別正確
    gt_df["gt_label"] = pd.to_numeric(gt_df["gt_label"], errors="coerce").astype(int)
    pred_df["Max_Score"] = pd.to_numeric(pred_df["Max_Score"], errors="coerce")

    # 3. 合併對答案（若 pred_df 有重複檔名，取分數最高的那筆）
    pred_clean = pred_df.groupby("clean_key", as_index=False)["Max_Score"].max()
    merged = pd.merge(gt_df, pred_clean, on="clean_key", how="left")

    # 防呆檢查：有沒有 GT 裡的圖片在推論報表裡找不到？
    missing = merged[merged["Max_Score"].isna()]
    if len(missing) > 0:
        print(f"⚠️  [警告] 有 {len(missing)} 張 GT 圖片在 {PRED_CSV} 裡找不到對應分數：")
        for f in missing["filename"].head(10):
            print(f"    -> {f}")
        merged = merged.dropna(subset=["Max_Score"]).copy()

    n_total = len(merged)
    n_pos = int((merged["gt_label"] == 1).sum())
    n_neg = int((merged["gt_label"] == 0).sum())

    print("=" * 65)
    print(f"✅ 成功比對總張數: {n_total} 張（瑕疵: {n_pos} 張 / 正常: {n_neg} 張）")
    print("=" * 65)

    # 4. 掃描所有候選 Threshold（每個分數都試一次，加上全部判正常的高門檻）
    scores = sorted(merged["Max_Score"].unique())
    candidate_ths = scores + [scores[-1] + 1e-4]

    best_acc_th, best_acc_metrics = None, {"Accuracy": -1}
    best_f1_th, best_f1_metrics = None, {"F1": -1}

    for th in candidate_ths:
        y_pred = (merged["Max_Score"] >= th).astype(int)
        m = calc_metrics(merged["gt_label"], y_pred)

        if m["Accuracy"] > best_acc_metrics["Accuracy"]:
            best_acc_th, best_acc_metrics = th, m

        if m["F1"] > best_f1_metrics["F1"]:
            best_f1_th, best_f1_metrics = th, m

    # 5. 印出最佳結果
    print_report("最高 Accuracy 切點", best_acc_th, best_acc_metrics)
    print_report("最高 F1-Score 切點（不平衡比例時建議參考）", best_f1_th, best_f1_metrics)

    # 6. 若 dataset1_gt.csv 有 batch 欄位，順便印出各批在最佳 Accuracy 切點下的表現
    if "batch" in merged.columns:
        print("\n" + "-" * 65)
        print(f"📊 各批次細部表現（套用最佳 Accuracy 切點 >= {best_acc_th:.4f}）：")
        for batch, group in merged.groupby("batch", sort=True):
            y_pred_sub = (group["Max_Score"] >= best_acc_th).astype(int)
            m_sub = calc_metrics(group["gt_label"], y_pred_sub)
            print(f"  - {batch}: Acc={m_sub['Accuracy']:.2%} | Recall={m_sub['Recall']:.2%} "
                  f"(TP={m_sub['TP']}, FN={m_sub['FN']}, FP={m_sub['FP']}, TN={m_sub['TN']})")

    # 7. 輸出詳細對答案明細（套用最佳 Accuracy 切點），方便檢查誤報與漏抓
    merged["pred_label"] = (merged["Max_Score"] >= best_acc_th).astype(int)
    merged["is_correct"] = (merged["pred_label"] == merged["gt_label"])
    merged = merged.sort_values(by="Max_Score", ascending=False)
    merged.drop(columns=["clean_key"]).to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print(f"💾 詳細對答案明細已存至：{OUTPUT_DETAIL_CSV}")


if __name__ == "__main__":
    main()