# 檔案位置: MuSc-V2/2_split_eval_results.py
# 用途: 將 MuSc-V2 推論結果依「每包張數 (--counts)」或「比例 (--ratios)」拆分為多份獨立 Eval 報表
import os
import re
import csv
import glob
import random
import shutil
import argparse
from typing import List, Dict, Tuple
import numpy as np


def find_latest_run_dir(base_dir: str) -> str:
    """若傳入的是父目錄，自動尋找底下最新的時間戳記結果資料夾。"""
    abs_base = os.path.abspath(base_dir)
    if os.path.isfile(os.path.join(abs_base, "escape_ranking_report.csv")):
        return abs_base

    csv_candidates = sorted(glob.glob(os.path.join(abs_base, "*", "escape_ranking_report.csv")))
    csv_candidates = [
        p for p in csv_candidates
        if not os.path.basename(os.path.dirname(p)).startswith("eval_pack_")
    ]
    if not csv_candidates:
        raise FileNotFoundError(
            f"在 {abs_base} 及其子目錄下找不到 escape_ranking_report.csv！請確認已完成推論。"
        )
    return os.path.dirname(csv_candidates[-1])


def parse_header_metadata(txt_path: str) -> Dict[str, str]:
    """從原始 escape_ranking_report.txt 讀取 Effective Threshold 與模型設定資訊。"""
    meta = {
        "score_mode": "FILTERED_ROBUST",
        "thresh_mode": "hybrid",
        "eff_thresh": "0.2500",
        "wafer_edge_info": "auto",
        "overfit_diag": "Zero-Shot Training-Free",
    }
    if not os.path.isfile(txt_path):
        return meta

    with open(txt_path, "r", encoding="utf-8") as f:
        head = f.read(2500)

    m_sm = re.search(r"Score Mode:\s*([A-Z_]+)", head)
    if m_sm:
        meta["score_mode"] = m_sm.group(1).strip()

    m_tm = re.search(r"Threshold Mode:\s*([a-zA-Z0-9_]+)", head)
    if m_tm:
        meta["thresh_mode"] = m_tm.group(1).strip()

    m_et = re.search(r"Effective Threshold:\s*([0-9.]+)", head)
    if m_et:
        meta["eff_thresh"] = m_et.group(1).strip()

    m_we = re.search(r"Wafer Edge Mode:\s*([^\n]+)", head)
    if m_we:
        meta["wafer_edge_info"] = m_we.group(1).strip()

    m_ov = re.search(r"Overfit Diagnostic\s*:\s*([^\n]+)", head)
    if m_ov:
        meta["overfit_diag"] = m_ov.group(1).strip()

    return meta


def estimate_blind_normal_core_stats(pool_scores: List[float]) -> Tuple[float, float, np.ndarray]:
    """真無 GT 盲池左半部正常核心估計器 (無需 Golden Support Set)。"""
    pool_arr = np.array(pool_scores, dtype=np.float32)
    if pool_arr.size == 0:
        return 0.20, 0.015, np.array([0.20], dtype=np.float32)

    sorted_pool = np.sort(pool_arr)
    n = sorted_pool.size
    half_n = max(1, int(np.ceil(n * 0.50)))
    lower_half = sorted_pool[:half_n]
    mu_core = float(np.median(lower_half))

    left_of_mu = sorted_pool[sorted_pool <= mu_core]
    left_diffs = mu_core - left_of_mu
    if left_diffs.size >= 2:
        sigma_left_rms = float(np.sqrt(np.mean(np.square(left_diffs))))
        sigma_left_mad = float(1.4826 * np.median(np.abs(left_of_mu - np.median(left_of_mu))))
        sigma_left = max(sigma_left_rms, sigma_left_mad)
    else:
        sigma_left = float(np.std(lower_half)) if lower_half.size > 1 else 0.015

    if lower_half.size >= 4:
        lower_iqr_std = float((np.percentile(lower_half, 75.0) - np.percentile(lower_half, 25.0)) / 1.349)
        sigma_left = max(sigma_left, lower_iqr_std * 0.80)

    sigma_left = float(np.clip(sigma_left, 0.012, 0.14))
    est_normal_mask = sorted_pool <= (mu_core + 2.5 * sigma_left)
    est_normal_arr = sorted_pool[est_normal_mask] if np.any(est_normal_mask) else lower_half
    return mu_core, sigma_left, est_normal_arr


def compute_dynamic_effective_threshold(
    pool_scores: List[float],
    thresh_mode: str,
    user_threshold: float,
    min_threshold: float,
    auto_percentile: float,
    sigma_k: float,
    top_k: int,
) -> Tuple[float, str]:
    mode = thresh_mode.lower().strip()
    scores_arr = np.array(pool_scores, dtype=np.float32)
    if scores_arr.size == 0:
        return max(user_threshold, min_threshold), "無樣本保底門檻"

    mu_core, sigma_left, est_normal_arr = estimate_blind_normal_core_stats(pool_scores)
    eff_sigma_k = min(float(sigma_k), 3.5) if scores_arr.size < 60 else float(sigma_k)
    left_stat_bound = float(mu_core + eff_sigma_k * sigma_left)

    p_clip = float(np.clip(auto_percentile, 50.0, 99.9))
    p_val_core = float(np.percentile(est_normal_arr, p_clip))
    p_val_raw = float(np.percentile(scores_arr, p_clip))

    n_total = scores_arr.size
    max_allowed_anoms = int(np.floor(n_total * ((100.0 - p_clip) / 100.0) + 1e-6))
    sorted_desc_all = np.sort(scores_arr)[::-1]
    if max_allowed_anoms <= 0:
        strict_pct_bound = float(sorted_desc_all[0] + 1e-5)
    elif max_allowed_anoms < n_total:
        hi_v = float(sorted_desc_all[max_allowed_anoms - 1])
        lo_v = float(sorted_desc_all[max_allowed_anoms])
        strict_pct_bound = float(0.5 * (hi_v + lo_v)) if hi_v > lo_v else float(hi_v + 1e-5)
    else:
        strict_pct_bound = p_val_raw

    raw_pct_floor = max(p_val_raw, strict_pct_bound)
    topk_bound = 0.0
    if top_k > 0 and scores_arr.size > top_k:
        topk_bound = float(0.5 * (sorted_desc_all[top_k - 1] + sorted_desc_all[top_k]))

    floor_val = max(float(min_threshold), float(user_threshold))

    if mode == "fixed":
        eff = float(user_threshold)
        desc = f"Fixed 固定門檻 = {eff:.4f}"
    elif mode == "mean_std":
        eff = max(float(min_threshold), left_stat_bound, topk_bound) if top_k > 0 else max(float(min_threshold), left_stat_bound)
        desc = f"Left-Core 統計門檻 (mu_core+{eff_sigma_k:g}*σ_left) = {eff:.4f} [mu_core={mu_core:.4f}, σ_left={sigma_left:.4f}]"
    elif mode == "percentile":
        eff = max(float(min_threshold), raw_pct_floor, topk_bound)
        desc = f"Auto-P{auto_percentile:g} 嚴格上限分位數門檻 = {eff:.4f} [Raw_P{auto_percentile:g}={p_val_raw:.4f}]"
    elif mode == "top_k":
        eff = max(float(min_threshold), topk_bound if top_k > 0 else left_stat_bound)
        desc = f"Auto-TopK (K={top_k}) 動態門檻 = {eff:.4f}"
    else:
        hybrid_stat = max(floor_val, left_stat_bound, raw_pct_floor)
        eff = max(hybrid_stat, topk_bound) if top_k > 0 else hybrid_stat
        desc = (
            f"Hybrid Zero-Shot 動態門檻 (上限<={100.0 - p_clip:g}%) = {eff:.4f} "
            f"[正常核心上界(mu+{eff_sigma_k:g}*σ_left)={left_stat_bound:.4f}, "
            f"Raw_P{auto_percentile:g}保底={raw_pct_floor:.4f} (Core_P{auto_percentile:g}={p_val_core:.4f}), "
            f"MinFloor={floor_val:.4f}]"
        )
    return eff, desc


def compute_pack_sizes(total_items: int, num_packs: int, counts_str: str, ratios_str: str) -> List[int]:
    if num_packs <= 0:
        raise ValueError(f"[設定錯誤] 拆分包數 (--num_packs={num_packs}) 必須大於 0！")
    if counts_str and ratios_str:
        raise ValueError("[設定錯誤] 請勿同時指定 --counts 與 --ratios，請擇一使用！")

    if counts_str:
        parts = [int(x.strip()) for x in re.split(r"[,:/\s]+", counts_str.strip()) if x.strip()]
        if not parts or any(p <= 0 for p in parts):
            raise ValueError(f"[設定錯誤] --counts ('{counts_str}') 內的每包張數皆必須為大於 0 的正整數！")
        if len(parts) == 1:
            sizes = [parts[0]] * num_packs
        else:
            if len(parts) != num_packs:
                raise ValueError(f"❌ [拆分包數與張數設定不一致] --num_packs={num_packs} 但 --counts={parts}")
            sizes = list(parts)
        if sum(sizes) != total_items:
            raise ValueError(f"❌ [拆分圖片總數量不符] 各包設定加總 {sum(sizes)} != 可拆分總數 {total_items}")
        return sizes

    if not ratios_str:
        ratios_str = ":".join(["1"] * num_packs)

    raw_ratios = [float(x.strip()) for x in re.split(r"[,:/\s]+", ratios_str.strip()) if x.strip()]
    if not raw_ratios or any(r <= 0 for r in raw_ratios):
        raise ValueError(f"[設定錯誤] 無效的比例設定: {ratios_str}")
    if len(raw_ratios) == 1 and num_packs > 1:
        raw_ratios = raw_ratios * num_packs
    elif len(raw_ratios) != num_packs:
        raise ValueError(f"❌ [拆分包數與比例數量不一致] --num_packs={num_packs} 但 --ratios={raw_ratios}")

    total_weight = sum(raw_ratios)
    exact_counts = [total_items * (r / total_weight) for r in raw_ratios]
    base_counts = [int(np.floor(c)) for c in exact_counts]
    remainder = total_items - sum(base_counts)

    frac_with_idx = sorted(
        enumerate([exact_counts[i] - base_counts[i] for i in range(len(raw_ratios))]),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remainder):
        base_counts[frac_with_idx[i][0]] += 1
    return base_counts


def split_records(records: List[Dict], pack_sizes: List[int], split_order: str, seed: int) -> List[List[Dict]]:
    n_packs = len(pack_sizes)
    work_records = list(records)

    if split_order == "rank":
        work_records.sort(key=lambda r: float(r["Max_Score"]), reverse=True)
    elif split_order == "filename":
        work_records.sort(key=lambda r: r["Original_Filename"])
    elif split_order == "random":
        rng = random.Random(seed)
        rng.shuffle(work_records)
    elif split_order == "stratified":
        rng = random.Random(seed)
        buckets: Dict[str, List[Dict]] = {}
        for r in work_records:
            key = "ESCAPE" if os.path.basename(r["Original_Filename"]).startswith("escape_") else r["Status"]
            buckets.setdefault(key, []).append(r)

        interleaved = []
        for k in sorted(buckets.keys()):
            rng.shuffle(buckets[k])
        max_len = max((len(v) for v in buckets.values()), default=0)
        for i in range(max_len):
            for k in sorted(buckets.keys()):
                if i < len(buckets[k]):
                    interleaved.append(buckets[k][i])
        work_records = interleaved

    packs: List[List[Dict]] = [[] for _ in range(n_packs)]
    if split_order == "stratified" and sum(pack_sizes) == len(work_records):
        remaining_cap = list(pack_sizes)
        idx = 0
        for item in work_records:
            while remaining_cap[idx % n_packs] <= 0:
                idx += 1
            target_pack = idx % n_packs
            packs[target_pack].append(item)
            remaining_cap[target_pack] -= 1
            idx += 1
    else:
        cursor = 0
        for p_idx, sz in enumerate(pack_sizes):
            packs[p_idx] = work_records[cursor : cursor + sz]
            cursor += sz

    for p in packs:
        p.sort(key=lambda r: float(r["Max_Score"]), reverse=True)
    return packs


def link_or_copy_file(src: str, dst: str, use_symlink: bool = False):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    if use_symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def generate_pack_eval_report(
    pack_idx: int,
    pack_records: List[Dict],
    src_run_dir: str,
    pack_out_dir: str,
    header_meta: Dict[str, str],
    recompute_pack_thresh: bool = False,
    thresh_mode_override: str = "",
    sigma_k: float = 3.5,
    auto_percentile: float = 95.0,
    min_threshold: float = 0.0,
    user_threshold: float = 0.0,
    top_k: int = 0,
    use_symlink: bool = False,
) -> Dict:
    os.makedirs(pack_out_dir, exist_ok=True)
    anom_vis_dir = os.path.join(pack_out_dir, "anomaly_images")
    anom_full_dir = os.path.join(anom_vis_dir, "full_res_bbox_only")
    amap_dir = os.path.join(pack_out_dir, "amap_npy")
    os.makedirs(anom_vis_dir, exist_ok=True)
    os.makedirs(anom_full_dir, exist_ok=True)
    os.makedirs(amap_dir, exist_ok=True)

    src_anom_vis_dir = os.path.join(src_run_dir, "anomaly_images")
    src_anom_full_dir = os.path.join(src_anom_vis_dir, "full_res_bbox_only")
    src_amap_dir = os.path.join(src_run_dir, "amap_npy")

    global_eff_thresh = float(header_meta.get("eff_thresh", 0.25))
    active_thresh_mode = thresh_mode_override if thresh_mode_override else header_meta.get("thresh_mode", "hybrid")

    pack_scores = [float(r["Max_Score"]) for r in pack_records]
    local_eff_thresh, local_thresh_desc = compute_dynamic_effective_threshold(
        pool_scores=pack_scores,
        thresh_mode=active_thresh_mode,
        user_threshold=user_threshold,
        min_threshold=min_threshold,
        auto_percentile=auto_percentile,
        sigma_k=sigma_k,
        top_k=top_k,
    )

    n_ng = len(pack_scores)
    if 0 < n_ng < 25:
        alpha = float(np.clip(n_ng / 25.0, 0.35, 1.0))
        blended_thresh = float(alpha * local_eff_thresh + (1.0 - alpha) * global_eff_thresh)
        if active_thresh_mode.lower().strip() in ("hybrid", "percentile"):
            p_clip = float(np.clip(auto_percentile, 50.0, 99.9))
            pack_raw_p = float(np.percentile(np.array(pack_scores, dtype=np.float32), p_clip))
            local_eff_thresh = max(blended_thresh, pack_raw_p)
        else:
            local_eff_thresh = blended_thresh
        local_thresh_desc += f" [小樣本平滑融合 α={alpha:.2f} with Global={global_eff_thresh:.4f}]"

    eff_thresh = local_eff_thresh if recompute_pack_thresh else global_eff_thresh
    thresh_source_tag = "PACK_LOCAL_RECOMPUTED" if recompute_pack_thresh else "GLOBAL_POOL_UNIFIED"

    anomaly_count = 0
    skipped_count = 0

    updated_records = []
    for local_rank, r in enumerate(pack_records, 1):
        row = dict(r)
        row["Global_Rank"] = r["Rank"]
        row["Pack_Rank"] = local_rank
        orig_status = str(r.get("Status", "PASS_SKIPPED"))
        row["Global_Status"] = orig_status
        score_val = float(r["Max_Score"])
        num_bboxes = int(r.get("Num_BBoxes", 0))
        saved_fname = r.get("Saved_Anomaly_Filename", "")

        if recompute_pack_thresh:
            if score_val >= eff_thresh:
                if orig_status == "PASS_NOISE_FILTERED":
                    new_status = "PASS_NOISE_FILTERED"
                elif num_bboxes > 0:
                    new_status = "ANOMALY"
                else:
                    new_status = "ANOMALY_LOCAL_TH"
            else:
                new_status = "PASS_SKIPPED"
                saved_fname = "(未超過本包獨立動態門檻，不輸出圖片)"
            row["Status"] = new_status
            row["Saved_Anomaly_Filename"] = saved_fname

        if "ANOMALY" in row["Status"] and saved_fname and not saved_fname.startswith("("):
            anomaly_count += 1
            link_or_copy_file(
                os.path.join(src_anom_vis_dir, saved_fname),
                os.path.join(anom_vis_dir, saved_fname),
                use_symlink=use_symlink,
            )
            link_or_copy_file(
                os.path.join(src_anom_full_dir, saved_fname),
                os.path.join(anom_full_dir, saved_fname),
                use_symlink=use_symlink,
            )
            orig_stem = re.sub(r"[\\/]+", "__", os.path.splitext(r["Original_Filename"])[0])
            for npy_src in glob.glob(os.path.join(src_amap_dir, f"*_{orig_stem}.npy")):
                link_or_copy_file(
                    npy_src,
                    os.path.join(amap_dir, os.path.basename(npy_src)),
                    use_symlink=use_symlink,
                )
        else:
            skipped_count += 1

        updated_records.append(row)

    total = len(updated_records)
    normal_recs = [
        r for r in updated_records if not os.path.basename(r["Original_Filename"]).startswith("escape_")
    ]
    escape_defects = [
        r for r in updated_records if os.path.basename(r["Original_Filename"]).startswith("escape_")
    ]
    target_normal = normal_recs if normal_recs else updated_records

    if target_normal:
        norm_scores = [float(r["Max_Score"]) for r in target_normal]
        raw_scores = [float(r.get("Raw_Max_Score", r["Max_Score"])) for r in target_normal]
        filt_scores = [float(r.get("Filtered_Max_Score", r["Max_Score"])) for r in target_normal]
        max_s = float(np.max(norm_scores))
        p99_s = float(np.percentile(norm_scores, 99))
        p95_s = float(np.percentile(norm_scores, 95))
        median_s = float(np.median(norm_scores))
        mean_s = float(np.mean(norm_scores))
        std_s = float(np.std(norm_scores))
        raw_max_s, raw_mean_s = float(np.max(raw_scores)), float(np.mean(raw_scores))
        filt_max_s, filt_mean_s = float(np.max(filt_scores)), float(np.mean(filt_scores))
    else:
        max_s = p99_s = p95_s = median_s = mean_s = std_s = 0.0
        raw_max_s = raw_mean_s = filt_max_s = filt_mean_s = 0.0

    m1std = mean_s + 1.0 * std_s
    m2std = mean_s + 2.0 * std_s
    m3std = mean_s + 3.0 * std_s

    fp_normal = [r for r in target_normal if "ANOMALY" in r["Status"]]
    caught_escape = [r for r in escape_defects if "ANOMALY" in r["Status"]]
    fpr_normal = (len(fp_normal) / len(target_normal) * 100.0) if target_normal else 0.0
    escape_recall = (len(caught_escape) / len(escape_defects) * 100.0) if escape_defects else 0.0

    report_txt = os.path.join(pack_out_dir, "escape_ranking_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"=== MuSc-V2 Split Eval Report — Pack #{pack_idx} ({os.path.basename(pack_out_dir)}) ===\n")
        f.write(
            f"Source Run Dir: {src_run_dir}\n"
            f"Score Mode: {header_meta['score_mode']} | Threshold Mode: {active_thresh_mode} | "
            f"Threshold Policy: {thresh_source_tag}\n"
            f"Effective Threshold (Used) : {eff_thresh:.4f} "
            f"(Pack_Local_Thresh={local_eff_thresh:.4f} vs Global_Pool_Thresh={global_eff_thresh:.4f}, "
            f"Diff={local_eff_thresh - global_eff_thresh:+.4f})\n"
            f"Pack Local Thresh Detail : {local_thresh_desc}\n"
            f"Pack Total Images: {total} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_count}\n"
            f"Normal False Positives (FPR): {len(fp_normal)}/{len(target_normal)} ({fpr_normal:.2f}%) | "
            f"Escape Caught (Recall): {len(caught_escape)}/{len(escape_defects)} ({escape_recall:.2f}%)\n"
            f"Normal Score Dist: Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | "
            f"Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
            f"Ablation Compare (Normal) : Raw_Max={raw_max_s:.4f}, Raw_Mean={raw_mean_s:.4f} vs "
            f"Filtered_Max={filt_max_s:.4f}, Filtered_Mean={filt_mean_s:.4f}\n"
            f"Threshold Ref (Mean+k*Std) : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
        )
        f.write("-" * 165 + "\n")
        f.write(
            f"{'P_Rank':<6} | {'G_Rank':<6} | {'Original_Filename':<26} | {'Status':<19} | "
            f"{'Global_Status':<19} | {'Max_Score':<9} | {'Raw_Score':<9} | {'Filt_Score':<10} | "
            f"{'Delta':<8} | {'BBoxes':<6} | {'Saved_Anomaly_Filename'}\n"
        )
        f.write("-" * 165 + "\n")
        for r in updated_records:
            raw_v = float(r.get("Raw_Max_Score", r["Max_Score"]))
            flt_v = float(r.get("Filtered_Max_Score", r["Max_Score"]))
            dlt_v = float(r.get("Filter_Delta", flt_v - raw_v))
            f.write(
                f"{r['Pack_Rank']:<6} | {r['Global_Rank']:<6} | {r['Original_Filename']:<26} | "
                f"{r['Status']:<19} | {r['Global_Status']:<19} | {float(r['Max_Score']):<9.4f} | "
                f"{raw_v:<9.4f} | {flt_v:<10.4f} | {dlt_v:<+8.4f} | {r['Num_BBoxes']:<6} | "
                f"{r['Saved_Anomaly_Filename']}\n"
            )

    report_csv = os.path.join(pack_out_dir, "escape_ranking_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pack_Rank", "Global_Rank", "Original_Filename", "Status", "Global_Status",
            "Effective_Threshold_Used", "Pack_Local_Threshold", "Global_Pool_Threshold",
            "Max_Score", "Raw_Max_Score", "Filtered_Max_Score", "Filter_Delta",
            "TopK_Mean_Score", "Num_BBoxes", "Saved_Anomaly_Filename", "Resolution"
        ])
        for r in updated_records:
            writer.writerow([
                r["Pack_Rank"],
                r["Global_Rank"],
                r["Original_Filename"],
                r["Status"],
                r["Global_Status"],
                f"{eff_thresh:.6f}",
                f"{local_eff_thresh:.6f}",
                f"{global_eff_thresh:.6f}",
                r["Max_Score"],
                r.get("Raw_Max_Score", r["Max_Score"]),
                r.get("Filtered_Max_Score", r["Max_Score"]),
                r.get("Filter_Delta", "0.000000"),
                r.get("TopK_Mean_Score", r["Max_Score"]),
                r["Num_BBoxes"],
                r["Saved_Anomaly_Filename"],
                r.get("Resolution", ""),
            ])

    return {
        "pack_idx": pack_idx,
        "pack_name": os.path.basename(pack_out_dir),
        "pack_dir": pack_out_dir,
        "report_csv": report_csv,
        "report_txt": report_txt,
        "total": total,
        "num_normal": len(target_normal),
        "num_escape": len(escape_defects),
        "caught_escape": len(caught_escape),
        "escape_recall": escape_recall,
        "fp_normal": len(fp_normal),
        "fpr_normal": fpr_normal,
        "anomaly_exported": anomaly_count,
        "eff_thresh_used": eff_thresh,
        "local_eff_thresh": local_eff_thresh,
        "global_eff_thresh": global_eff_thresh,
        "max_s": max_s,
        "p99_s": p99_s,
        "p95_s": p95_s,
        "median_s": median_s,
        "mean_s": mean_s,
        "std_s": std_s,
        "m1std": m1std,
        "m2std": m2std,
        "m3std": m3std,
    }


def main():
    parser = argparse.ArgumentParser(description="MuSc-V2 推論結果拆分與多包獨立 Eval 評估器")
    parser.add_argument("--src_dir", type=str, default="./output/muscv2_no_gt")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_packs", type=int, default=3)
    parser.add_argument("--counts", type=str, default="")
    parser.add_argument("--ratios", type=str, default="")
    parser.add_argument("--split_order", type=str, default="stratified", choices=["stratified", "random", "rank", "filename"])
    parser.add_argument("--recompute_pack_thresh", action="store_true")
    parser.add_argument("--thresh_mode", type=str, default="", choices=["", "hybrid", "mean_std", "percentile", "top_k", "fixed"])
    parser.add_argument("--sigma_k", type=float, default=3.5)
    parser.add_argument("--auto_percentile", type=float, default=95.0)
    parser.add_argument("--min_threshold", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--seed", "--random_seed", dest="seed", type=int, default=42)
    args = parser.parse_args()

    src_run_dir = find_latest_run_dir(args.src_dir)
    src_csv = os.path.join(src_run_dir, "escape_ranking_report.csv")
    src_txt = os.path.join(src_run_dir, "escape_ranking_report.txt")

    header_meta = parse_header_metadata(src_txt)
    with open(src_csv, "r", encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))

    if not all_records:
        raise ValueError(f"報表 {src_csv} 內沒有任何紀錄！")

    total_avail = len(all_records)
    pack_sizes = compute_pack_sizes(
        total_items=total_avail,
        num_packs=args.num_packs,
        counts_str=args.counts,
        ratios_str=args.ratios,
    )
    packs = split_records(
        records=all_records,
        pack_sizes=pack_sizes,
        split_order=args.split_order,
        seed=args.seed,
    )

    out_root = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(src_run_dir, f"split_eval_{len(packs)}packs")
    )
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    summaries = []
    for i, pack_recs in enumerate(packs, 1):
        pack_dir = os.path.join(out_root, f"eval_pack_{i}")
        s = generate_pack_eval_report(
            pack_idx=i,
            pack_records=pack_recs,
            src_run_dir=src_run_dir,
            pack_out_dir=pack_dir,
            header_meta=header_meta,
            recompute_pack_thresh=args.recompute_pack_thresh,
            thresh_mode_override=args.thresh_mode,
            sigma_k=args.sigma_k,
            auto_percentile=args.auto_percentile,
            min_threshold=args.min_threshold,
            user_threshold=args.threshold,
            top_k=args.top_k,
            use_symlink=args.symlink,
        )
        summaries.append(s)

    overview_txt = os.path.join(out_root, "all_3_evals_comparison.txt")
    overview_csv = os.path.join(out_root, "all_3_evals_comparison.csv")
    dynamic_txt = os.path.join(out_root, f"all_{len(summaries)}_evals_comparison.txt")
    dynamic_csv = os.path.join(out_root, f"all_{len(summaries)}_evals_comparison.csv")

    lines = [
        "=" * 115,
        f" 🎯 [MuSc-V2 最終 {len(summaries)} 份 Eval 結果並列總覽 (All {len(summaries)} Split Evaluations Comparison)]",
        "-" * 115,
        f" 📂 來源完整推論目錄 : {src_run_dir}",
        f" 📂 拆分結果根目錄 : {out_root}",
        f" 🖼️ 拆分數量核對 : 各包張數 {pack_sizes} (加總={sum(pack_sizes)} 張 / 資料集總數={total_avail} 張)",
        f" 🎚️ 門檻判定策略 : {'各包獨立重算動態門檻 (PACK_LOCAL_RECOMPUTED)' if args.recompute_pack_thresh else '全池統一動態門檻 (GLOBAL_POOL_UNIFIED)'} "
        f"| 全池基準門檻={header_meta['eff_thresh']} (Mode: {header_meta['thresh_mode']}, Score: {header_meta['score_mode']})",
        "-" * 115,
        f"{'Eval包名稱':<12} | {'總張數':<6} | {'正常圖數':<8} | {'實際採用門檻':<12} | {'本包獨立門檻':<12} | {'正常誤報(FPR)':<16} | {'抓出Escape瑕疵':<15} | {'Normal Mean±Std':<16} | {'mean+3*std'}",
        "-" * 115,
    ]
    for s in summaries:
        fpr_str = f"{s['fp_normal']}/{s['num_normal']} ({s['fpr_normal']:.1f}%)"
        esc_str = (
            f"{s['caught_escape']}/{s['num_escape']} ({s['escape_recall']:.1f}%)"
            if s["num_escape"] > 0
            else "0/0 (N/A)"
        )
        ms_str = f"{s['mean_s']:.4f}±{s['std_s']:.4f}"
        lines.append(
            f"{s['pack_name']:<12} | {s['total']:<6} | {s['num_normal']:<8} | "
            f"{s['eff_thresh_used']:<12.4f} | {s['local_eff_thresh']:<12.4f} | "
            f"{fpr_str:<16} | {esc_str:<15} | {ms_str:<16} | {s['m3std']:.4f}"
        )
    lines.append("=" * 115)

    full_summary_text = "\n".join(lines)
    print(full_summary_text)

    for txt_p in {overview_txt, dynamic_txt}:
        with open(txt_p, "w", encoding="utf-8") as f:
            f.write(full_summary_text + "\n")

    for csv_p in {overview_csv, dynamic_csv}:
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Eval_Pack", "Total_Images", "Normal_Images",
                "Effective_Threshold_Used", "Pack_Local_Threshold", "Global_Pool_Threshold",
                "Normal_FP_Count", "Normal_FPR_Percent", "Escape_Total", "Escape_Caught",
                "Escape_Recall_Percent", "Normal_Max", "Normal_P99", "Normal_P95",
                "Normal_Median", "Normal_Mean", "Normal_Std", "Mean_Plus_1Std",
                "Mean_Plus_2Std", "Mean_Plus_3Std", "Pack_Dir"
            ])
            for s in summaries:
                writer.writerow([
                    s["pack_name"], s["total"], s["num_normal"],
                    f"{s['eff_thresh_used']:.6f}", f"{s['local_eff_thresh']:.6f}", f"{s['global_eff_thresh']:.6f}",
                    s["fp_normal"], f"{s['fpr_normal']:.2f}", s["num_escape"], s["caught_escape"],
                    f"{s['escape_recall']:.2f}", f"{s['max_s']:.6f}", f"{s['p99_s']:.6f}", f"{s['p95_s']:.6f}",
                    f"{s['median_s']:.6f}", f"{s['mean_s']:.6f}", f"{s['std_s']:.6f}", f"{s['m1std']:.6f}",
                    f"{s['m2std']:.6f}", f"{s['m3std']:.6f}", s["pack_dir"]
                ])


if __name__ == "__main__":
    main()