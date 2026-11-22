# 檔案位置: MuSc-V2/1_infer_muscv2_no_gt.py
# 用途: MuSc-V2 Training-Free & Zero-Shot 無損加速版
# (GPU端: 高Batch/TF32/AMP/多執行緒預讀 + RAM充足時單次Forward收齊全部層；
#  I/O端: 原生解析度float32 .npy + 無壓縮PNG + 圓心座標快取；
#  後處理端: 1024x1024降維畫布 + ProcessPoolExecutor多核心平行縫合濾波 + 僅異常圖映回1:1原圖畫框)
import os
import re
import gc
import csv
import sys
import glob
import json
import time
import stat
import shutil
import hashlib
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 避免超大解析度晶圓圖觸發 PIL DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

TZ_TAIPEI = timezone(timedelta(hours=8))
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IGNORE_POOL_DIRS = {"masks", "masks2", "mask", "ground_truth", "gt", "with_gt", "__macosx"}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def get_available_ram_bytes() -> int:
    """讀取 Linux 系統目前可用主記憶體 (MemAvailable) 位元組數。"""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    return 32 * (1024 ** 3)


def format_duration(seconds: float) -> str:
    """將秒數格式化為 HH:MM:SS.xx 與總秒數。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f} ({seconds:.2f} 秒)"


def log_and_print(msg: str, log_files: List[str]):
    """同時在 Terminal 顯示並附加寫入 log.txt 檔案。"""
    print(msg, flush=True)
    for lf in log_files:
        if lf:
            with open(lf, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def safe_rmtree(target_dir: str):
    """安全刪除可能含有唯讀檔案的資料夾。"""
    if not target_dir or not os.path.exists(target_dir):
        return

    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IRWXU)
            func(path)
        except Exception:
            pass

    shutil.rmtree(target_dir, onerror=_onerror)


def safe_copy(src: str, dst: str):
    """複製檔案內容但不複製來源的唯讀鎖定屬性，並確保目標檔案可讀寫 (0644)。"""
    if os.path.exists(dst):
        try:
            os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    shutil.copyfile(src, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass


def scan_pool_images(pool_dir: str) -> List[str]:
    """遞迴搜尋資料夾與所有子資料夾內的圖片（自動略過 GT 遮罩資料夾），回傳排序後的完整路徑清單。"""
    if not os.path.isdir(pool_dir):
        return []
    img_paths = []
    for root, dirs, files in os.walk(pool_dir):
        dirs[:] = [d for d in sorted(dirs) if d.lower() not in IGNORE_POOL_DIRS]
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTENSIONS) and not f.startswith("."):
                img_paths.append(os.path.join(root, f))
    return sorted(img_paths)


def compute_pool_signature(pool_dir: str) -> str:
    """遞迴計算測試盲池所有有效影像之 (相對路徑, 檔案大小) MD5 指紋。"""
    if not os.path.isdir(pool_dir):
        return ""
    img_paths = scan_pool_images(pool_dir)
    h = hashlib.md5()
    h.update(os.path.abspath(pool_dir).encode("utf-8"))
    for fpath in img_paths:
        rel_path = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        try:
            sz = os.path.getsize(fpath)
        except OSError:
            sz = 0
        h.update(f"{rel_path}:{sz};".encode("utf-8"))
    return h.hexdigest()


def _robust_normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """使用 P0.1 ~ P99.9 穩健百分位數將 16-bit / float 陣列轉為 uint8，防止單點飽和亮點壓扁對比度。"""
    arr_f = arr.astype(np.float32)
    p_low, p_high = np.percentile(arr_f, [0.1, 99.9])
    if p_high <= p_low:
        p_low, p_high = float(np.min(arr_f)), float(np.max(arr_f))
    if p_high > p_low:
        arr_f = (arr_f - p_low) / (p_high - p_low) * 255.0
    else:
        arr_f = np.zeros_like(arr_f, dtype=np.float32)
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def load_image_rgb_uint8(img_path: str) -> np.ndarray:
    """安全讀取影像並統一轉為 (H, W, 3) uint8 RGB 格式 (支援 16-bit TIFF / 灰階 / 8-bit)。"""
    try:
        with Image.open(img_path) as pil_img:
            if pil_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                arr_u8 = _robust_normalize_to_uint8(np.array(pil_img))
                return np.stack([arr_u8, arr_u8, arr_u8], axis=-1)
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)
    except Exception:
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"無法讀取影像檔案: {img_path}")
        if raw.dtype != np.uint8:
            raw = _robust_normalize_to_uint8(raw)
        if raw.ndim == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        if raw.shape[2] == 4:
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def load_image_bgr_uint8(img_path: str) -> np.ndarray:
    """安全讀取影像並統一轉為 (H, W, 3) uint8 BGR 格式供 OpenCV 畫圖與疊圖使用。"""
    rgb = load_image_rgb_uint8(img_path)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _print_inline_progress(
    stage_title: str,
    current: int,
    total: int,
    start_t: float,
    extra_info: str = "",
    bar_len: int = 24,
) -> None:
    """在終端機單行即時更新視覺化進度條 (含百分比、處理速度與預估剩餘時間 ETA)。"""
    total = max(1, total)
    elapsed = max(1e-4, time.perf_counter() - start_t)
    rate = current / elapsed
    eta_sec = max(0.0, (total - current) / max(rate, 1e-6))
    pct = min(100.0, (current / total) * 100.0)
    filled = int(round(bar_len * current / total))
    bar = "█" * filled + "░" * (bar_len - filled)
    eta_str = format_duration(eta_sec) if current < total else "0.00秒"
    el_str = format_duration(elapsed)
    extra_str = f" | {extra_info}" if extra_info else ""
    msg = (
        f"\r⏳ [{stage_title}] |{bar}| {current}/{total} ({pct:5.1f}%) "
        f"[{el_str}<{eta_str}, {rate:5.2f}張/s]{extra_str}   "
    )
    print(msg, end="" if current < total else "\n", flush=True)


def prepare_mock_unlabeled_pool_from_mvtec(
    mvtec_cat_dir: str, mock_pool_dir: str, log_files: List[str], defect_limit: int = 8
):
    """從 MVTec / MVTec-2 / Mura 建立模擬無 GT 盲池 (clean_<原檔名> 與 escape_<瑕疵類別>_<原檔名>)。"""
    safe_rmtree(mock_pool_dir)
    os.makedirs(mock_pool_dir, exist_ok=True)
    test_dir = os.path.join(mvtec_cat_dir, "test")
    if not os.path.isdir(os.path.join(test_dir, "good")) and os.path.isdir(test_dir):
        for d in sorted(os.listdir(test_dir)):
            cand = os.path.join(test_dir, d)
            if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "good")):
                test_dir = cand
                break

    total_count = 0
    good_dir = os.path.join(test_dir, "good")
    for f in sorted(os.listdir(good_dir)):
        if f.lower().endswith(IMG_EXTENSIONS):
            safe_copy(os.path.join(good_dir, f), os.path.join(mock_pool_dir, f"clean_{f}"))
            total_count += 1

    defect_count = 0
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good" or sub.lower() in IGNORE_POOL_DIRS:
            continue
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for f in sorted(os.listdir(sub_dir)):
            if f.lower().endswith(IMG_EXTENSIONS) and defect_count < defect_limit:
                safe_copy(os.path.join(sub_dir, f), os.path.join(mock_pool_dir, f"escape_{sub}_{f}"))
                defect_count += 1
                total_count += 1

    log_and_print(
        f"[模擬無GT盲池建立完成] 共 {total_count} 張圖 (含 {defect_count} 張隱藏 Escape 瑕疵) "
        f"-> {os.path.abspath(mock_pool_dir)}",
        log_files,
    )


# =====================================================================
# 圓心偵測模組 (支援直接從記憶體 RGB 陣列偵測並快取，免去 Step 3 重複讀大圖)
# =====================================================================

def detect_circular_wafer_geometry_from_rgb(
    full_rgb: np.ndarray, wafer_edge_mode: str, category: str
) -> Tuple[bool, float, float, float]:
    """直接對已載入記憶體的 RGB 原圖偵測圓形晶圓幾何資訊，回傳 (是否啟用, cy, cx, radius)。"""
    h, w = full_rgb.shape[:2]
    mode = wafer_edge_mode.lower().strip()
    if mode == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5

    cat_lower = category.lower()
    is_circular_cat = any(k in cat_lower for k in ["wafer", "bottle", "cable", "coin", "lens", "circle"])

    scale = 256.0 / max(h, w)
    sh, sw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
    try:
        small_rgb = cv2.resize(full_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        corner_sz = max(4, min(sh, sw) // 10)
        corners = np.concatenate([
            blur[:corner_sz, :corner_sz].reshape(-1),
            blur[:corner_sz, -corner_sz:].reshape(-1),
            blur[-corner_sz:, :corner_sz].reshape(-1),
            blur[-corner_sz:, -corner_sz:].reshape(-1),
        ])
        center_patch = blur[sh // 3: 2 * sh // 3, sw // 3: 2 * sw // 3].reshape(-1)
        corner_mean = float(np.mean(corners))
        center_mean = float(np.mean(center_patch))
        corner_std = float(np.std(corners))

        if abs(center_mean - corner_mean) > 12.0:
            thresh_type = cv2.THRESH_BINARY if center_mean > corner_mean else cv2.THRESH_BINARY_INV
            _, bw = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 0.18 * (sh * sw):
                    (cx_s, cy_s), r_s = cv2.minEnclosingCircle(c)
                    circle_area = np.pi * (r_s ** 2) + 1e-6
                    circularity_fill = area / circle_area
                    if (
                        0.60 <= circularity_fill <= 1.15
                        and abs(cx_s - sw * 0.5) < 0.18 * sw
                        and abs(cy_s - sh * 0.5) < 0.18 * sh
                        and r_s <= 0.58 * max(sh, sw)
                    ):
                        if mode == "on" or is_circular_cat or corner_std < 28.0:
                            return True, float(cy_s / scale), float(cx_s / scale), float(r_s / scale)
    except Exception:
        pass

    if mode == "on" or is_circular_cat:
        return True, h * 0.5, w * 0.5, min(h, w) * 0.48
    return False, h * 0.5, w * 0.5, min(h, w) * 0.5


def detect_circular_wafer_geometry(
    img_path: str, h: int, w: int, wafer_edge_mode: str, category: str
) -> Tuple[bool, float, float, float]:
    """容錯版：若 manifest 未快取 wafer_geom 時，才從磁碟讀取原圖偵測圓心。"""
    if wafer_edge_mode.lower().strip() == "off":
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5
    try:
        full_rgb = load_image_rgb_uint8(img_path)
        res = detect_circular_wafer_geometry_from_rgb(full_rgb, wafer_edge_mode, category)
        del full_rgb
        return res
    except Exception:
        return False, h * 0.5, w * 0.5, min(h, w) * 0.5


# =====================================================================
# 智慧解析度防呆平行切塊 (Resolution-Aware Parallel Tiling) 與無壓縮存檔模組
# =====================================================================

def get_axis_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """計算滑動視窗座標，最後一格自動回推對齊邊界，確保所有切塊維持固定正方形不變形。"""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size, stride))
    if not positions or positions[-1] + tile_size < length:
        positions.append(length - tile_size)
    return positions


def is_background_tile(tile_rgb: np.ndarray, bg_std_thresh: float) -> bool:
    """判斷是否為純黑/純白無資訊背景塊。"""
    if bg_std_thresh <= 0:
        return False
    return float(np.std(tile_rgb)) < bg_std_thresh


def inspect_image_size_fast(img_path: str) -> Tuple[int, int]:
    """快速讀取圖片尺寸 (H, W)，不解碼整張像素以加速防呆檢查。"""
    try:
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size
            return int(h), int(w)
    except Exception:
        img = load_image_rgb_uint8(img_path)
        return int(img.shape[0]), int(img.shape[1])


def estimate_safe_tiling_params(
    hw_list: List[Tuple[int, int]],
    tile_size: int,
    tile_overlap: int,
    min_tiling_res: int,
    disable_tiling: bool,
    log_files: List[str],
) -> Tuple[int, int]:
    """主記憶體 RAM 防爆保護器。"""
    if disable_tiling:
        return tile_size, tile_overlap

    avail_bytes = get_available_ram_bytes()
    bytes_per_tile = 37 * 37 * 1024 * 2
    max_safe_tiles = max(2000, int((avail_bytes * 0.55) / bytes_per_tile))
    overlap_ratio = tile_overlap / max(1.0, float(tile_size))
    cur_ts = tile_size
    cur_ov = tile_overlap

    while cur_ts < 2048:
        stride = max(1, cur_ts - cur_ov)
        est_tiles = 0
        for (h, w) in hw_list:
            if max(h, w) >= min_tiling_res and (h > cur_ts or w > cur_ts):
                est_tiles += len(get_axis_positions(h, cur_ts, stride)) * len(get_axis_positions(w, cur_ts, stride))
            else:
                est_tiles += 1
        if est_tiles <= max_safe_tiles:
            break
        next_ts = cur_ts * 2
        next_ov = int(round(next_ts * overlap_ratio))
        log_and_print(
            f"[🛡️ 記憶體防爆自動調節] 預估切塊數 ({est_tiles} 塊) 接近可用 RAM 上限 "
            f"(Avail={avail_bytes / (1024**3):.1f} GiB, 安全上限≈{max_safe_tiles} 塊)，"
            f"自動將切塊尺寸從 {cur_ts}x{cur_ts} 升級為 {next_ts}x{next_ts} (Overlap={next_ov})！",
            log_files,
        )
        cur_ts, cur_ov = next_ts, next_ov
    return cur_ts, cur_ov


def _tile_single_image_worker(task_args: Tuple) -> Tuple[str, Dict, int, int]:
    (
        fpath,
        h,
        w,
        rel_fname,
        stem,
        tiled_pool_dir,
        tile_size,
        stride,
        min_tiling_res,
        bg_std_thresh,
        disable_tiling,
        wafer_edge_mode,
        category,
    ) = task_args
    cv2.setNumThreads(1)
    abs_fpath = os.path.abspath(fpath)
    should_tile = (not disable_tiling) and (max(h, w) >= min_tiling_res) and (h > tile_size or w > tile_size)

    if not should_tile:
        tile_stem = f"{stem}__full_y0000_x0000_h{h:04d}_w{w:04d}"
        tiles_info = [{
            "tile_stem": tile_stem,
            "tile_fname": os.path.basename(abs_fpath),
            "tile_abs_path": abs_fpath,
            "is_full_image": True,
            "y": 0,
            "x": 0,
            "h": int(h),
            "w": int(w),
        }]
        entry = {
            "stem": stem,
            "path": abs_fpath,
            "height": int(h),
            "width": int(w),
            "tiling_applied": False,
            "tiles": tiles_info,
        }
        return rel_fname, entry, 1, 0

    img = load_image_rgb_uint8(fpath)
    # 順便在原圖已載入記憶體時快取圓心座標，Step 3 完全免重讀 5000x5000 原圖
    use_circle, cy_f, cx_f, r_f = detect_circular_wafer_geometry_from_rgb(
        img, wafer_edge_mode=wafer_edge_mode, category=category
    )

    ys = get_axis_positions(h, tile_size, stride)
    xs = get_axis_positions(w, tile_size, stride)
    tiles_info = []
    skipped_bg = 0
    for y in ys:
        for x in xs:
            ch = min(tile_size, h - y)
            cw = min(tile_size, w - x)
            crop = img[y:y + ch, x:x + cw]
            if is_background_tile(crop, bg_std_thresh) and (len(ys) * len(xs) > 1):
                skipped_bg += 1
                continue
            tile_stem = f"{stem}__tile_y{y:04d}_x{x:04d}_h{ch:04d}_w{cw:04d}"
            tile_fname = f"{tile_stem}.png"
            tile_out_path = os.path.join(tiled_pool_dir, tile_fname)
            # 取消 zlib 壓縮 (compress_level=0) 大幅降低切塊寫檔耗時
            Image.fromarray(crop).save(tile_out_path, format="PNG", compress_level=0)
            tiles_info.append({
                "tile_stem": tile_stem,
                "tile_fname": tile_fname,
                "tile_abs_path": os.path.abspath(tile_out_path),
                "is_full_image": False,
                "y": int(y),
                "x": int(x),
                "h": int(ch),
                "w": int(cw),
            })

    if not tiles_info:
        cy, cx = max(0, (h - tile_size) // 2), max(0, (w - tile_size) // 2)
        ch, cw = min(tile_size, h - cy), min(tile_size, w - cx)
        crop = img[cy:cy + ch, cx:cx + cw]
        tile_stem = f"{stem}__tile_y{cy:04d}_x{cx:04d}_h{ch:04d}_w{cw:04d}"
        tile_fname = f"{tile_stem}.png"
        tile_out_path = os.path.join(tiled_pool_dir, tile_fname)
        Image.fromarray(crop).save(tile_out_path, format="PNG", compress_level=0)
        tiles_info.append({
            "tile_stem": tile_stem,
            "tile_fname": tile_fname,
            "tile_abs_path": os.path.abspath(tile_out_path),
            "is_full_image": False,
            "y": int(cy),
            "x": int(cx),
            "h": int(ch),
            "w": int(cw),
        })

    del img
    entry = {
        "stem": stem,
        "path": abs_fpath,
        "height": int(h),
        "width": int(w),
        "tiling_applied": True,
        "wafer_geom": [bool(use_circle), float(cy_f), float(cx_f), float(r_f)],
        "tiles": tiles_info,
    }
    return rel_fname, entry, len(tiles_info), skipped_bg


def prepare_tiled_test_pool(
    pool_dir: str,
    tiled_pool_dir: str,
    manifest_save_dir: str,
    tile_size: int,
    tile_overlap: int,
    min_tiling_res: int,
    bg_std_thresh: float,
    disable_tiling: bool,
    pool_signature: str,
    tile_config_key: str,
    cached_manifest_path: str,
    log_files: List[str],
    wafer_edge_mode: str = "auto",
    category: str = "wafer",
    num_workers: int = 8,
) -> Dict:
    """遞迴搜尋測試盲池中的所有圖片，執行大圖防呆判定與多核心平行切塊。"""
    safe_rmtree(tiled_pool_dir)
    img_paths = scan_pool_images(pool_dir)
    if not img_paths:
        raise FileNotFoundError(f"在 {pool_dir} (含子資料夾) 中找不到任何圖片！")

    hw_list = [inspect_image_size_fast(p) for p in img_paths]
    tile_size, tile_overlap = estimate_safe_tiling_params(
        hw_list=hw_list,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        min_tiling_res=min_tiling_res,
        disable_tiling=disable_tiling,
        log_files=log_files,
    )
    stride = max(1, tile_size - tile_overlap)
    max_sides = [max(h, w) for (h, w) in hw_list]
    med_res = int(np.median(max_sides))
    max_res = int(np.max(max_sides))
    min_res = int(np.min(max_sides))

    large_img_count = sum(
        1 for (h, w) in hw_list
        if (not disable_tiling) and (max(h, w) >= min_tiling_res) and (h > tile_size or w > tile_size)
    )

    if disable_tiling:
        guardrail_msg = (
            f"[🛡️ 防呆檢測 — 已手動關閉切塊 (--disable_tiling)] "
            f"資料集解析度範圍: {min_res}~{max_res}px (中位數={med_res}px) -> 全部 {len(img_paths)} 張採用整張原圖推論。"
        )
    elif large_img_count == 0:
        guardrail_msg = (
            f"[🛡️ 防呆檢測 — 偵測為中低解析度資料集，自動跳過切塊] "
            f"盲池解析度範圍: {min_res}~{max_res}px (中位數={med_res}px) < 大圖門檻 ({min_tiling_res}px)。\n"
            f"   ↳ 為確保對廠外公開資料集之公平性與語意完整性，已自動切換為【整張原圖直接推論 (Full-Image Mode)】！"
        )
    else:
        os.makedirs(tiled_pool_dir, exist_ok=True)
        guardrail_msg = (
            f"[🛡️ 防呆檢測 — 偵測到高解析度大圖，自動啟用平行切塊] "
            f"盲池解析度範圍: {min_res}~{max_res}px (中位數={med_res}px) >= 大圖門檻 ({min_tiling_res}px)。\n"
            f"   ↳ 共 {large_img_count}/{len(img_paths)} 張大圖將執行 {tile_size}x{tile_size} (Overlap={tile_overlap}) "
            f"高解析度重疊切塊 (Workers={num_workers}, 無壓縮 PNG + 圓心快取)！"
        )
    log_and_print("\n" + guardrail_msg, log_files)

    manifest = {
        "pool_signature": pool_signature,
        "tile_config_key": tile_config_key,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "min_tiling_res": min_tiling_res,
        "large_img_count": large_img_count,
        "images": {},
    }

    tasks = []
    for fpath, (h, w) in zip(img_paths, hw_list):
        rel_fname = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_fname)[0]
        stem = re.sub(r"[\\/]+", "__", rel_no_ext)
        tasks.append((
            fpath, h, w, rel_fname, stem, tiled_pool_dir,
            tile_size, stride, min_tiling_res, bg_std_thresh,
            disable_tiling, wafer_edge_mode, category,
        ))

    total_tiles = 0
    skipped_bg_tiles = 0
    num_pool_imgs = len(tasks)
    eff_workers = max(1, min(int(num_workers), num_pool_imgs)) if large_img_count > 0 else 1
    t_tile_loop_start = time.perf_counter()

    ordered_results: Dict[str, Dict] = {}
    if eff_workers <= 1:
        for img_idx, t_args in enumerate(tasks, 1):
            rel_fname, entry, n_t, n_skip = _tile_single_image_worker(t_args)
            ordered_results[rel_fname] = entry
            total_tiles += n_t
            skipped_bg_tiles += n_skip
            if large_img_count > 0 and (img_idx == 1 or img_idx % 5 == 0 or img_idx == num_pool_imgs):
                _print_inline_progress(
                    stage_title="Step 1 大圖切塊準備",
                    current=img_idx,
                    total=num_pool_imgs,
                    start_t=t_tile_loop_start,
                    extra_info=f"累計圖塊: {total_tiles} (略過背景: {skipped_bg_tiles})",
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=eff_workers) as executor:
            futures = [executor.submit(_tile_single_image_worker, t_args) for t_args in tasks]
            for fut in as_completed(futures):
                rel_fname, entry, n_t, n_skip = fut.result()
                ordered_results[rel_fname] = entry
                total_tiles += n_t
                skipped_bg_tiles += n_skip
                completed += 1
                if completed == 1 or completed % 5 == 0 or completed == num_pool_imgs:
                    _print_inline_progress(
                        stage_title=f"Step 1 平行切塊(x{eff_workers})",
                        current=completed,
                        total=num_pool_imgs,
                        start_t=t_tile_loop_start,
                        extra_info=f"累計圖塊: {total_tiles} (略過背景: {skipped_bg_tiles})",
                    )

    for t_args in tasks:
        rf = t_args[3]
        if rf in ordered_results:
            manifest["images"][rf] = ordered_results[rf]

    gc.collect()
    manifest["total_tiles"] = total_tiles
    manifest["num_images"] = len(img_paths)
    manifest_path = os.path.join(manifest_save_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if cached_manifest_path:
        os.makedirs(os.path.dirname(cached_manifest_path), exist_ok=True)
        safe_copy(manifest_path, cached_manifest_path)

    if large_img_count > 0:
        log_and_print(
            f"[Tiling-Test] 已將 {len(img_paths)} 張原圖準備為 {total_tiles} 張推論圖塊 "
            f"(自動跳過 {skipped_bg_tiles} 張純背景角落塊, Workers={eff_workers}) -> {os.path.abspath(tiled_pool_dir)}",
            log_files,
        )
    return manifest


def is_tiled_png_pool_valid(
    tiled_pool_dir: str,
    cached_manifest_path: str,
    pool_signature: str,
    tile_config_key: str,
) -> Optional[Dict]:
    """檢查既有 cached_tile_manifest.json 與對應影像路徑是否完整且指紋相符。"""
    if not os.path.isfile(cached_manifest_path):
        return None
    try:
        with open(cached_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (
            manifest.get("pool_signature") != pool_signature
            or manifest.get("tile_config_key") != tile_config_key
            or not manifest.get("images")
        ):
            return None
        for meta in manifest["images"].values():
            if not os.path.exists(meta.get("path", "")):
                return None
            for t in meta.get("tiles", []):
                t_path = t.get("tile_abs_path") or os.path.join(tiled_pool_dir, t["tile_fname"])
                if not os.path.exists(t_path):
                    return None
        return manifest
    except Exception:
        return None


# =====================================================================
# MuSc-V2 Training-Free & Zero-Shot 互評推理引擎 (單次多層提取 + 多執行緒預讀 + 原生解析度存檔)
# =====================================================================

def resolve_offline_vit_ckpt(ckpt_path: str, log_files: List[str]) -> Optional[str]:
    """若 ckpt_path 為 'auto' 或不存在，自動搜尋本機快取中的 DINOv2 / DINOv3 / ViT 權重檔。"""
    if ckpt_path and ckpt_path.lower() != "auto":
        if os.path.exists(ckpt_path):
            return ckpt_path
        log_and_print(f"[MuSc-V2 Warning] 指定權重路徑不存在 ({ckpt_path})，嘗試自動搜尋本機快取...", log_files)

    search_dirs = [
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        os.path.expanduser("~/.cache/torch/hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(REPO_ROOT, "hub_cache"),
        os.path.join(REPO_ROOT, "checkpoints"),
        REPO_ROOT,
    ]
    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            for ext in ("*.pth", "*.pt", "*.bin", "*.safetensors"):
                candidates.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    for keyword in ("dinov2_vitl14", "dinov3", "dinov2", "dino", "vit"):
        matched = [p for p in candidates if keyword in os.path.basename(p).lower()]
        if matched:
            chosen = sorted(matched)[0]
            log_and_print(f"[MuSc-V2 Backbone] 已自動找到本機離線權重 ({keyword.upper()}): {chosen}", log_files)
            return chosen
    return None


def load_muscv2_vision_backbone(
    backbone_name: str,
    ckpt_path: str,
    device: torch.device,
    log_files: List[str],
) -> Tuple[nn.Module, int, int, int, List[int]]:
    """載入 MuSc-V2 2D 視覺特徵骨幹，自動偵測 Embed_Dim、Patch Size、輸入解析度與提取層數。"""
    real_ckpt = resolve_offline_vit_ckpt(ckpt_path, log_files)
    model = None
    patch_size = 14
    num_layers = 24
    embed_dim = 1024

    if real_ckpt is not None and real_ckpt.endswith((".pth", ".pt", ".bin")):
        ckpt = torch.load(real_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("teacher", ckpt.get("state_dict", ckpt)))
        cleaned_sd = {}
        for k, v in state_dict.items():
            nk = re.sub(r"^(backbone\.|module\.|visual\.)", "", k)
            cleaned_sd[nk] = v

        proj_w = cleaned_sd.get("patch_embed.proj.weight", None)
        if proj_w is not None:
            embed_dim = int(proj_w.shape[0])
            patch_size = int(proj_w.shape[2])
        else:
            embed_dim = 1024 if "vitl" in backbone_name.lower() or "large" in backbone_name.lower() else 768
            patch_size = 16 if "16" in backbone_name else 14

        block_ids = [
            int(k.split(".")[1]) for k in cleaned_sd.keys()
            if k.startswith("blocks.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        ]
        if block_ids:
            num_layers = max(block_ids) + 1
        has_reg = any("register_tokens" in k for k in cleaned_sd.keys())

        log_and_print(
            f"[MuSc-V2 Backbone 規格偵測] Embed={embed_dim} | Patch={patch_size}x{patch_size} | "
            f"Blocks={num_layers} | Registers={has_reg}",
            log_files,
        )

        try:
            import timm
            if patch_size == 14:
                if embed_dim == 1024:
                    timm_name = "vit_large_patch14_dinov2.lvd142m"
                elif embed_dim == 768:
                    timm_name = "vit_base_patch14_dinov2.lvd142m"
                elif embed_dim == 384:
                    timm_name = "vit_small_patch14_dinov2.lvd142m"
                else:
                    timm_name = "vit_giant_patch14_dinov2.lvd142m"
            else:
                if embed_dim == 1024:
                    timm_name = "vit_large_patch16_dinov3.lvd1689m" if "dinov3" in real_ckpt.lower() else "vit_large_patch16_224"
                elif embed_dim == 768:
                    timm_name = "vit_base_patch16_dinov3.lvd1689m" if "dinov3" in real_ckpt.lower() else "vit_base_patch16_224"
                else:
                    timm_name = "vit_small_patch16_dinov3.lvd1689m" if "dinov3" in real_ckpt.lower() else "vit_small_patch16_224"

            model = timm.create_model(
                timm_name,
                pretrained=False,
                img_size=518 if patch_size == 14 else 512,
                num_classes=0,
            )
            missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
            log_and_print(
                f"[MuSc-V2 Backbone] 已透過 timm ({timm_name}) 離線載入本地權重 "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                log_files,
            )
        except Exception as e_timm:
            log_and_print(f"[MuSc-V2 Info] timm 離線建構略過 ({e_timm})，改用 MuSc-V2 內建 DINOv2 架構載入...", log_files)
            from models.backbone.dinov2.vision_transformer import DinoVisionTransformer
            model = DinoVisionTransformer(
                img_size=518,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=num_layers,
                num_heads=max(1, embed_dim // 64),
                num_register_tokens=4 if has_reg else 0,
            )
            model.load_state_dict(cleaned_sd, strict=False)

    if model is None:
        try:
            from models.backbone import _backbones
            model = _backbones.load(backbone_name)
            patch_size = 14 if "14" in backbone_name else 16
            num_layers = len(getattr(model, "blocks", [0] * 24))
            embed_dim = getattr(model, "embed_dim", 1024)
            log_and_print(f"[MuSc-V2 Backbone] 已透過 MuSc-V2 內建 _backbones 載入: {backbone_name}", log_files)
        except Exception:
            import timm
            t_name = "vit_large_patch14_dinov2.lvd142m" if "14" in backbone_name else backbone_name
            model = timm.create_model(t_name, pretrained=True, img_size=518, num_classes=0)
            patch_size = 14 if "14" in t_name else 16
            num_layers = len(getattr(model, "blocks", [0] * 24))
            embed_dim = getattr(model, "embed_dim", 1024)
            log_and_print(f"[MuSc-V2 Backbone] 已透過 timm 載入: {t_name}", log_files)

    model = model.to(device).eval()
    img_input_size = 518 if patch_size == 14 else 512
    if num_layers == 24:
        default_layers = [5, 11, 17, 23]
    elif num_layers == 12:
        default_layers = [2, 5, 8, 11]
    else:
        default_layers = sorted(list({
            max(0, int(round((num_layers - 1) * q))) for q in (0.22, 0.48, 0.74, 1.0)
        }))
    return model, embed_dim, patch_size, img_input_size, default_layers


def _run_msm_for_single_layer(
    layer_raw_buf: torch.Tensor,
    l_idx: int,
    layer_step: int,
    r_list: List[int],
    total_tiles: int,
    num_patches: int,
    embed_dim: int,
    grid_hw: int,
    parent_ids_tensor: torch.Tensor,
    mutual_ratio: float,
    max_peer_tiles: int,
    num_combos: int,
    msm_chunk_size: int,
    device: torch.device,
    patch_scores_accum: torch.Tensor,
    t_total_infer_start: float,
) -> None:
    """針對單一層的特徵緩衝區，依序計算所有 r_list 尺度的跨片互評 (MSM)。"""
    for r_step, r in enumerate(r_list, 1):
        combo_idx = (layer_step - 1) * len(r_list) + r_step
        t_msm_step = time.perf_counter()

        bank_dev = device if (device.type == "cuda" and total_tiles <= 1200) else torch.device("cpu")
        bank_host = torch.empty((total_tiles, num_patches, embed_dim), dtype=torch.float16, device=bank_dev)

        if r == 1:
            for b_s in range(0, total_tiles, 128):
                b_e = min(total_tiles, b_s + 128)
                normed = F.normalize(layer_raw_buf[b_s:b_e].to(device).float(), dim=-1).half()
                bank_host[b_s:b_e].copy_(normed)
                del normed
        else:
            for b_s in range(0, total_tiles, 64):
                b_e = min(total_tiles, b_s + 64)
                sub_2d = (
                    layer_raw_buf[b_s:b_e].to(device).float()
                    .transpose(1, 2).reshape(b_e - b_s, embed_dim, grid_hw, grid_hw)
                )
                agg_2d = F.avg_pool2d(sub_2d, kernel_size=r, stride=1, padding=r // 2)
                agg_flat = agg_2d.flatten(2).transpose(1, 2)
                normed = F.normalize(agg_flat, dim=-1).half()
                bank_host[b_s:b_e].copy_(normed)
                del sub_2d, agg_2d, agg_flat, normed

        for q_start in range(0, total_tiles, msm_chunk_size):
            q_end = min(total_tiles, q_start + msm_chunk_size)
            q_feat = bank_host[q_start:q_end].to(device)
            b_q = q_end - q_start

            q_parents = parent_ids_tensor[q_start:q_end].unsqueeze(1)
            valid_peer_mask = (parent_ids_tensor.unsqueeze(0) != q_parents)
            if not torch.any(valid_peer_mask):
                self_indices = torch.arange(q_start, q_end).unsqueeze(1)
                valid_peer_mask = (torch.arange(total_tiles).unsqueeze(0) != self_indices)

            if max_peer_tiles > 0 and total_tiles > max_peer_tiles:
                peer_indices = torch.randperm(total_tiles)[:max_peer_tiles]
                sub_bank_t = bank_host[peer_indices].to(device).transpose(1, 2).contiguous()
                sub_mask = valid_peer_mask[:, peer_indices].to(device)
            else:
                sub_bank_t = bank_host.to(device).transpose(1, 2).contiguous()
                sub_mask = valid_peer_mask.to(device)

            num_sub_peers = sub_bank_t.shape[0]
            peer_step = 64 if device.type == "cuda" else 32
            min_dist_chunks = []
            for p_s in range(0, num_sub_peers, peer_step):
                p_e = min(num_sub_peers, p_s + peer_step)
                sim_chunk = torch.einsum("qic,mcj->qmij", q_feat, sub_bank_t[p_s:p_e])
                min_dist_chunks.append((1.0 - sim_chunk.max(dim=-1).values).float())
                del sim_chunk

            min_dist_per_peer = torch.cat(min_dist_chunks, dim=1)
            del min_dist_chunks, sub_bank_t

            invalid_mask_3d = (~sub_mask).unsqueeze(-1).expand_as(min_dist_per_peer)
            min_dist_per_peer = min_dist_per_peer.masked_fill(invalid_mask_3d, float("inf"))

            sorted_dists, _ = torch.sort(min_dist_per_peer, dim=1)
            valid_counts = sub_mask.sum(dim=1).clamp(min=1)
            for bi in range(b_q):
                v_cnt = int(valid_counts[bi].item())
                k_take = max(1, int(round(v_cnt * mutual_ratio)))
                k_take = min(k_take, v_cnt)
                score_bi = sorted_dists[bi, :k_take, :].mean(dim=0)
                patch_scores_accum[q_start + bi] += score_bi.cpu() / float(num_combos)
            del min_dist_per_peer, sorted_dists

        del bank_host
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _print_inline_progress(
            stage_title="Step 2.2 MuSc-V2 互評 (MSM)",
            current=combo_idx,
            total=num_combos,
            start_t=t_total_infer_start,
            extra_info=f"Layer={l_idx}, r={r} ({time.perf_counter() - t_msm_step:.1f}s)",
        )


@torch.inference_mode()
def run_muscv2_zero_shot_inference(
    manifest: Dict,
    tiled_pool_dir: str,
    tile_amap_dir: str,
    backbone_name: str,
    ckpt_path: str,
    r_list: List[int],
    custom_layers: List[int],
    mutual_ratio: float,
    rscon_weight: float,
    infer_batch_size: int,
    max_peer_tiles: int,
    log_files: List[str],
    num_workers: int = 8,
    amp_dtype_str: str = "bf16",
):
    """
    執行 MuSc-V2 Training-Free & Zero-Shot 互評推論管線 (無損加速版)：
    - 開啟 TF32 + cuDNN benchmark + AMP (bf16/fp16) + ThreadPoolExecutor 多執行緒讀圖與 Pin Memory。
    - 智慧單次 Forward 機制：若可用 RAM 足夠同時暫存全部目標層，僅執行 1 次 Backbone Forward Pass（提速 4 倍）；
      若 RAM 不足則自動退回單層串流防爆模式。
    - 輸出原生解析度 (grid_hw x grid_hw) float32 .npy，磁碟寫入量縮小 ~180 倍且零精度損失。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_str = amp_dtype_str.lower().strip()
    amp_dtype = torch.bfloat16 if amp_str == "bf16" else (torch.float16 if amp_str == "fp16" else None)
    use_amp = (amp_dtype is not None) and (device.type == "cuda")

    model, embed_dim, patch_size, img_input_size, default_layers = load_muscv2_vision_backbone(
        backbone_name=backbone_name,
        ckpt_path=ckpt_path,
        device=device,
        log_files=log_files,
    )
    use_layers = custom_layers if custom_layers else default_layers
    grid_hw = img_input_size // patch_size
    num_patches = grid_hw * grid_hw

    tile_records = []
    for parent_idx, (fname, meta) in enumerate(manifest["images"].items()):
        for t in meta["tiles"]:
            t_path = t.get("tile_abs_path") or os.path.join(tiled_pool_dir, t["tile_fname"])
            tile_records.append({
                "parent_idx": parent_idx,
                "parent_fname": fname,
                "tile_stem": t["tile_stem"],
                "tile_fname": t["tile_fname"],
                "tile_path": t_path,
                "h": int(t["h"]),
                "w": int(t["w"]),
            })
    total_tiles = len(tile_records)

    # 評估系統可用 RAM 是否足以一次收集所有層特徵 (免去重複跑 4 次 Backbone 與讀圖)
    num_layers_to_run = len(use_layers)
    single_layer_bytes = total_tiles * num_patches * embed_dim * 2
    avail_ram = get_available_ram_bytes()
    can_single_pass = (single_layer_bytes * (num_layers_to_run + 1)) <= int(avail_ram * 0.60)
    extract_mode_str = (
        f"單次 Forward 全層收集 (4x提速, 預估佔用 {(single_layer_bytes * num_layers_to_run)/(1024**3):.1f} GiB)"
        if can_single_pass
        else f"單層串流省記憶體模式 (AvailRAM={avail_ram/(1024**3):.1f} GiB)"
    )

    mode_desc = "高解析度大圖切塊互評" if manifest.get("large_img_count", 0) > 0 else "整張原圖直接互評 (Full-Image)"
    log_and_print(
        f"\n[Step 2 — MuSc-V2 Zero-Shot 互評推理 ({mode_desc})] Device={device} | Units={total_tiles} | "
        f"Layers={use_layers} | r_list={r_list} | Mutual_Ratio={mutual_ratio:.2f} | RsCon={rscon_weight:.2f}\n"
        f"   ↳ 加速配置: BatchSize={infer_batch_size} | I/O Workers={num_workers} | "
        f"AMP={amp_str if use_amp else 'off'} | NativeFloat32Npy=ON | {extract_mode_str}",
        log_files,
    )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("載入的視覺骨幹模型不含 .blocks 屬性，無法掛載 MuSc-V2 多層特徵提取 Hook！")

    transform = transforms.Compose([
        transforms.Resize((img_input_size, img_input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _load_and_transform(rec: Dict) -> torch.Tensor:
        rgb = load_image_rgb_uint8(rec["tile_path"])
        return transform(Image.fromarray(rgb))

    global_feats_accum = torch.zeros((total_tiles, embed_dim), dtype=torch.float32, device="cpu")
    patch_scores_accum = torch.zeros((total_tiles, num_patches), dtype=torch.float32, device="cpu")
    parent_ids_tensor = torch.tensor([r["parent_idx"] for r in tile_records], dtype=torch.long)

    num_combos = num_layers_to_run * len(r_list)
    msm_chunk_size = (8 if total_tiles >= 600 else 16) if device.type == "cuda" else 8
    t_total_infer_start = time.perf_counter()

    if can_single_pass:
        # 【快速路徑】單次 Forward Pass 同時 Hook 收集所有目標層，省去 75% 讀圖與 Backbone 運算
        all_layer_bufs = [
            torch.empty((total_tiles, num_patches, embed_dim), dtype=torch.float16, device="cpu")
            for _ in use_layers
        ]
        hook_holder: Dict[int, torch.Tensor] = {}
        handles = []
        for slot_i, l_idx in enumerate(use_layers):
            def _make_hook(slot_idx: int):
                def _hook(_mod, _inp, out):
                    hook_holder[slot_idx] = out[0] if isinstance(out, tuple) else out
                return _hook
            handles.append(blocks[l_idx].register_forward_hook(_make_hook(slot_i)))

        t_feat = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as io_pool:
            for start_i in range(0, total_tiles, infer_batch_size):
                end_i = min(total_tiles, start_i + infer_batch_size)
                batch_recs = tile_records[start_i:end_i]
                batch_tensors = list(io_pool.map(_load_and_transform, batch_recs))
                x_batch = torch.stack(batch_tensors, dim=0)
                if device.type == "cuda":
                    x_batch = x_batch.pin_memory().to(device, non_blocking=True)
                else:
                    x_batch = x_batch.to(device)

                hook_holder.clear()
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _ = model(x_batch)

                for slot_i in range(num_layers_to_run):
                    tok = hook_holder[slot_i].float()
                    patch_tok = tok[:, -num_patches:, :]
                    g_batch = F.normalize(patch_tok.mean(dim=1), dim=-1).cpu()
                    global_feats_accum[start_i:end_i] += g_batch / float(num_layers_to_run)
                    all_layer_bufs[slot_i][start_i:end_i].copy_(patch_tok.to(dtype=torch.float16, device="cpu"))
                    del tok, patch_tok
                del x_batch
                hook_holder.clear()

                if end_i == total_tiles or (start_i // infer_batch_size) % 4 == 0:
                    _print_inline_progress(
                        stage_title="Step 2.1 單次全層特徵提取",
                        current=end_i,
                        total=total_tiles,
                        start_t=t_feat,
                        extra_info=f"Layers={use_layers}",
                    )

        for hdl in handles:
            hdl.remove()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for layer_step, l_idx in enumerate(use_layers, 1):
            cur_buf = all_layer_bufs[layer_step - 1]
            _run_msm_for_single_layer(
                layer_raw_buf=cur_buf,
                l_idx=l_idx,
                layer_step=layer_step,
                r_list=r_list,
                total_tiles=total_tiles,
                num_patches=num_patches,
                embed_dim=embed_dim,
                grid_hw=grid_hw,
                parent_ids_tensor=parent_ids_tensor,
                mutual_ratio=mutual_ratio,
                max_peer_tiles=max_peer_tiles,
                num_combos=num_combos,
                msm_chunk_size=msm_chunk_size,
                device=device,
                patch_scores_accum=patch_scores_accum,
                t_total_infer_start=t_total_infer_start,
            )
            all_layer_bufs[layer_step - 1] = None
            del cur_buf
            gc.collect()
        del all_layer_bufs
    else:
        # 【省記憶體備援路徑】單層串流防爆記憶體模式 (同樣享有高 Batch + AMP + 多執行緒預讀)
        storage_device = device if (device.type == "cuda" and total_tiles <= 350) else torch.device("cpu")
        layer_raw_buf = torch.empty((total_tiles, num_patches, embed_dim), dtype=torch.float16, device=storage_device)

        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as io_pool:
            for layer_step, l_idx in enumerate(use_layers, 1):
                hook_holder = {}
                def _hook(_mod, _inp, out):
                    hook_holder["feat"] = out[0] if isinstance(out, tuple) else out
                handle = blocks[l_idx].register_forward_hook(_hook)

                t_layer_feat = time.perf_counter()
                for start_i in range(0, total_tiles, infer_batch_size):
                    end_i = min(total_tiles, start_i + infer_batch_size)
                    batch_recs = tile_records[start_i:end_i]
                    batch_tensors = list(io_pool.map(_load_and_transform, batch_recs))
                    x_batch = torch.stack(batch_tensors, dim=0)
                    if device.type == "cuda":
                        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
                    else:
                        x_batch = x_batch.to(device)

                    hook_holder.clear()
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        _ = model(x_batch)

                    tok = hook_holder["feat"].float()
                    patch_tok = tok[:, -num_patches:, :]
                    g_batch = F.normalize(patch_tok.mean(dim=1), dim=-1).cpu()
                    global_feats_accum[start_i:end_i] += g_batch / float(num_layers_to_run)
                    layer_raw_buf[start_i:end_i].copy_(patch_tok.to(dtype=torch.float16, device=storage_device))
                    del x_batch, tok, patch_tok
                    hook_holder.clear()

                    if end_i == total_tiles or (start_i // infer_batch_size) % 4 == 0:
                        _print_inline_progress(
                            stage_title=f"Step 2.{layer_step}a Layer-{l_idx} 特徵提取",
                            current=end_i,
                            total=total_tiles,
                            start_t=t_layer_feat,
                            extra_info=f"緩衝區={storage_device}",
                        )
                handle.remove()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                _run_msm_for_single_layer(
                    layer_raw_buf=layer_raw_buf,
                    l_idx=l_idx,
                    layer_step=layer_step,
                    r_list=r_list,
                    total_tiles=total_tiles,
                    num_patches=num_patches,
                    embed_dim=embed_dim,
                    grid_hw=grid_hw,
                    parent_ids_tensor=parent_ids_tensor,
                    mutual_ratio=mutual_ratio,
                    max_peer_tiles=max_peer_tiles,
                    num_combos=num_combos,
                    msm_chunk_size=msm_chunk_size,
                    device=device,
                    patch_scores_accum=patch_scores_accum,
                    t_total_infer_start=t_total_infer_start,
                )
        del layer_raw_buf

    gc.collect()
    global_feats_all = F.normalize(global_feats_accum, dim=-1)
    raw_tile_max = patch_scores_accum.max(dim=1).values
    rscon_scale = torch.ones(total_tiles, dtype=torch.float32)

    if rscon_weight > 0 and total_tiles > 2:
        g_sim = torch.mm(global_feats_all, global_feats_all.t())
        same_parent = (parent_ids_tensor.unsqueeze(1) == parent_ids_tensor.unsqueeze(0))
        if not torch.all(same_parent):
            g_sim = g_sim.masked_fill(same_parent, -1e4)
        else:
            g_sim.fill_diagonal_(-1e4)
        k_nbr = min(max(3, int(round(total_tiles * 0.15))), total_tiles - 1)
        top_sim, top_idx = torch.topk(g_sim, k=k_nbr, dim=1)
        nbr_weights = F.softmax(top_sim * 10.0, dim=1)
        nbr_scores = raw_tile_max[top_idx]
        expected_baseline = (nbr_weights * nbr_scores).sum(dim=1)
        global_med_baseline = float(torch.median(raw_tile_max).item())
        excess_shift = torch.clamp(expected_baseline - global_med_baseline, min=0.0)
        calibrated_max = torch.clamp(raw_tile_max - rscon_weight * excess_shift, min=0.0)
        rscon_scale = calibrated_max / torch.clamp(raw_tile_max, min=1e-6)

    # 直接以原生解析度 (grid_hw x grid_hw) float32 存檔，單塊僅 ~5.5 KB，寫入量縮小 ~180 倍且零精度損失
    os.makedirs(tile_amap_dir, exist_ok=True)
    amap_2d_all = patch_scores_accum.reshape(total_tiles, grid_hw, grid_hw).numpy()
    for idx, rec in enumerate(tile_records):
        amap = (amap_2d_all[idx] * float(rscon_scale[idx].item())).astype(np.float32)
        np.save(os.path.join(tile_amap_dir, f"{rec['tile_stem']}.npy"), amap)

    del model, patch_scores_accum, amap_2d_all, global_feats_accum
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# =====================================================================
# 降維縫合、四合一防誤報純化、多核心平行處理與紅框定位
# =====================================================================

def build_2d_feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """建立 2D 餘弦平滑權重遮罩 (Raised-Cosine Window)，消除重疊切塊拼接時的邊界接縫。"""
    def _1d_ramp(length: int, ov: int) -> np.ndarray:
        w1d = np.ones(length, dtype=np.float32)
        ramp = min(max(ov, 4), max(1, length // 3))
        if ramp > 1:
            t = np.linspace(0.0, np.pi, ramp, dtype=np.float32)
            rising = 0.1 + 0.9 * (0.5 * (1.0 - np.cos(t)))
            w1d[:ramp] = rising
            w1d[-ramp:] = rising[::-1]
        return w1d
    return np.outer(_1d_ramp(h, overlap), _1d_ramp(w, overlap)).astype(np.float32)


def find_tile_npy(tile_amap_dir: str, tile_stem: str, tile_fname: str) -> str:
    """容錯尋找輸出的子圖塊 .npy 熱力圖檔案。"""
    for c in [
        os.path.join(tile_amap_dir, f"{tile_stem}.npy"),
        os.path.join(tile_amap_dir, f"{tile_fname}.npy"),
    ]:
        if os.path.exists(c):
            return c
    matches = glob.glob(os.path.join(tile_amap_dir, f"*{tile_stem}*.npy"), recursive=True)
    return matches[0] if matches else ""


def compute_stitch_canvas_shape(orig_h: int, orig_w: int, stitch_max_dim: int) -> Tuple[int, int, float, float]:
    """計算降維縫合畫布尺寸 (sh, sw) 與縮放比例 (scale_y, scale_x)。"""
    if stitch_max_dim > 0 and max(orig_h, orig_w) > stitch_max_dim:
        s = stitch_max_dim / float(max(orig_h, orig_w))
        sh = max(1, int(round(orig_h * s)))
        sw = max(1, int(round(orig_w * s)))
        return sh, sw, sh / float(orig_h), sw / float(orig_w)
    return orig_h, orig_w, 1.0, 1.0


def stitch_single_image_amap(
    img_meta: Dict,
    tile_amap_dir: str,
    tile_overlap: int,
    stitch_max_dim: int = 1024,
    delete_tiles: bool = True,
) -> np.ndarray:
    """將單張影像的所有子圖塊 .npy 無縫縫合至目標畫布 (預設 <= 1024px)，並在縫合後立即刪除暫存 .npy。"""
    orig_h, orig_w = img_meta["height"], img_meta["width"]
    sh, sw, scale_y, scale_x = compute_stitch_canvas_shape(orig_h, orig_w, stitch_max_dim)
    tiles = img_meta["tiles"]

    if len(tiles) == 1 and tiles[0].get("is_full_image", False):
        t = tiles[0]
        npy_path = find_tile_npy(tile_amap_dir, t["tile_stem"], t["tile_fname"])
        if npy_path:
            amap = np.squeeze(np.load(npy_path).astype(np.float32))
            if amap.shape[0] != sh or amap.shape[1] != sw:
                amap = cv2.resize(amap, (sw, sh), interpolation=cv2.INTER_LINEAR)
            sigma = max(0.8, 4.0 * min(scale_y, scale_x))
            amap = cv2.GaussianBlur(amap, (0, 0), sigmaX=sigma)
            if delete_tiles:
                try:
                    os.remove(npy_path)
                except OSError:
                    pass
            return amap

    canvas_sum = np.zeros((sh, sw), dtype=np.float32)
    weight_sum = np.zeros((sh, sw), dtype=np.float32)
    weight_cache: Dict[Tuple[int, int], np.ndarray] = {}
    consumed_tile_npys: List[str] = []
    scaled_overlap = max(4, int(round(tile_overlap * min(scale_y, scale_x))))

    for t in tiles:
        npy_path = find_tile_npy(tile_amap_dir, t["tile_stem"], t["tile_fname"])
        if not npy_path:
            continue
        amap = np.load(npy_path).astype(np.float32)
        consumed_tile_npys.append(npy_path)
        amap = np.squeeze(amap)
        if amap.ndim != 2:
            continue

        y, x, th, tw = t["y"], t["x"], t["h"], t["w"]
        sy1 = int(round(y * scale_y))
        sx1 = int(round(x * scale_x))
        sy2 = min(sh, max(sy1 + 1, int(round((y + th) * scale_y))))
        sx2 = min(sw, max(sx1 + 1, int(round((x + tw) * scale_x))))
        sth, stw = sy2 - sy1, sx2 - sx1

        if amap.shape[0] != sth or amap.shape[1] != stw:
            amap = cv2.resize(amap, (stw, sth), interpolation=cv2.INTER_LINEAR)

        key = (sth, stw)
        if key not in weight_cache:
            weight_cache[key] = build_2d_feather_weight(sth, stw, scaled_overlap)
        w2d = weight_cache[key]

        canvas_sum[sy1:sy2, sx1:sx2] += amap * w2d
        weight_sum[sy1:sy2, sx1:sx2] += w2d

    valid_mask = weight_sum > 1e-6
    stitched_amap = np.zeros((sh, sw), dtype=np.float32)
    stitched_amap[valid_mask] = canvas_sum[valid_mask] / weight_sum[valid_mask]
    if np.any(valid_mask) and not np.all(valid_mask):
        baseline = float(np.percentile(stitched_amap[valid_mask], 5.0))
        stitched_amap[~valid_mask] = baseline

    # 在縫合後的 1024px 畫布做等比例輕量平滑，取代原本對每個 512px 圖塊做 sigma=4.0 的重複模糊
    sigma = max(0.8, 4.0 * min(scale_y, scale_x))
    stitched_amap = cv2.GaussianBlur(stitched_amap, (0, 0), sigmaX=sigma)

    if delete_tiles:
        for p in consumed_tile_npys:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
    return stitched_amap


def refine_amap_for_wafer_and_noise(
    full_amap: np.ndarray,
    img_path: str,
    category: str,
    border_ratio: float,
    wafer_edge_mode: str,
    wafer_edge_margin: float,
    speckle_filter_ratio: float,
    cached_wafer_geom: Optional[List] = None,
    orig_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """防誤報熱力圖純化器：矩形 Padding 抑制 + 圓周邊緣衰減 + 形態學微塵開運算 (支援降維畫布與圓心快取)。"""
    h, w = full_amap.shape[:2]
    work_map = full_amap.astype(np.float32).copy()
    img_median = float(np.median(work_map))

    if speckle_filter_ratio > 0:
        k_speckle = max(3, int(round(max(h, w) * speckle_filter_ratio)) | 1)
        if k_speckle >= 3:
            disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_speckle, k_speckle))
            opened_map = cv2.morphologyEx(work_map, cv2.MORPH_OPEN, disk_kernel)
            work_map = 0.75 * opened_map + 0.25 * cv2.GaussianBlur(work_map, (k_speckle, k_speckle), 0)

    if border_ratio > 0:
        by = max(2, int(round(h * border_ratio)))
        bx = max(2, int(round(w * border_ratio)))
        work_map[:by, :] = img_median
        work_map[-by:, :] = img_median
        work_map[:, :bx] = img_median
        work_map[:, -bx:] = img_median

    orig_h, orig_w = orig_hw if orig_hw is not None else (h, w)
    scale_y = h / float(max(1, orig_h))
    scale_x = w / float(max(1, orig_w))
    if wafer_edge_mode.lower().strip() == "off":
        use_circle, cy, cx, radius = False, h * 0.5, w * 0.5, min(h, w) * 0.5
    elif cached_wafer_geom is not None and len(cached_wafer_geom) == 4:
        use_circle = bool(cached_wafer_geom[0])
        cy = float(cached_wafer_geom[1]) * scale_y
        cx = float(cached_wafer_geom[2]) * scale_x
        radius = float(cached_wafer_geom[3]) * min(scale_y, scale_x)
    else:
        use_circle, cy_f, cx_f, r_f = detect_circular_wafer_geometry(
            img_path=img_path, h=orig_h, w=orig_w, wafer_edge_mode=wafer_edge_mode, category=category
        )
        cy = cy_f * scale_y
        cx = cx_f * scale_x
        radius = r_f * min(scale_y, scale_x)

    if use_circle and wafer_edge_margin > 0:
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.hypot(yy - cy, xx - cx).astype(np.float32)
        inner_r = radius * (1.0 - wafer_edge_margin)
        outer_r = radius * (1.0 - wafer_edge_margin * 0.15)
        radial_weight = np.ones((h, w), dtype=np.float32)
        transition_mask = (dist_from_center > inner_r) & (dist_from_center < outer_r)
        if np.any(transition_mask):
            t = (dist_from_center[transition_mask] - inner_r) / max(outer_r - inner_r, 1.0)
            radial_weight[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * t))
        radial_weight[dist_from_center >= outer_r] = 0.0
        work_map = img_median + radial_weight * (work_map - img_median)

    return work_map


def merge_nearby_bboxes(
    bboxes: List[Tuple[int, int, int, int, float, int]], merge_gap: int
) -> List[Tuple[int, int, int, int, float, int]]:
    """自動合併重疊或距離極近的碎片紅框。"""
    if len(bboxes) <= 1:
        return bboxes
    boxes = [list(b) for b in bboxes]
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2, score, area = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2, bscore, barea = boxes[j]
                if not (
                    x2 + merge_gap < bx1
                    or bx2 + merge_gap < x1
                    or y2 + merge_gap < by1
                    or by2 + merge_gap < y1
                ):
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    score = max(score, bscore)
                    area += barea
                    used[j] = True
                    merged = True
            new_boxes.append([x1, y1, x2, y2, score, area])
            used[i] = True
        boxes = new_boxes
    boxes.sort(key=lambda item: item[4], reverse=True)
    return [tuple(b) for b in boxes]


def extract_tight_bboxes_for_anomaly_image(
    refined_amap: np.ndarray,
    img_threshold: float,
    core_ratio: float = 0.55,
    min_area_ratio: float = 0.00012,
    raw_score_mode: bool = False,
) -> Tuple[List[Tuple[int, int, int, int, float, int]], float]:
    """峰值自適應核心收斂演算法，切出緊密貼合異常核心的紅框。"""
    h, w = refined_amap.shape[:2]
    work_map = refined_amap.copy()
    img_median = float(np.median(work_map))
    max_score = float(np.max(work_map))
    if max_score < img_threshold:
        return [], img_threshold

    core_thresh = img_median + core_ratio * (max_score - img_median)
    pixel_box_thresh = max(float(img_threshold * 0.92), float(core_thresh))

    binary_mask = (work_map >= pixel_box_thresh).astype(np.uint8) * 255
    if not raw_score_mode:
        k_size = max(3, int(round(max(h, w) / 500)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 1 if raw_score_mode else max(4, int(round((h * w) * min_area_ratio)))
    vis_scale = max(1.0, max(h, w) / 900.0)
    pad = max(3, int(round(4 * vis_scale)))

    raw_bboxes = []
    for lbl_idx in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl_idx]
        if area < min_area:
            continue
        region_mask = (labels[by:by + bh, bx:bx + bw] == lbl_idx)
        region_pixels = work_map[by:by + bh, bx:bx + bw][region_mask]
        if region_pixels.size == 0:
            continue
        if raw_score_mode:
            region_score = float(np.max(region_pixels))
        else:
            top_n = max(1, int(round(region_pixels.size * 0.25)))
            region_score = float(np.mean(np.partition(region_pixels, -top_n)[-top_n:]))
        if region_score < img_threshold:
            continue
        x1, y1 = max(0, int(bx - pad)), max(0, int(by - pad))
        x2, y2 = min(w - 1, int(bx + bw + pad)), min(h - 1, int(by + bh + pad))
        raw_bboxes.append((x1, y1, x2, y2, region_score, int(area)))

    if not raw_bboxes:
        return [], pixel_box_thresh
    merge_gap = max(6, int(round(max(h, w) * 0.015)))
    merged_bboxes = merge_nearby_bboxes(raw_bboxes, merge_gap=merge_gap)
    return merged_bboxes, pixel_box_thresh


def compute_dynamic_effective_threshold(
    pool_scores: List[float],
    thresh_mode: str = "auto",
    user_threshold: float = 0.0,
    min_threshold: float = 0.0,
    auto_percentile: float = 95.0,
    sigma_k: float = 3.5,
    top_k: int = 0,
) -> Tuple[float, str]:
    """雙模式門檻判定 (MuSc-V2 Zero-Shot 版)：Fixed 或 Min(3-Sigma, Modified Z-Score)。"""
    scores_arr = np.array(pool_scores, dtype=np.float32)
    if scores_arr.size == 0:
        fallback = float(user_threshold) if user_threshold > 0 else float(min_threshold)
        return fallback, f"無樣本保底門檻 ({fallback:.4f})"

    total_n = int(scores_arr.size)
    mean_val = float(np.mean(scores_arr))
    std_val = float(np.std(scores_arr))
    thresh_3sigma = mean_val + 3.0 * std_val

    med_val = float(np.median(scores_arr))
    mad_val = float(np.median(np.abs(scores_arr - med_val)))
    thresh_mod_z = med_val + (3.5 / 0.6745) * mad_val

    if user_threshold is not None and float(user_threshold) > 0.0:
        eff_thresh = float(user_threshold)
        eff_count = int(np.sum(scores_arr >= eff_thresh))
        eff_pct = (eff_count / total_n) * 100.0
        desc = (
            f"Fixed 固定門檻 (沿用基準集設定) = {eff_thresh:.4f} "
            f"(本時期切出 {eff_count}/{total_n} 張, {eff_pct:.2f}%)\n"
            f"           ├─ [本時期飄移參考] 3-Sigma (Shewhart 1931)         : {thresh_3sigma:.4f} (Mean={mean_val:.4f}, Std={std_val:.4f})\n"
            f"           └─ [本時期飄移參考] Modified Z-Score (Iglewicz 1993): {thresh_mod_z:.4f} (Median={med_val:.4f}, MAD={mad_val:.4f})"
        )
        return eff_thresh, desc

    if thresh_3sigma <= thresh_mod_z:
        eff_thresh, eff_name = thresh_3sigma, "3-Sigma (mean+3*std)"
        high_thresh, high_name = thresh_mod_z, "Modified Z-Score (3.5)"
    else:
        eff_thresh, eff_name = thresh_mod_z, "Modified Z-Score (3.5)"
        high_thresh, high_name = thresh_3sigma, "3-Sigma (mean+3*std)"

    eff_count = int(np.sum(scores_arr >= eff_thresh))
    eff_pct = (eff_count / total_n) * 100.0
    high_count = int(np.sum(scores_arr >= high_thresh))
    high_pct = (high_count / total_n) * 100.0
    high_percentile = float(np.mean(scores_arr < high_thresh) * 100.0)

    if high_count == 0:
        high_cut_info = f"高於全體最高分 ({float(np.max(scores_arr)):.4f})，切出 0 張 (0.00%)"
    else:
        high_cut_info = (
            f"切在 Rank {high_count}/{total_n} 之後 "
            f"(共 {high_count} 張 >= 該門檻, 佔 {high_pct:.2f}%, 約 P{high_percentile:.2f})"
        )

    desc = (
        f"Min(3-Sigma, Mod-Z) 採用 [{eff_name}] = {eff_thresh:.4f} "
        f"(切出 {eff_count}/{total_n} 張, {eff_pct:.2f}%)\n"
        f"           ├─ 3-Sigma (Shewhart 1931)         : {thresh_3sigma:.4f} (Mean={mean_val:.4f}, Std={std_val:.4f})\n"
        f"           ├─ Modified Z-Score (Iglewicz 1993): {thresh_mod_z:.4f} (Median={med_val:.4f}, MAD={mad_val:.4f})\n"
        f"           └─ 較高門檻 [{high_name}] = {high_thresh:.4f} -> {high_cut_info}"
    )
    return float(eff_thresh), desc


def compute_blind_pool_eval_metrics(ranking_records: List[Dict]) -> Optional[Dict[str, float]]:
    """若盲池檔名包含模擬標籤 (escape_*)，自動計算 Image-AUROC、I-AP、F1-max、Recall、FPR、Precision、F1。"""
    escape_recs = [r for r in ranking_records if os.path.basename(r["filename"]).startswith("escape_")]
    normal_recs = [r for r in ranking_records if not os.path.basename(r["filename"]).startswith("escape_")]
    if not escape_recs or not normal_recs:
        return None

    y_true = np.array(
        [1 if os.path.basename(r["filename"]).startswith("escape_") else 0 for r in ranking_records],
        dtype=np.int32,
    )
    y_score = np.array([float(r["max_score"]) for r in ranking_records], dtype=np.float64)
    y_pred = np.array([1 if "ANOMALY" in str(r["status"]) else 0 for r in ranking_records], dtype=np.int32)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    ranks = np.zeros(len(y_score_sorted), dtype=np.float64)
    i = 0
    n = len(y_score_sorted)
    while i < n:
        j = i + 1
        while j < n and y_score_sorted[j] == y_score_sorted[i]:
            j += 1
        avg_rank = 0.5 * ((n - j + 1) + (n - i))
        ranks[i:j] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[y_true_sorted == 1]))
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))

    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    thresh_idxs = np.r_[distinct_idx, n - 1]

    tps = tp_cum[thresh_idxs]
    fps = fp_cum[thresh_idxs]
    precisions = tps / np.maximum(1, tps + fps)
    recalls = tps / max(1, n_pos)

    rec_diff = np.diff(np.r_[0.0, recalls])
    ap = float(np.sum(rec_diff * precisions))

    f1_curve = (2.0 * precisions * recalls) / np.maximum(1e-12, precisions + recalls)
    best_idx = int(np.argmax(f1_curve))
    f1_max = float(f1_curve[best_idx])
    best_thresh = float(y_score_sorted[thresh_idxs[best_idx]])

    tp_cur = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_cur = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_cur = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall_cur = tp_cur / max(1, n_pos)
    fpr_cur = fp_cur / max(1, n_neg)
    prec_cur = tp_cur / max(1, tp_cur + fp_cur)
    f1_cur = (2.0 * prec_cur * recall_cur) / max(1e-12, prec_cur + recall_cur)

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": auroc,
        "ap": ap,
        "f1_max": f1_max,
        "best_thresh": best_thresh,
        "tp": tp_cur,
        "fp": fp_cur,
        "fn": fn_cur,
        "recall": recall_cur,
        "fpr": fpr_cur,
        "precision": prec_cur,
        "f1_cur": f1_cur,
    }


def _stitch_and_refine_single_worker(task: Dict) -> Dict:
    cv2.setNumThreads(1)
    idx = task["idx"]
    fname = task["fname"]
    meta = task["meta"]
    tile_amap_dir = task["tile_amap_dir"]
    tile_overlap = task["tile_overlap"]
    stitch_max_dim = task["stitch_max_dim"]
    refined_cache_dir = task["refined_cache_dir"]
    category = task["category"]
    border_ratio = task["border_ratio"]
    wafer_edge_mode = task["wafer_edge_mode"]
    wafer_edge_margin = task["wafer_edge_margin"]
    speckle_filter_ratio = task["speckle_filter_ratio"]
    raw_score_mode = task["raw_score_mode"]

    raw_amap = stitch_single_image_amap(
        img_meta=meta,
        tile_amap_dir=tile_amap_dir,
        tile_overlap=tile_overlap,
        stitch_max_dim=stitch_max_dim,
        delete_tiles=True,
    )
    raw_flat = raw_amap.reshape(-1)
    raw_single_max = float(np.max(raw_flat))
    peak_k = max(8, int(round(raw_flat.size * 0.00015)))
    raw_top_peak = float(np.mean(np.partition(raw_flat, -peak_k)[-peak_k:]))

    refined_amap = refine_amap_for_wafer_and_noise(
        full_amap=raw_amap,
        img_path=meta["path"],
        category=category,
        border_ratio=border_ratio,
        wafer_edge_mode=wafer_edge_mode,
        wafer_edge_margin=wafer_edge_margin,
        speckle_filter_ratio=speckle_filter_ratio,
        cached_wafer_geom=meta.get("wafer_geom", None),
        orig_hw=(meta["height"], meta["width"]),
    )
    ref_flat = refined_amap.reshape(-1)
    filtered_robust_max = float(np.mean(np.partition(ref_flat, -peak_k)[-peak_k:]))

    if raw_score_mode:
        active_amap = raw_amap
        active_max_score = raw_single_max
        active_flat = raw_flat
    else:
        active_amap = refined_amap
        active_max_score = filtered_robust_max
        active_flat = ref_flat

    top_k_count = max(20, int(active_flat.size * 0.0005))
    top_k_score = float(np.mean(np.partition(active_flat, -top_k_count)[-top_k_count:]))

    cache_npy_path = os.path.join(refined_cache_dir, f"{idx:05d}_{meta['stem']}.npy")
    try:
        np.save(cache_npy_path, active_amap.astype(np.float16))
    except Exception:
        cache_npy_path = ""
    del raw_amap, refined_amap, active_amap

    return {
        "idx": idx,
        "filename": fname,
        "stem": meta["stem"],
        "meta": meta,
        "cache_npy_path": cache_npy_path,
        "max_score": active_max_score,
        "raw_max_score": raw_single_max,
        "raw_top_peak_score": raw_top_peak,
        "filtered_max_score": filtered_robust_max,
        "filter_delta": filtered_robust_max - raw_single_max,
        "top_k_score": top_k_score,
    }


def stitch_and_filter_anomalies(
    manifest: Dict,
    tile_amap_dir: str,
    final_amap_dir: str,
    out_dir: str,
    category: str,
    thresh_mode: str,
    user_threshold: float,
    min_threshold: float,
    auto_percentile: float,
    top_k: int,
    sigma_k: float,
    core_ratio: float,
    border_ratio: float,
    wafer_edge_mode: str,
    wafer_edge_margin: float,
    speckle_filter_ratio: float,
    min_area_ratio: float,
    raw_score_mode: bool,
    log_files: List[str],
    stitch_max_dim: int = 1024,
    num_workers: int = 8,
) -> Tuple[int, int, float, str, str, str]:
    """多核心平行縫合與篩選（支援降維畫布 <= 1024px 與異常圖 1:1 座標映回）。"""
    os.makedirs(final_amap_dir, exist_ok=True)
    anomaly_dir = os.path.join(out_dir, "anomaly_images")
    anomaly_fullres_bbox_dir = os.path.join(anomaly_dir, "full_res_bbox_only")
    refined_cache_dir = os.path.join(out_dir, "_refined_cache_tmp")
    safe_rmtree(refined_cache_dir)
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(anomaly_fullres_bbox_dir, exist_ok=True)
    os.makedirs(refined_cache_dir, exist_ok=True)

    tile_overlap = manifest["tile_overlap"]
    total_imgs = len(manifest["images"])
    score_mode_label = (
        "RAW_UNFILTERED (未經防誤報濾波 + 單點最大值 np.max)"
        if raw_score_mode
        else "FILTERED_ROBUST (防誤報純化濾波 + Top-0.015% 抗噪點峰值)"
    )
    canvas_desc = f"降維畫布 <= {stitch_max_dim}px" if stitch_max_dim > 0 else "原圖全解析度"

    log_and_print(
        f"\n[Step 3.1] 啟動 RAM 多核心平行縫合與碎片銷毀模式 (共 {total_imgs} 張原圖 | "
        f"畫布={canvas_desc} | Workers={num_workers}) | 計分模式: {score_mode_label}",
        log_files,
    )

    tasks = []
    for idx, (fname, meta) in enumerate(manifest["images"].items(), 1):
        tasks.append({
            "idx": idx,
            "fname": fname,
            "meta": meta,
            "tile_amap_dir": tile_amap_dir,
            "tile_overlap": tile_overlap,
            "stitch_max_dim": stitch_max_dim,
            "refined_cache_dir": refined_cache_dir,
            "category": category,
            "border_ratio": border_ratio,
            "wafer_edge_mode": wafer_edge_mode,
            "wafer_edge_margin": wafer_edge_margin,
            "speckle_filter_ratio": speckle_filter_ratio,
            "raw_score_mode": raw_score_mode,
        })

    stitched_info = []
    all_pool_scores = []
    normal_candidate_scores = []

    eff_workers = max(1, min(int(num_workers), total_imgs))
    t_stitch_loop_start = time.perf_counter()

    if eff_workers <= 1:
        for completed, t_item in enumerate(tasks, 1):
            res = _stitch_and_refine_single_worker(t_item)
            all_pool_scores.append(res["max_score"])
            if not os.path.basename(res["filename"]).startswith("escape_"):
                normal_candidate_scores.append(res["max_score"])
            stitched_info.append(res)
            if completed == 1 or completed % 5 == 0 or completed == total_imgs:
                _print_inline_progress(
                    stage_title="Step 3.1 RAM縫合與濾波",
                    current=completed,
                    total=total_imgs,
                    start_t=t_stitch_loop_start,
                    extra_info=f"最新分數={res['max_score']:.4f}",
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=eff_workers) as executor:
            futures = [executor.submit(_stitch_and_refine_single_worker, t_item) for t_item in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                all_pool_scores.append(res["max_score"])
                if not os.path.basename(res["filename"]).startswith("escape_"):
                    normal_candidate_scores.append(res["max_score"])
                stitched_info.append(res)
                completed += 1
                if completed == 1 or completed % 5 == 0 or completed == total_imgs:
                    _print_inline_progress(
                        stage_title=f"Step 3.1 平行縫合濾波(x{eff_workers})",
                        current=completed,
                        total=total_imgs,
                        start_t=t_stitch_loop_start,
                        extra_info=f"最新分數={res['max_score']:.4f}",
                    )

    safe_rmtree(tile_amap_dir)
    log_and_print(
        f"[🧹 碎片清理完成] 所有熱力圖已於 RAM 內完成重建，已自動銷毀全部暫存碎片 .npy -> {os.path.abspath(tile_amap_dir)}",
        log_files,
    )

    ref_normal_scores = normal_candidate_scores if normal_candidate_scores else all_pool_scores
    effective_threshold, thresh_mode_str = compute_dynamic_effective_threshold(
        pool_scores=ref_normal_scores,
        thresh_mode=thresh_mode,
        user_threshold=user_threshold,
        min_threshold=min_threshold,
        auto_percentile=auto_percentile,
        sigma_k=sigma_k,
        top_k=top_k,
    )

    log_and_print(
        f"[Step 3.2] 異常判定門檻設定: {thresh_mode_str}\n"
        f"           (模型範式=MuSc-V2 Training-Free Zero-Shot | 縫合畫布={canvas_desc} | "
        f"計分模式={'RAW未濾波' if raw_score_mode else '防誤報純化濾波'}, "
        f"圓周邊緣抑制={'OFF(Raw)' if raw_score_mode else wafer_edge_mode}, "
        f"邊界縮減={0.0 if raw_score_mode else wafer_edge_margin * 100:.1f}%, "
        f"微塵開運算比例={0.0 if raw_score_mode else speckle_filter_ratio:.4f})",
        log_files,
    )

    stitched_info.sort(key=lambda item: item["max_score"], reverse=True)
    ranking_records = []
    anomaly_count = 0
    skipped_normal_count = 0

    for rank_idx, item in enumerate(stitched_info, 1):
        fname = item["filename"]
        stem = item["stem"]
        meta = item["meta"]
        cache_npy_path = item["cache_npy_path"]
        max_score = item["max_score"]
        raw_max_score = item["raw_max_score"]
        filtered_max_score = item["filtered_max_score"]
        filter_delta = item["filter_delta"]
        top_k_score = item["top_k_score"]
        h, w = meta["height"], meta["width"]

        if max_score < effective_threshold:
            if cache_npy_path and os.path.exists(cache_npy_path):
                try:
                    os.remove(cache_npy_path)
                except OSError:
                    pass
            skipped_normal_count += 1
            ranking_records.append({
                "rank": rank_idx,
                "filename": fname,
                "status": "PASS_SKIPPED",
                "max_score": max_score,
                "raw_max_score": raw_max_score,
                "filtered_max_score": filtered_max_score,
                "filter_delta": filter_delta,
                "top_k_score": top_k_score,
                "num_bboxes": 0,
                "saved_file": "(未超過動態門檻，不輸出圖片)",
                "resolution": f"{w}x{h}",
            })
            continue

        if cache_npy_path and os.path.exists(cache_npy_path):
            active_amap = np.load(cache_npy_path).astype(np.float32)
            try:
                os.remove(cache_npy_path)
            except OSError:
                pass
        else:
            sh, sw, _, _ = compute_stitch_canvas_shape(h, w, stitch_max_dim)
            active_amap = np.zeros((sh, sw), dtype=np.float32)

        amap_h, amap_w = active_amap.shape[:2]
        bboxes_scaled, _ = extract_tight_bboxes_for_anomaly_image(
            refined_amap=active_amap,
            img_threshold=effective_threshold,
            core_ratio=core_ratio,
            min_area_ratio=min_area_ratio,
            raw_score_mode=raw_score_mode,
        )

        if not bboxes_scaled:
            del active_amap
            skipped_normal_count += 1
            ranking_records.append({
                "rank": rank_idx,
                "filename": fname,
                "status": "PASS_NOISE_FILTERED",
                "max_score": max_score,
                "raw_max_score": raw_max_score,
                "filtered_max_score": filtered_max_score,
                "filter_delta": filter_delta,
                "top_k_score": top_k_score,
                "num_bboxes": 0,
                "saved_file": "(微小單點噪點已自動濾除，不輸出圖片)",
                "resolution": f"{w}x{h}",
            })
            continue

        # 將降維畫布紅框座標映回 1:1 原圖解析度
        inv_sx = w / float(max(1, amap_w))
        inv_sy = h / float(max(1, amap_h))
        bboxes_full = []
        for (x1, y1, x2, y2, region_max, area_s) in bboxes_scaled:
            fx1 = max(0, min(w - 1, int(round(x1 * inv_sx))))
            fy1 = max(0, min(h - 1, int(round(y1 * inv_sy))))
            fx2 = max(0, min(w - 1, int(round(x2 * inv_sx))))
            fy2 = max(0, min(h - 1, int(round(y2 * inv_sy))))
            farea = int(round(area_s * inv_sx * inv_sy))
            bboxes_full.append((fx1, fy1, fx2, fy2, region_max, farea))

        anomaly_count += 1
        save_dtype = np.float16 if max(active_amap.shape) > 1024 else np.float32
        np.save(
            os.path.join(final_amap_dir, f"Rank{rank_idx:03d}_Score{max_score:.4f}_{stem}.npy"),
            active_amap.astype(save_dtype),
        )

        orig_bgr = load_image_bgr_uint8(meta["path"])
        vis_scale = max(1.0, max(h, w) / 900.0)
        box_thickness = max(2, int(round(2.5 * vis_scale)))
        font_scale = max(0.5, 0.55 * vis_scale)
        font_thickness = max(1, int(round(1.5 * vis_scale)))

        bbox_canvas = orig_bgr.copy()
        for (x1, y1, x2, y2, region_max, _) in bboxes_full:
            cv2.rectangle(bbox_canvas, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            label_txt = f"{region_max:.3f}"
            (tw, th_txt), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            ty1 = max(0, y1 - th_txt - 8)
            cv2.rectangle(bbox_canvas, (x1, ty1), (x1 + tw + 6, ty1 + th_txt + 6), (0, 0, 255), -1)
            cv2.putText(
                bbox_canvas, label_txt, (x1 + 3, ty1 + th_txt + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

        safe_out_fname = re.sub(r"[\\/]+", "__", fname)
        prefixed_img_name = f"Rank{rank_idx:03d}_Score{max_score:.4f}_Box{len(bboxes_full):02d}_{safe_out_fname}"

        fullres_out_path = os.path.join(anomaly_fullres_bbox_dir, prefixed_img_name)
        cv2.imwrite(fullres_out_path, bbox_canvas)

        # 三合一對照圖直接於 panel_h 尺度合成熱力圖疊圖
        panel_h = min(900, h)
        panel_w = int(round(w * (panel_h / float(h))))
        p1 = cv2.resize(orig_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(bbox_canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        amap_panel = cv2.resize(active_amap, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        img_med = float(np.median(amap_panel))
        norm_map = np.clip((amap_panel - img_med) / max(float(np.max(amap_panel)) - img_med, 1e-6), 0.0, 1.0)
        hm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        p2 = cv2.addWeighted(p1, 0.55, hm_color, 0.45, 0)
        tri_panel = np.hstack([p1, p2, p3])

        banner_h = max(36, int(round(panel_h * 0.055)))
        banner = np.zeros((banner_h, tri_panel.shape[1], 3), dtype=np.uint8)
        mode_tag = "MuScV2-RAW" if raw_score_mode else "MuScV2-FILT"
        banner_txt = (
            f"[{mode_tag}] Orig: {fname} | Score: {max_score:.4f} (Raw: {raw_max_score:.4f}, Filt: {filtered_max_score:.4f}, "
            f"EffThresh: {effective_threshold:.4f}) | Status: ANOMALY (Boxes: {len(bboxes_full)})"
        )
        cv2.putText(
            banner, banner_txt, (12, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, banner_h / 55.0), (0, 220, 255), 2, cv2.LINE_AA
        )
        final_vis = np.vstack([banner, tri_panel])
        vis_out_path = os.path.join(anomaly_dir, prefixed_img_name)
        cv2.imwrite(vis_out_path, final_vis)

        del active_amap, orig_bgr, bbox_canvas, amap_panel, hm_color, p1, p2, p3, tri_panel, final_vis
        gc.collect()

        ranking_records.append({
            "rank": rank_idx,
            "filename": fname,
            "status": "ANOMALY",
            "max_score": max_score,
            "raw_max_score": raw_max_score,
            "filtered_max_score": filtered_max_score,
            "filter_delta": filter_delta,
            "top_k_score": top_k_score,
            "num_bboxes": len(bboxes_full),
            "saved_file": prefixed_img_name,
            "resolution": f"{w}x{h}",
        })

    safe_rmtree(refined_cache_dir)
    gc.collect()

    normal_recs = [r for r in ranking_records if not os.path.basename(r["filename"]).startswith("escape_")]
    target_recs = normal_recs if normal_recs else ranking_records

    dist_scores = [float(r["max_score"]) for r in target_recs]
    raw_dist_scores = [float(r["raw_max_score"]) for r in target_recs]
    filt_dist_scores = [float(r["filtered_max_score"]) for r in target_recs]

    max_s = float(np.max(dist_scores))
    p99_s = float(np.percentile(dist_scores, 99))
    p95_s = float(np.percentile(dist_scores, 95))
    median_s = float(np.median(dist_scores))
    mean_s = float(np.mean(dist_scores))
    std_s = float(np.std(dist_scores))
    m1std = mean_s + 1.0 * std_s
    m2std = mean_s + 2.0 * std_s
    m3std = mean_s + 3.0 * std_s

    raw_max_s = float(np.max(raw_dist_scores))
    raw_mean_s = float(np.mean(raw_dist_scores))
    raw_std_s = float(np.std(raw_dist_scores))
    filt_max_s = float(np.max(filt_dist_scores))
    filt_mean_s = float(np.mean(filt_dist_scores))
    filt_std_s = float(np.std(filt_dist_scores))

    eval_metrics = compute_blind_pool_eval_metrics(ranking_records)
    metrics_summary_line = ""
    metrics_report_header = ""
    if eval_metrics is not None:
        metrics_summary_line = (
            f"      🏆 模擬盲池量化指標 (Eval)     : "
            f"I-AUROC={eval_metrics['auroc']:.4f} | I-AP={eval_metrics['ap']:.4f} | "
            f"F1-max={eval_metrics['f1_max']:.4f} (@{eval_metrics['best_thresh']:.4f})\n"
            f"      🎯 當前門檻實測表現 (Current)    : "
            f"Recall={eval_metrics['recall']*100:.2f}% ({eval_metrics['tp']}/{eval_metrics['n_pos']}) | "
            f"FPR={eval_metrics['fpr']*100:.2f}% ({eval_metrics['fp']}/{eval_metrics['n_neg']}) | "
            f"Precision={eval_metrics['precision']*100:.2f}% | F1={eval_metrics['f1_cur']*100:.2f}%\n"
        )
        metrics_report_header = (
            f"Benchmark Metrics (No-GT Sim) : I-AUROC={eval_metrics['auroc']:.4f} | I-AP={eval_metrics['ap']:.4f} | "
            f"F1-max={eval_metrics['f1_max']:.4f} (Best_Th={eval_metrics['best_thresh']:.4f})\n"
            f"Current Threshold Performance : Recall={eval_metrics['recall']*100:.2f}% ({eval_metrics['tp']}/{eval_metrics['n_pos']}) | "
            f"FPR={eval_metrics['fpr']*100:.2f}% ({eval_metrics['fp']}/{eval_metrics['n_neg']}) | "
            f"Precision={eval_metrics['precision']*100:.2f}% | F1={eval_metrics['f1_cur']*100:.2f}%\n"
        )

    dist_summary_lines = (
        f"{metrics_summary_line}"
        f"      📊 正常圖分數分佈 ({'RAW未濾波' if raw_score_mode else '防誤報濾波後'}) : "
        f"Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
        f"      📐 正常圖門檻參考值              : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
        f"      🔬 濾波前後正常圖對照 (Ablation) : "
        f"[未濾波 Raw] Max={raw_max_s:.4f}, Mean={raw_mean_s:.4f}, Std={raw_std_s:.4f} vs "
        f"[防誤報濾波] Max={filt_max_s:.4f}, Mean={filt_mean_s:.4f}, Std={filt_std_s:.4f}\n"
        f"      🩺 模型過擬合自檢 (Overfit)      : 🟢 Zero-Shot Training-Free (無參數訓練，天生 0 過擬合風險)"
    )

    report_txt = os.path.join(out_dir, "escape_ranking_report.txt")
    report_csv = os.path.join(out_dir, "escape_ranking_report.csv")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=== MuSc-V2 High-Res Zero-Shot Anomaly Detection Report (Training-Free) ===\n")
        f.write(
            f"Score Mode: {'RAW_UNFILTERED' if raw_score_mode else 'FILTERED_ROBUST'} | "
            f"Threshold Mode: {thresh_mode} | Effective Threshold: {effective_threshold:.4f} | "
            f"Stitch Canvas: {canvas_desc} | "
            f"Wafer Edge Mode: {'off(raw)' if raw_score_mode else wafer_edge_mode} (Margin: {wafer_edge_margin:.3f})\n"
            f"Core Box Ratio: {core_ratio:.2f} | Exported Anomaly Images: {anomaly_count} | "
            f"Skipped Normal Images: {skipped_normal_count}\n"
            f"{metrics_report_header}"
            f"Non-Golden Normal Score Dist: Max={max_s:.4f} | P99={p99_s:.4f} | P95={p95_s:.4f} | "
            f"Median={median_s:.4f} | Mean={mean_s:.4f} | Std={std_s:.4f}\n"
            f"Ablation Compare (Normal)   : Raw_Max={raw_max_s:.4f}, Raw_Mean={raw_mean_s:.4f} vs "
            f"Filtered_Max={filt_max_s:.4f}, Filtered_Mean={filt_mean_s:.4f}\n"
            f"Threshold Ref (Mean+k*Std)  : mean+1*std={m1std:.4f} | mean+2*std={m2std:.4f} | mean+3*std={m3std:.4f}\n"
            f"Overfit Diagnostic          : 🟢 Training-Free Zero-Shot (No Training Overfit)\n"
        )
        f.write("-" * 145 + "\n")
        f.write(
            f"{'Rank':<5} | {'Original_Filename':<26} | {'Status':<19} | {'Max_Score':<9} | "
            f"{'Raw_Score':<9} | {'Filt_Score':<10} | {'Delta':<8} | {'BBoxes':<6} | {'Saved_Anomaly_Filename'}\n"
        )
        f.write("-" * 145 + "\n")
        for r in ranking_records:
            f.write(
                f"{r['rank']:<5} | {r['filename']:<26} | {r['status']:<19} | "
                f"{r['max_score']:<9.4f} | {r['raw_max_score']:<9.4f} | {r['filtered_max_score']:<10.4f} | "
                f"{r['filter_delta']:<+8.4f} | {r['num_bboxes']:<6} | {r['saved_file']}\n"
            )

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Original_Filename", "Status", "Max_Score",
            "Raw_Max_Score", "Filtered_Max_Score", "Filter_Delta",
            "TopK_Mean_Score", "Num_BBoxes", "Saved_Anomaly_Filename", "Resolution"
        ])
        for r in ranking_records:
            writer.writerow([
                r["rank"], r["filename"], r["status"],
                f"{r['max_score']:.6f}",
                f"{r['raw_max_score']:.6f}",
                f"{r['filtered_max_score']:.6f}",
                f"{r['filter_delta']:+.6f}",
                f"{r['top_k_score']:.6f}",
                r["num_bboxes"], r["saved_file"], r["resolution"],
            ])

    log_and_print(
        f"[Step 3.3] 篩選與畫框完成：共 {anomaly_count} 張超過有效門檻 ({effective_threshold:.4f}) 已畫框輸出；"
        f"其餘 {skipped_normal_count} 張正常圖已自動略過不輸出。",
        log_files,
    )
    return anomaly_count, skipped_normal_count, effective_threshold, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines


def main():
    default_workers = min(8, os.cpu_count() or 4)
    parser = argparse.ArgumentParser(description="MuSc-V2 High-Res Zero-Shot Anomaly Inference Pipeline (無損加速版)")
    parser.add_argument("--pool_dir", type=str, required=True, help="待測無 GT 盲池資料夾路徑")
    parser.add_argument("--ckpt", type=str, default="auto", help="本機離線 DINOv2/DINOv3 權重路徑 (預設 auto 自動搜尋)")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14", help="視覺骨幹名稱")
    parser.add_argument("--out_dir", type=str, default="./output/muscv2_no_gt")
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--data_name", type=str, default="muscv2_blind_pool")
    parser.add_argument("--category", type=str, default="auto")

    # MuSc-V2 核心互評超參數
    parser.add_argument("--r_list", type=str, default="1,3,5", help="MuSc-V2 局部鄰域聚合尺度清單 (例如 1,3,5)")
    parser.add_argument("--feature_layers", type=str, default="", help="指定提取的 Transformer 層數 (留空自動依骨幹深度配置)")
    parser.add_argument("--mutual_ratio", type=float, default=0.30, help="MuSc-V2 互評機制取最低距離的比例 (預設 0.30 即 30%%)")
    parser.add_argument("--rscon_weight", type=float, default=0.15, help="MuSc-V2 RsCon 全域鄰域重評校準權重 (0 則關閉)")
    parser.add_argument("--max_peer_tiles", type=int, default=400, help="大盲池互評時每個 Query 最多抽樣比較的他片圖塊數")

    # 門檻與動態對齊參數
    parser.add_argument(
        "--thresh_mode", type=str, default="auto",
        choices=["auto", "hybrid", "mean_std", "percentile", "top_k", "fixed"],
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min_threshold", type=float, default=0.0)
    parser.add_argument("--auto_percentile", type=float, default=95.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--sigma_k", type=float, default=3.5)

    # 防誤報純化與智慧切塊防呆參數
    parser.add_argument("--raw_score_mode", action="store_true")
    parser.add_argument("--core_ratio", type=float, default=0.55)
    parser.add_argument("--border_ratio", type=float, default=0.02)
    parser.add_argument("--wafer_edge_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--wafer_edge_margin", type=float, default=0.04)
    parser.add_argument("--speckle_filter_ratio", type=float, default=0.008)
    parser.add_argument("--min_area_ratio", type=float, default=0.00012)
    parser.add_argument("--force_infer", action="store_true")
    parser.add_argument("--mock_from_mvtec", type=str, default="")
    parser.add_argument("--min_tiling_res", type=int, default=1536)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--bg_std_thresh", type=float, default=1.5)
    parser.add_argument("--disable_tiling", action="store_true")

    # 無損加速參數
    parser.add_argument("--infer_batch_size", type=int, default=64, help="提取特徵時的 Batch Size (預設由 16 提升至 64)")
    parser.add_argument("--num_workers", type=int, default=default_workers, help="切塊、讀圖與縫合之多核心 Worker 數")
    parser.add_argument("--stitch_max_dim", type=int, default=1024, help="縫合與濾波畫布最大邊長 (預設 1024，設 0 則退回原圖解析度)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "off"], help="GPU 混合精度設定")
    args = parser.parse_args()

    if not args.category or args.category.lower() == "auto":
        if args.mock_from_mvtec:
            args.category = os.path.basename(os.path.normpath(args.mock_from_mvtec))
        else:
            args.category = os.path.basename(os.path.normpath(args.pool_dir))

    r_list = [int(x.strip()) for x in args.r_list.split(",") if x.strip()] or [1, 3, 5]
    custom_layers = [int(x.strip()) for x in args.feature_layers.split(",") if x.strip()]

    now_tpe = datetime.now(TZ_TAIPEI)
    timestamp = now_tpe.strftime("%Y%m%d_%H%M%S")
    base_out_dir = os.path.abspath(args.out_dir)
    run_out_dir = os.path.join(base_out_dir, timestamp)
    os.makedirs(run_out_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)
    abs_datasets_dir = os.path.abspath(args.datasets_dir)

    fixed_log_txt = os.path.join(run_out_dir, "log.txt")
    parent_latest_log = os.path.join(base_out_dir, "latest_log.txt")
    log_files = [fixed_log_txt, parent_latest_log]

    pipeline_start_t = time.perf_counter()
    start_dt_str = now_tpe.strftime("%Y-%m-%d %H:%M:%S (UTC+8)")
    log_and_print(
        f"\n{'=' * 75}\n"
        f"[MuSc-V2 Zero-Shot 無GT抓漏純推論啟動時間] {start_dt_str}\n"
        f"[本次專屬時間戳記輸出目錄] {run_out_dir}\n"
        f"[無損加速設定] BatchSize={args.infer_batch_size} | Workers={args.num_workers} | "
        f"StitchCanvas={args.stitch_max_dim}px | AMP={args.amp_dtype}\n"
        f"{'=' * 75}",
        log_files,
    )

    if args.mock_from_mvtec:
        existing_pool_imgs = scan_pool_images(args.pool_dir) if os.path.isdir(args.pool_dir) else []
        if existing_pool_imgs and not args.force_infer:
            log_and_print(
                f"[⚡ 模擬無GT盲池快取命中] 直接沿用既有 {len(existing_pool_imgs)} 張盲池圖片 "
                f"-> {os.path.abspath(args.pool_dir)}",
                log_files,
            )
        else:
            prepare_mock_unlabeled_pool_from_mvtec(args.mock_from_mvtec, args.pool_dir, log_files, defect_limit=8)

    pool_signature = compute_pool_signature(args.pool_dir)
    tile_config_key = (
        f"mtr{args.min_tiling_res}_t{args.tile_size}_ov{args.tile_overlap}_bg{args.bg_std_thresh:g}_dt{int(args.disable_tiling)}"
    )

    amap_dump_dir = os.path.join(run_out_dir, "amap_npy")
    report_txt_path = os.path.join(run_out_dir, "escape_ranking_report.txt")
    report_csv_path = os.path.join(run_out_dir, "escape_ranking_report.csv")
    tiled_pool_dir = os.path.join(abs_datasets_dir, f"{args.data_name}_tiled_pool")
    cached_manifest_path = os.path.join(abs_datasets_dir, f"{args.data_name}_cached_tile_manifest.json")
    tile_amap_tmp_dir = os.path.join(run_out_dir, "_tmp_tile_amap_npy")

    test_tile_sec = 0.0
    used_png_tile_cache = False
    manifest = None

    if not args.force_infer:
        manifest = is_tiled_png_pool_valid(
            tiled_pool_dir=tiled_pool_dir,
            cached_manifest_path=cached_manifest_path,
            pool_signature=pool_signature,
            tile_config_key=tile_config_key,
        )

    if manifest is not None:
        used_png_tile_cache = True
        safe_copy(cached_manifest_path, os.path.join(run_out_dir, "tile_manifest.json"))
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)
        cache_mode_note = (
            f"切塊子圖塊 PNG ({total_test_tiles} 塊)"
            if manifest.get("large_img_count", 0) > 0
            else f"整張原圖直接推論 ({num_test_images} 張，防呆跳過切塊)"
        )
        log_and_print(
            f"\n[⚡ 測試池清單快取命中] 測試圖片池與解析度參數未變動，直接沿用既有 {cache_mode_note} (0.00 秒)",
            log_files,
        )
    else:
        t_test_tile_start = time.perf_counter()
        manifest = prepare_tiled_test_pool(
            pool_dir=args.pool_dir,
            tiled_pool_dir=tiled_pool_dir,
            manifest_save_dir=run_out_dir,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            min_tiling_res=args.min_tiling_res,
            bg_std_thresh=args.bg_std_thresh,
            disable_tiling=args.disable_tiling,
            pool_signature=pool_signature,
            tile_config_key=tile_config_key,
            cached_manifest_path=cached_manifest_path,
            log_files=log_files,
            wafer_edge_mode=args.wafer_edge_mode,
            category=args.category,
            num_workers=args.num_workers,
        )
        test_tile_sec = time.perf_counter() - t_test_tile_start
        num_test_images = manifest.get("num_images", 1)
        total_test_tiles = manifest.get("total_tiles", num_test_images)

    safe_rmtree(tile_amap_tmp_dir)
    os.makedirs(tile_amap_tmp_dir, exist_ok=True)

    t_infer_start = time.perf_counter()
    run_muscv2_zero_shot_inference(
        manifest=manifest,
        tiled_pool_dir=tiled_pool_dir,
        tile_amap_dir=tile_amap_tmp_dir,
        backbone_name=args.backbone,
        ckpt_path=args.ckpt,
        r_list=r_list,
        custom_layers=custom_layers,
        mutual_ratio=args.mutual_ratio,
        rscon_weight=args.rscon_weight,
        infer_batch_size=args.infer_batch_size,
        max_peer_tiles=args.max_peer_tiles,
        log_files=log_files,
        num_workers=args.num_workers,
        amp_dtype_str=args.amp_dtype,
    )
    model_infer_sec = time.perf_counter() - t_infer_start

    t_stitch_start = time.perf_counter()
    anomaly_count, skipped_count, eff_thresh, anomaly_dir, anomaly_fullres_bbox_dir, dist_summary_lines = stitch_and_filter_anomalies(
        manifest=manifest,
        tile_amap_dir=tile_amap_tmp_dir,
        final_amap_dir=amap_dump_dir,
        out_dir=run_out_dir,
        category=args.category,
        thresh_mode=args.thresh_mode,
        user_threshold=args.threshold,
        min_threshold=args.min_threshold,
        auto_percentile=args.auto_percentile,
        top_k=args.top_k,
        sigma_k=args.sigma_k,
        core_ratio=args.core_ratio,
        border_ratio=args.border_ratio,
        wafer_edge_mode=args.wafer_edge_mode,
        wafer_edge_margin=args.wafer_edge_margin,
        speckle_filter_ratio=args.speckle_filter_ratio,
        min_area_ratio=args.min_area_ratio,
        raw_score_mode=args.raw_score_mode,
        log_files=log_files,
        stitch_max_dim=args.stitch_max_dim,
        num_workers=args.num_workers,
    )
    stitch_vis_sec = time.perf_counter() - t_stitch_start

    total_infer_time = test_tile_sec + model_infer_sec + stitch_vis_sec
    avg_infer_per_img = total_infer_time / max(1, num_test_images)
    total_wall_time = time.perf_counter() - pipeline_start_t

    step1_label = (
        f"Step 1 大圖高解析度切塊 ({total_test_tiles} 塊)"
        if manifest.get("large_img_count", 0) > 0
        else f"Step 1 解析度防呆檢查 ({num_test_images} 張全圖)"
    )

    infer_done_summary = (
        f"\n{'=' * 75}\n"
        f" [✅ MuSc-V2 推論完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-' * 75}\n"
        f"  [1] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ {step1_label:<38} : {format_duration(test_tile_sec)}"
        f"{' (Cached)' if used_png_tile_cache else ''}\n"
        f"      ├─ Step 2 MuSc-V2 Zero-Shot 互評推理 (BS={args.infer_batch_size}) : {format_duration(model_infer_sec)}\n"
        f"      ├─ Step 3 RAM 縫合銷毀碎片與{'未濾波' if args.raw_score_mode else '防誤報濾波'}畫框 (Canvas<={args.stitch_max_dim}px) : {format_duration(stitch_vis_sec)}\n"
        f"      └─ 平均每張原圖推論耗時 ({num_test_images} 張原圖)         : {avg_infer_per_img:.2f} 秒/張\n"
        f"{'-' * 75}\n"
        f"  [2] 任務全程總執行時間 (Total Wall Time)      : {format_duration(total_wall_time)}\n"
        f"{'-' * 75}\n"
        f"  [3] 門檻篩選統計與分數分佈 (Score={'RAW' if args.raw_score_mode else 'FILTERED'}, Mode={args.thresh_mode}, Eff_Thresh={eff_thresh:.4f}):\n"
        f"      🚨 超過門檻判定異常並畫框輸出 : {anomaly_count} 張\n"
        f"      ✅ 未達門檻或微塵濾除直接略過 : {skipped_count} 張 (不輸出圖片)\n"
        f"{dist_summary_lines}\n"
        f"{'-' * 75}\n"
        f"  [4] 本次推論結果儲存資料夾清單 (Saved Directories & Files):\n"
        f"      🧹 子圖塊碎片 Tile .npy 狀態  : 已於 RAM 拼接完成後立即自動銷毀 (0 殘留)\n"
        f"      📂 本次時間戳記推論主資料夾   : {run_out_dir}\n"
        f"          ├── 異常紅框三合一對照圖  : {anomaly_dir}\n"
        f"          ├── 異常1:1全解析度紅框圖 : {anomaly_fullres_bbox_dir}\n"
        f"          ├── 異常熱力圖矩陣(.npy)  : {amap_dump_dir}\n"
        f"          ├── 完整排名報表 (TXT)    : {report_txt_path}\n"
        f"          ├── 完整排名報表 (CSV)    : {report_csv_path}\n"
        f"          └── 本次執行日誌 (log.txt): {fixed_log_txt}\n"
        f"{'=' * 75}"
    )

    log_and_print(infer_done_summary, log_files)
    if os.path.exists(report_txt_path):
        with open(report_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + infer_done_summary + "\n")


if __name__ == "__main__":
    main()
