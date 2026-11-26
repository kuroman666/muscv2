# 檔案位置: MuSc-V2/1_infer_muscv2_no_gt.py
# 用途: MuSc-V2 Training-Free & Zero-Shot 無損加速版
# (Step 0: 以第一張圖為準自動對齊旋轉方向；
#  Step 1: 圓形預遮罩 + 平行切塊 + 圓心與旋轉角快取；
#  Step 2: Position-wise 同座標 SNAMD SWPooling + MSM + 全晶圓級 RsCon k=7 WMO；
#  Step 3: 1024px 降維縫合濾波 + 熱力圖逆向轉回原圖方向 + 1:1 原圖畫框)
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
    """讀取 Linux 可用主記憶體，並自動偵測 Docker / cgroup 容器記憶體上限。"""
    host_avail = 32 * (1024 ** 3)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        host_avail = int(parts[1]) * 1024
                        break
    except Exception:
        pass

    cgroup_avail = host_avail
    # 檢查 cgroup v2 (/sys/fs/cgroup/memory.max)
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as f:
            max_str = f.read().strip()
        if max_str.isdigit():
            limit = int(max_str)
            with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as f:
                cur = int(f.read().strip())
            cgroup_avail = max(0, limit - cur)
    except Exception:
        # 退回檢查 cgroup v1
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r", encoding="utf-8") as f:
                limit = int(f.read().strip())
            if limit < (1 << 60):
                with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r", encoding="utf-8") as f:
                    cur = int(f.read().strip())
                cgroup_avail = max(0, limit - cur)
        except Exception:
            pass

    return min(host_avail, cgroup_avail)


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
# Step 0: 影像方向自動對齊模組 (以資料集第一張圖為準，並支援 Step 3 逆向轉回)
# =====================================================================

def _build_align_template(img_rgb: np.ndarray, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """將影像降採樣為 (size x size) 灰階並建立內接圓遮罩，用於穩健計算旋轉對齊角。"""
    gray = cv2.cvtColor(cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray_f = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) * 0.5
    r = size * 0.46
    circle_mask = ((yy - c) ** 2 + (xx - c) ** 2) <= (r ** 2)
    vals = gray_f[circle_mask]
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals)) + 1e-6
    norm_gray = np.zeros_like(gray_f, dtype=np.float32)
    norm_gray[circle_mask] = (vals - mean_v) / std_v
    return norm_gray, circle_mask


def estimate_rotation_angle_to_ref(
    img_rgb: np.ndarray,
    ref_norm_gray: np.ndarray,
    circle_mask: np.ndarray,
    align_mode: str = "full360",
) -> float:
    """估測將 img_rgb 旋轉對齊至參考圖所需的逆時針角度 (度)。"""
    mode = align_mode.lower().strip()
    if mode == "off":
        return 0.0

    size = ref_norm_gray.shape[0]
    cur_norm, _ = _build_align_template(img_rgb, size=size)
    center = (size * 0.5, size * 0.5)
    ref_vec = ref_norm_gray[circle_mask]

    def _score_angle(deg: float) -> float:
        if abs(deg) < 1e-6:
            rot = cur_norm
        else:
            mat = cv2.getRotationMatrix2D(center, deg, 1.0)
            rot = cv2.warpAffine(cur_norm, mat, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return float(np.dot(rot[circle_mask], ref_vec))

    if mode == "discrete90":
        candidates = [0.0, 90.0, 180.0, 270.0]
        best_ang = max(candidates, key=_score_angle)
        return float(best_ang)

    # full360: 粗搜 (每 5 度，含 0/90/180/270) + 精搜 (±5 度內每 0.5 度)
    coarse_angles = [float(a) for a in range(0, 360, 5)]
    best_coarse = max(coarse_angles, key=_score_angle)

    fine_angles = [best_coarse + 0.5 * k for k in range(-10, 11)]
    best_fine = max(fine_angles, key=_score_angle) % 360.0

    # 若接近 0 度 (±0.5° 內) 則直接歸零，避免無意義的雙線性插值
    if best_fine <= 0.5 or best_fine >= 359.5:
        return 0.0
    return float(best_fine)


def rotate_image_or_amap(
    arr: np.ndarray,
    angle_deg: float,
    border_value=0,
    is_amap: bool = False,
) -> np.ndarray:
    """以影像中心旋轉圖片或熱力圖 (正值為逆時針，負值為順時針逆轉回原圖)。"""
    if abs(angle_deg) < 1e-3:
        return arr
    h, w = arr.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(angle_deg), 1.0)
    interp = cv2.INTER_LINEAR
    if is_amap:
        return cv2.warpAffine(
            arr.astype(np.float32), mat, (w, h),
            flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value),
        )
    bv = (int(border_value), int(border_value), int(border_value)) if arr.ndim == 3 else int(border_value)
    return cv2.warpAffine(
        arr, mat, (w, h),
        flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=bv,
    )

def rotate_point_xy(cx: float, cy: float, w: int, h: int, angle_deg: float) -> Tuple[float, float]:
    """將原圖圓心 (cx, cy) 依相同旋轉矩陣映射至旋轉後的新座標。"""
    if abs(angle_deg) < 1e-3:
        return float(cx), float(cy)
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(angle_deg), 1.0)
    nx = mat[0, 0] * cx + mat[0, 1] * cy + mat[0, 2]
    ny = mat[1, 0] * cx + mat[1, 1] * cy + mat[1, 2]
    return float(nx), float(ny)

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


def get_wafer_geometry(
    full_rgb: np.ndarray,
    wafer_edge_mode: str,
    category: str,
    manual_cx: float = -1.0,
    manual_cy: float = -1.0,
    manual_r: float = -1.0,
) -> Tuple[bool, float, float, float]:
    """整合自動偵測與手動指定圓心/半徑 (輸入 0~1 為比例，>1 為像素，<=0 則用自動偵測)。"""
    h, w = full_rgb.shape[:2]
    has_manual = (manual_cx > 0) or (manual_cy > 0) or (manual_r > 0)

    if manual_cx > 0 and manual_cy > 0 and manual_r > 0:
        cx = manual_cx * w if manual_cx <= 1.0 else manual_cx
        cy = manual_cy * h if manual_cy <= 1.0 else manual_cy
        r = manual_r * min(h, w) if manual_r <= 1.0 else manual_r
        return True, float(cy), float(cx), float(r)

    use_circle, cy, cx, r = detect_circular_wafer_geometry_from_rgb(
        full_rgb, wafer_edge_mode=wafer_edge_mode, category=category
    )
    if has_manual:
        use_circle = True
        if manual_cx > 0:
            cx = manual_cx * w if manual_cx <= 1.0 else manual_cx
        if manual_cy > 0:
            cy = manual_cy * h if manual_cy <= 1.0 else manual_cy
        if manual_r > 0:
            r = manual_r * min(h, w) if manual_r <= 1.0 else manual_r
    return use_circle, float(cy), float(cx), float(r)


def apply_circular_pre_mask(
    full_rgb: np.ndarray,
    wafer_edge_mode: str,
    category: str,
    pre_mask_margin: float = 0.0,
    fill_value: int = 0,
    cached_geom: Optional[Tuple[bool, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[bool, float, float, float], float]:
    """在推論前將圓形晶圓外的背景塗掉，回傳 (遮罩後RGB, 幾何資訊, 實際遮罩半徑)。"""
    h, w = full_rgb.shape[:2]
    if cached_geom is not None:
        use_circle, cy, cx, radius = cached_geom
    else:
        use_circle, cy, cx, radius = detect_circular_wafer_geometry_from_rgb(
            full_rgb, wafer_edge_mode=wafer_edge_mode, category=category
        )

    mask_radius = max(1.0, radius * (1.0 - pre_mask_margin))
    if not use_circle:
        return full_rgb, (use_circle, cy, cx, radius), mask_radius

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(mask_radius)), 255, -1)

    masked_rgb = full_rgb.copy()
    masked_rgb[mask == 0] = fill_value
    return masked_rgb, (use_circle, cy, cx, radius), mask_radius


def run_mask_preview(
    pool_dir: str,
    out_dir: str,
    num_samples: int,
    wafer_edge_mode: str,
    category: str,
    pre_mask_margin: float,
    fill_value: int,
    manual_cx: float,
    manual_cy: float,
    manual_r: float,
    log_files: List[str],
):
    """抽樣 N 張圖產出圓形 Mask 預覽對照圖，並輸出可直接貼回 run_inside_muscv2.sh 的參數建議。"""
    img_paths = scan_pool_images(pool_dir)
    if not img_paths:
        raise FileNotFoundError(f"在 {pool_dir} 找不到任何圖片！")

    indices = np.linspace(0, len(img_paths) - 1, min(num_samples, len(img_paths)), dtype=int)
    preview_dir = os.path.join(out_dir, "mask_preview")
    os.makedirs(preview_dir, exist_ok=True)

    log_and_print(
        f"\n[🔍 Mask 預覽模式] 抽樣 {len(indices)}/{len(img_paths)} 張 | "
        f"wafer_edge_mode={wafer_edge_mode} | pre_mask_margin={pre_mask_margin:g} | fill={fill_value} | "
        f"manual=(cx={manual_cx:g}, cy={manual_cy:g}, r={manual_r:g})",
        log_files,
    )

    stats_list = []
    for idx in indices:
        fpath = img_paths[idx]
        fname = os.path.basename(fpath)
        rgb = load_image_rgb_uint8(fpath)
        h, w = rgb.shape[:2]

        geom = get_wafer_geometry(
            rgb,
            wafer_edge_mode=wafer_edge_mode,
            category=category,
            manual_cx=manual_cx,
            manual_cy=manual_cy,
            manual_r=manual_r,
        )
        masked_rgb, (use_circle, cy, cx, r_raw), r_mask = apply_circular_pre_mask(
            rgb,
            wafer_edge_mode=wafer_edge_mode,
            category=category,
            pre_mask_margin=pre_mask_margin,
            fill_value=fill_value,
            cached_geom=geom,
        )
        stats_list.append((w, h, cx, cy, r_raw, r_mask, use_circle))

        orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
        thick = max(2, int(round(max(h, w) / 400)))
        if use_circle:
            cv2.circle(orig_bgr, (int(round(cx)), int(round(cy))), int(round(r_raw)), (0, 0, 255), thick)
            cv2.circle(orig_bgr, (int(round(cx)), int(round(cy))), int(round(r_mask)), (0, 255, 0), thick)
            cv2.drawMarker(
                orig_bgr,
                (int(round(cx)), int(round(cy))),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                max(20, int(max(h, w) * 0.02)),
                thick,
            )

        ph = min(800, h)
        pw = int(round(w * (ph / float(h))))
        vis = np.hstack([
            cv2.resize(orig_bgr, (pw, ph), interpolation=cv2.INTER_AREA),
            cv2.resize(masked_bgr, (pw, ph), interpolation=cv2.INTER_AREA),
        ])
        info_txt = (
            f"{fname} ({w}x{h}) | center=({cx:.0f},{cy:.0f}) | "
            f"raw_r={r_raw:.0f} -> mask_r={r_mask:.0f} (ratio={r_mask/min(h,w):.4f})"
        )
        cv2.putText(vis, info_txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(preview_dir, f"preview_{fname}")
        cv2.imwrite(out_path, vis)
        log_and_print(f"  -> {info_txt} | 已存檔: {out_path}", log_files)

    med_w = float(np.median([s[0] for s in stats_list]))
    med_h = float(np.median([s[1] for s in stats_list]))
    med_cx = float(np.median([s[2] for s in stats_list]))
    med_cy = float(np.median([s[3] for s in stats_list]))
    med_r_raw = float(np.median([s[4] for s in stats_list]))
    med_r_mask = float(np.median([s[5] for s in stats_list]))
    min_side = max(1.0, min(med_w, med_h))

    rec_msg = (
        f"\n{'=' * 78}\n"
        f" [✅ Mask 預覽完成] 請開啟資料夾檢視對照圖 (紅圈=原始半徑 / 綠圈=實際遮罩 / 黃十字=圓心):\n"
        f"    📂 {os.path.abspath(preview_dir)}\n"
        f"{'-' * 78}\n"
        f" [📐 抽樣中位數幾何統計 (原圖約 {med_w:.0f}x{med_h:.0f})]\n"
        f"    • 像素值 : cx={med_cx:.1f}, cy={med_cy:.1f}, raw_r={med_r_raw:.1f}, 實際 mask_r={med_r_mask:.1f}\n"
        f"    • 比例值 : cx={med_cx/med_w:.4f}, cy={med_cy/med_h:.4f}, raw_r={med_r_raw/min_side:.4f}, mask_r={med_r_mask/min_side:.4f}\n"
        f"{'=' * 78}"
    )
    log_and_print(rec_msg, log_files)


# =====================================================================
# Step 0 + Step 1: 方向對齊與智慧解析度防呆平行切塊模組
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
    """主記憶體 RAM 防爆保護器 (已適配 Position-wise 同座標分批互評，單批上限為總圖片數 N)。"""
    if disable_tiling:
        return tile_size, tile_overlap

    avail_bytes = get_available_ram_bytes()
    bytes_per_tile = 37 * 37 * 1024 * 2
    max_safe_tiles = max(2000, int((avail_bytes * 0.55) / bytes_per_tile))
    if len(hw_list) <= max_safe_tiles:
        return tile_size, tile_overlap

    log_and_print(
        f"[⚠️ 記憶體提醒] 單一座標圖塊數 ({len(hw_list)}) 超過建議上限 ({max_safe_tiles})，"
        f"推論階段將自動切換為單層串流省記憶體模式。",
        log_files,
    )
    return tile_size, tile_overlap


def _tile_single_image_worker(task_args: Tuple) -> Tuple[str, Dict, int, int, float]:
    (
        img_idx,
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
        pre_mask,
        pre_mask_margin,
        pre_mask_fill,
        manual_cx,
        manual_cy,
        manual_r,
        align_mode,
        ref_norm_gray,
        ref_circle_mask,
    ) = task_args
    cv2.setNumThreads(1)
    abs_fpath = os.path.abspath(fpath)
    should_tile = (not disable_tiling) and (max(h, w) >= min_tiling_res) and (h > tile_size or w > tile_size)
    do_align = (align_mode != "off") and (ref_norm_gray is not None) and (img_idx > 0)

    if not should_tile:
        tile_stem = f"{stem}__full_y0000_x0000_h{h:04d}_w{w:04d}"
        out_tile_path = abs_fpath
        out_tile_fname = os.path.basename(abs_fpath)
        wafer_geom_cache = None
        align_angle = 0.0

        if pre_mask or do_align:
            img = load_image_rgb_uint8(fpath)
            if pre_mask:
                use_circle, cy_f, cx_f, r_f = get_wafer_geometry(
                    img,
                    wafer_edge_mode=wafer_edge_mode,
                    category=category,
                    manual_cx=manual_cx,
                    manual_cy=manual_cy,
                    manual_r=manual_r,
                )
                img, _, _ = apply_circular_pre_mask(
                    img,
                    wafer_edge_mode=wafer_edge_mode,
                    category=category,
                    pre_mask_margin=pre_mask_margin,
                    fill_value=pre_mask_fill,
                    cached_geom=(use_circle, cy_f, cx_f, r_f),
                )
                wafer_geom_cache = [bool(use_circle), float(cy_f), float(cx_f), float(r_f)]
            if do_align:
                align_angle = estimate_rotation_angle_to_ref(img, ref_norm_gray, ref_circle_mask, align_mode)
                if abs(align_angle) > 1e-3:
                    img = rotate_image_or_amap(img, align_angle, border_value=pre_mask_fill, is_amap=False)
                    if wafer_geom_cache is not None:
                        nx, ny = rotate_point_xy(wafer_geom_cache[2], wafer_geom_cache[1], w, h, align_angle)
                        wafer_geom_cache[1], wafer_geom_cache[2] = ny, nx
            if pre_mask or abs(align_angle) > 1e-3:
                out_tile_fname = f"{tile_stem}.png"
                out_tile_path = os.path.abspath(os.path.join(tiled_pool_dir, out_tile_fname))
                Image.fromarray(img).save(out_tile_path, format="PNG", compress_level=0)
            del img

        tiles_info = [{
            "tile_stem": tile_stem,
            "tile_fname": out_tile_fname,
            "tile_abs_path": out_tile_path,
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
            "align_angle": float(align_angle),
            "tiles": tiles_info,
        }
        if wafer_geom_cache is not None:
            entry["wafer_geom"] = wafer_geom_cache
        return rel_fname, entry, 1, 0, float(align_angle)

    img = load_image_rgb_uint8(fpath)

    # 先在未旋轉的原圖上偵測/套用圓形 Mask (100% 對齊 test_mask_muscv2.sh 的 MASK_CX / MASK_CY)
    use_circle, cy_f, cx_f, r_f = get_wafer_geometry(
        img,
        wafer_edge_mode=wafer_edge_mode,
        category=category,
        manual_cx=manual_cx,
        manual_cy=manual_cy,
        manual_r=manual_r,
    )
    if pre_mask:
        img, _, _ = apply_circular_pre_mask(
            img,
            wafer_edge_mode=wafer_edge_mode,
            category=category,
            pre_mask_margin=pre_mask_margin,
            fill_value=pre_mask_fill,
            cached_geom=(use_circle, cy_f, cx_f, r_f),
        )

    # Step 0: 旋轉對齊第一張圖的方向，並同步旋轉圓心座標供 Step 3 邊緣抑制使用
    align_angle = 0.0
    if do_align:
        align_angle = estimate_rotation_angle_to_ref(img, ref_norm_gray, ref_circle_mask, align_mode)
        if abs(align_angle) > 1e-3:
            img = rotate_image_or_amap(img, align_angle, border_value=pre_mask_fill, is_amap=False)
            cx_f, cy_f = rotate_point_xy(cx_f, cy_f, w, h, align_angle)

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
        "align_angle": float(align_angle),
        "wafer_geom": [bool(use_circle), float(cy_f), float(cx_f), float(r_f)],
        "tiles": tiles_info,
    }
    return rel_fname, entry, len(tiles_info), skipped_bg, float(align_angle)


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
    pre_mask: bool = False,
    pre_mask_margin: float = 0.0,
    pre_mask_fill: int = 0,
    manual_cx: float = -1.0,
    manual_cy: float = -1.0,
    manual_r: float = -1.0,
    align_mode: str = "full360",
) -> Dict:
    """Step 0 (對齊第一張方向) + Step 1 (大圖防呆判定與多核心平行切塊)。"""
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

    # Step 0: 以第一張圖建立旋轉對齊參考模板
    ref_norm_gray, ref_circle_mask = None, None
    if align_mode != "off":
        os.makedirs(tiled_pool_dir, exist_ok=True)
        ref_rgb = load_image_rgb_uint8(img_paths[0])
        ref_norm_gray, ref_circle_mask = _build_align_template(ref_rgb, size=256)
        del ref_rgb
        log_and_print(
            f"\n[Step 0 — 方向自動對齊已啟用 ({align_mode})] 基準圖: {os.path.basename(img_paths[0])} "
            f"-> 其餘 {len(img_paths) - 1} 張將先對齊基準圖方向再切塊，最終輸出時自動逆轉回原圖角度！",
            log_files,
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
            f"高解析度重疊切塊 (Workers={num_workers}, 無壓縮 PNG + 圓心與旋轉角快取)！"
        )
    if pre_mask:
        os.makedirs(tiled_pool_dir, exist_ok=True)
        guardrail_msg += (
            f"\n   ↳ [⭕ 推論前圓形 Mask 已啟用] margin={pre_mask_margin:g}, fill={pre_mask_fill}, "
            f"manual=(cx={manual_cx:g}, cy={manual_cy:g}, r={manual_r:g})"
        )
    log_and_print("\n" + guardrail_msg, log_files)

    manifest = {
        "pool_signature": pool_signature,
        "tile_config_key": tile_config_key,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "min_tiling_res": min_tiling_res,
        "large_img_count": large_img_count,
        "align_mode": align_mode,
        "align_ref_image": os.path.basename(img_paths[0]),
        "images": {},
    }

    tasks = []
    for img_idx, (fpath, (h, w)) in enumerate(zip(img_paths, hw_list)):
        rel_fname = os.path.relpath(fpath, pool_dir).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_fname)[0]
        stem = re.sub(r"[\\/]+", "__", rel_no_ext)
        tasks.append((
            img_idx, fpath, h, w, rel_fname, stem, tiled_pool_dir,
            tile_size, stride, min_tiling_res, bg_std_thresh,
            disable_tiling, wafer_edge_mode, category,
            pre_mask, pre_mask_margin, pre_mask_fill,
            manual_cx, manual_cy, manual_r,
            align_mode, ref_norm_gray, ref_circle_mask,
        ))

    total_tiles = 0
    skipped_bg_tiles = 0
    rotated_img_count = 0
    num_pool_imgs = len(tasks)
    eff_workers = max(1, min(int(num_workers), num_pool_imgs)) if (large_img_count > 0 or pre_mask or align_mode != "off") else 1
    t_tile_loop_start = time.perf_counter()

    ordered_results: Dict[str, Dict] = {}
    if eff_workers <= 1:
        for img_idx, t_args in enumerate(tasks, 1):
            rel_fname, entry, n_t, n_skip, ang = _tile_single_image_worker(t_args)
            ordered_results[rel_fname] = entry
            total_tiles += n_t
            skipped_bg_tiles += n_skip
            if abs(ang) > 1e-3:
                rotated_img_count += 1
            if (large_img_count > 0 or align_mode != "off") and (img_idx == 1 or img_idx % 5 == 0 or img_idx == num_pool_imgs):
                _print_inline_progress(
                    stage_title="Step 0+1 對齊與切塊",
                    current=img_idx,
                    total=num_pool_imgs,
                    start_t=t_tile_loop_start,
                    extra_info=f"已旋轉={rotated_img_count} | 圖塊={total_tiles}",
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=eff_workers) as executor:
            futures = [executor.submit(_tile_single_image_worker, t_args) for t_args in tasks]
            for fut in as_completed(futures):
                rel_fname, entry, n_t, n_skip, ang = fut.result()
                ordered_results[rel_fname] = entry
                total_tiles += n_t
                skipped_bg_tiles += n_skip
                if abs(ang) > 1e-3:
                    rotated_img_count += 1
                completed += 1
                if completed == 1 or completed % 5 == 0 or completed == num_pool_imgs:
                    _print_inline_progress(
                        stage_title=f"Step 0+1 平行對齊切塊(x{eff_workers})",
                        current=completed,
                        total=num_pool_imgs,
                        start_t=t_tile_loop_start,
                        extra_info=f"已旋轉={rotated_img_count} | 圖塊={total_tiles}",
                    )

    for t_args in tasks:
        rf = t_args[4]
        if rf in ordered_results:
            manifest["images"][rf] = ordered_results[rf]

    gc.collect()
    manifest["total_tiles"] = total_tiles
    manifest["num_images"] = len(img_paths)
    manifest["rotated_img_count"] = rotated_img_count
    manifest_path = os.path.join(manifest_save_dir, "tile_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if cached_manifest_path:
        os.makedirs(os.path.dirname(cached_manifest_path), exist_ok=True)
        safe_copy(manifest_path, cached_manifest_path)

    log_and_print(
        f"[Step 0+1 完成] 共 {len(img_paths)} 張原圖 (其中 {rotated_img_count} 張已對齊旋轉) -> "
        f"{total_tiles} 張推論圖塊 (略過純背景塊: {skipped_bg_tiles}, Workers={eff_workers})",
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
# Step 2: MuSc-V2 Training-Free & Zero-Shot 互評推理引擎
# (Position-wise 同座標 SNAMD SWPooling + MSM + 全晶圓級 RsCon k=7 WMO)
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


def _swpooling_2d(feat_2d: torch.Tensor, r: int) -> torch.Tensor:
    """MuSc-V2 原論文 SNAMD 的 Similarity-Weighted Pooling (Eq. 5 & 6)。"""
    b, c, h, w = feat_2d.shape
    center = feat_2d.flatten(2).unsqueeze(2)
    nbrs = F.unfold(feat_2d, kernel_size=r, padding=r // 2, stride=1).view(b, c, r * r, h * w)
    dist = torch.norm(nbrs - center, p=2, dim=1, keepdim=True)
    weights = torch.exp(-dist)
    agg = (weights * nbrs).mean(dim=2)
    return agg.transpose(1, 2)


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
    num_layers_to_run: int,
    msm_chunk_size: int,
    device: torch.device,
    patch_scores_accum: torch.Tensor,
    t_total_infer_start: float,
) -> None:
    """針對單一層特徵：先執行原論文 SNAMD (L2正規化 + 多尺度 SWPooling 融合壓縮)，再執行單次 MSM 互評。"""
    bank_dev = device if (device.type == "cuda" and total_tiles <= 1200) else torch.device("cpu")
    bank_host = torch.empty((total_tiles, num_patches, embed_dim), dtype=torch.float16, device=bank_dev)

    prep_bs = 16 if any(r > 1 for r in r_list) else 128
    for b_s in range(0, total_tiles, prep_bs):
        b_e = min(total_tiles, b_s + prep_bs)
        raw_norm = F.normalize(layer_raw_buf[b_s:b_e].to(device).float(), dim=-1)
        sub_2d = raw_norm.transpose(1, 2).reshape(b_e - b_s, embed_dim, grid_hw, grid_hw)

        fused = torch.zeros_like(raw_norm)
        for r in r_list:
            if r == 1:
                fused += raw_norm
            else:
                fused += _swpooling_2d(sub_2d, r=r)
        fused /= float(len(r_list))

        normed = F.normalize(fused, dim=-1).half()
        bank_host[b_s:b_e].copy_(normed)
        del raw_norm, sub_2d, fused, normed

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
            patch_scores_accum[q_start + bi] += score_bi.cpu() / float(num_layers_to_run)
        del min_dist_per_peer, sorted_dists

    del bank_host
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    執行 MuSc-V2 Training-Free & Zero-Shot 互評推論管線 (Position-wise 同座標 MSM + 全晶圓級 RsCon)：
    - 依 (y, x) 座標分批執行 SNAMD (SWPooling) 與 MSM，單批僅佔 N 塊記憶體 (700 張僅 ~7.2 GiB / 4層)。
    - 跨座標彙整每片晶圓「全片最高分 Patch」的倒數第二層特徵，最後統一執行原論文全晶圓級 RsCon (k=7 WMO)。
    - 公開中低解析度資料集全部切塊皆為 (0, 0)，自動無縫退化為標準整圖互評。
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
    num_layers_to_run = len(use_layers)
    penultimate_step = max(1, num_layers_to_run - 1)

    pos_groups: Dict[Tuple[int, int], List[Dict]] = {}
    img_fnames: List[str] = []
    total_tiles_all = 0
    for parent_idx, (fname, meta) in enumerate(manifest["images"].items()):
        img_fnames.append(fname)
        for t in meta["tiles"]:
            t_path = t.get("tile_abs_path") or os.path.join(tiled_pool_dir, t["tile_fname"])
            pos_key = (int(t["y"]), int(t["x"]))
            pos_groups.setdefault(pos_key, []).append({
                "parent_idx": parent_idx,
                "parent_fname": fname,
                "tile_stem": t["tile_stem"],
                "tile_fname": t["tile_fname"],
                "tile_path": t_path,
                "h": int(t["h"]),
                "w": int(t["w"]),
            })
            total_tiles_all += 1

    num_images = len(img_fnames)
    sorted_pos_keys = sorted(pos_groups.keys())
    num_positions = len(sorted_pos_keys)
    max_group_size = max(len(v) for v in pos_groups.values())

    avail_ram = get_available_ram_bytes()
    single_layer_bytes = max_group_size * num_patches * embed_dim * 2
    can_single_pass = (single_layer_bytes * (num_layers_to_run + 1)) <= int(avail_ram * 0.60)
    extract_mode_str = (
        f"單次 Forward 全層收集 (4x提速, 單座標峰值≈{(single_layer_bytes * num_layers_to_run)/(1024**3):.2f} GiB)"
        if can_single_pass
        else f"單層串流省記憶體模式 (AvailRAM={avail_ram/(1024**3):.1f} GiB)"
    )

    mode_desc = (
        f"同座標分批互評 Position-wise ({num_positions} 個座標格, 每格≤{max_group_size}塊)"
        if manifest.get("large_img_count", 0) > 0
        else "整張原圖直接互評 (Full-Image)"
    )
    peer_desc = "全體互評 (N×N)" if (max_peer_tiles <= 0 or max_group_size <= max_peer_tiles) else f"抽樣互評 (MaxPeers={max_peer_tiles})"
    log_and_print(
        f"\n[Step 2 — MuSc-V2 Zero-Shot 互評推理 ({mode_desc})] Device={device} | Images={num_images} | TotalTiles={total_tiles_all}\n"
        f"   ↳ 核心演算法: Layers={use_layers} | r_list={r_list} (SNAMD SWPooling) | {peer_desc} | Mutual_Ratio={mutual_ratio:.2f} | 全晶圓級 RsCon={'ON(k=7)' if rscon_weight > 0 else 'OFF'}\n"
        f"   ↳ 加速配置  : BatchSize={infer_batch_size} | I/O Workers={num_workers} | AMP={amp_str if use_amp else 'off'} | {extract_mode_str}",
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

    wafer_best_score = torch.full((num_images,), -1.0, dtype=torch.float32, device="cpu")
    wafer_salient_feat = torch.zeros((num_images, embed_dim), dtype=torch.float32, device="cpu")

    os.makedirs(tile_amap_dir, exist_ok=True)
    t_total_infer_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as io_pool:
        for pos_idx, pos_key in enumerate(sorted_pos_keys, 1):
            group_recs = pos_groups[pos_key]
            m_tiles = len(group_recs)

            if m_tiles < 2:
                zero_map = np.zeros((grid_hw, grid_hw), dtype=np.float32)
                np.save(os.path.join(tile_amap_dir, f"{group_recs[0]['tile_stem']}.npy"), zero_map)
                continue

            group_patch_scores = torch.zeros((m_tiles, num_patches), dtype=torch.float32, device="cpu")
            group_salient_feats = torch.zeros((m_tiles, embed_dim), dtype=torch.float32, device="cpu")
            group_parent_ids = torch.tensor([r["parent_idx"] for r in group_recs], dtype=torch.long)
            msm_chunk_size = (8 if m_tiles >= 600 else 16) if device.type == "cuda" else 8

            if can_single_pass:
                all_layer_bufs = [
                    torch.empty((m_tiles, num_patches, embed_dim), dtype=torch.float16, device="cpu")
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

                for start_i in range(0, m_tiles, infer_batch_size):
                    end_i = min(m_tiles, start_i + infer_batch_size)
                    batch_tensors = list(io_pool.map(_load_and_transform, group_recs[start_i:end_i]))
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
                        all_layer_bufs[slot_i][start_i:end_i].copy_(patch_tok.to(dtype=torch.float16, device="cpu"))
                        del tok, patch_tok
                    del x_batch
                    hook_holder.clear()

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
                        total_tiles=m_tiles,
                        num_patches=num_patches,
                        embed_dim=embed_dim,
                        grid_hw=grid_hw,
                        parent_ids_tensor=group_parent_ids,
                        mutual_ratio=mutual_ratio,
                        max_peer_tiles=max_peer_tiles,
                        num_layers_to_run=num_layers_to_run,
                        msm_chunk_size=msm_chunk_size,
                        device=device,
                        patch_scores_accum=group_patch_scores,
                        t_total_infer_start=t_total_infer_start,
                    )
                    if layer_step == penultimate_step:
                        max_p_idx = group_patch_scores.argmax(dim=1)
                        group_salient_feats.copy_(cur_buf[torch.arange(m_tiles), max_p_idx].float())
                    all_layer_bufs[layer_step - 1] = None
                    del cur_buf
                del all_layer_bufs
            else:
                storage_device = device if (device.type == "cuda" and m_tiles <= 350) else torch.device("cpu")
                layer_raw_buf = torch.empty((m_tiles, num_patches, embed_dim), dtype=torch.float16, device=storage_device)

                for layer_step, l_idx in enumerate(use_layers, 1):
                    hook_holder = {}
                    def _hook(_mod, _inp, out):
                        hook_holder["feat"] = out[0] if isinstance(out, tuple) else out
                    handle = blocks[l_idx].register_forward_hook(_hook)

                    for start_i in range(0, m_tiles, infer_batch_size):
                        end_i = min(m_tiles, start_i + infer_batch_size)
                        batch_tensors = list(io_pool.map(_load_and_transform, group_recs[start_i:end_i]))
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
                        layer_raw_buf[start_i:end_i].copy_(patch_tok.to(dtype=torch.float16, device=storage_device))
                        del x_batch, tok, patch_tok
                        hook_holder.clear()

                    handle.remove()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    _run_msm_for_single_layer(
                        layer_raw_buf=layer_raw_buf,
                        l_idx=l_idx,
                        layer_step=layer_step,
                        r_list=r_list,
                        total_tiles=m_tiles,
                        num_patches=num_patches,
                        embed_dim=embed_dim,
                        grid_hw=grid_hw,
                        parent_ids_tensor=group_parent_ids,
                        mutual_ratio=mutual_ratio,
                        max_peer_tiles=max_peer_tiles,
                        num_layers_to_run=num_layers_to_run,
                        msm_chunk_size=msm_chunk_size,
                        device=device,
                        patch_scores_accum=group_patch_scores,
                        t_total_infer_start=t_total_infer_start,
                    )
                    if layer_step == penultimate_step:
                        max_p_idx = group_patch_scores.argmax(dim=1).to(storage_device)
                        t_idx = torch.arange(m_tiles, device=storage_device)
                        group_salient_feats.copy_(layer_raw_buf[t_idx, max_p_idx].float().cpu())
                del layer_raw_buf

            group_tile_max = group_patch_scores.max(dim=1).values
            for i_local, rec in enumerate(group_recs):
                p_id = rec["parent_idx"]
                sc_val = float(group_tile_max[i_local].item())
                if sc_val > float(wafer_best_score[p_id].item()):
                    wafer_best_score[p_id] = sc_val
                    wafer_salient_feat[p_id].copy_(group_salient_feats[i_local])

            amap_2d_group = group_patch_scores.reshape(m_tiles, grid_hw, grid_hw).numpy()
            for i_local, rec in enumerate(group_recs):
                np.save(
                    os.path.join(tile_amap_dir, f"{rec['tile_stem']}.npy"),
                    amap_2d_group[i_local].astype(np.float32),
                )

            del group_patch_scores, group_salient_feats, amap_2d_group
            if device.type == "cuda":
                torch.cuda.empty_cache()

            _print_inline_progress(
                stage_title="Step 2 座標分批互評總進度",
                current=pos_idx,
                total=num_positions,
                start_t=t_total_infer_start,
                extra_info=f"pos=(y={pos_key[0]},x={pos_key[1]}), N={m_tiles}",
            )

    gc.collect()

    # 全晶圓級 RsCon (原論文 Eq. 14-17)
    wafer_rscon_scale = torch.ones(num_images, dtype=torch.float32)
    if rscon_weight > 0 and num_images > 2:
        valid_scores = torch.clamp(wafer_best_score, min=0.0)
        salient_norm = F.normalize(wafer_salient_feat, dim=-1)
        w_sim = torch.mm(salient_norm, salient_norm.t())
        w_sim.fill_diagonal_(-1e4)
        k_nbr = min(7, num_images - 1)
        top_w, top_idx = torch.topk(w_sim, k=k_nbr, dim=1)
        top_w = torch.clamp(top_w, min=1e-6)
        norm_w = top_w / top_w.sum(dim=1, keepdim=True)
        nbr_scores = valid_scores[top_idx]
        calibrated_max = 0.5 * valid_scores + 0.5 * (norm_w * nbr_scores).sum(dim=1)
        wafer_rscon_scale = calibrated_max / torch.clamp(valid_scores, min=1e-6)

    for p_id, fname in enumerate(img_fnames):
        manifest["images"][fname]["rscon_scale"] = float(wafer_rscon_scale[p_id].item())

    del model, wafer_best_score, wafer_salient_feat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# =====================================================================
# Step 3: 降維縫合、四合一防誤報純化、逆向轉回原圖角度與紅框定位
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
    """將單張影像的所有子圖塊 .npy 無縫縫合至目標畫布 (預設 <= 1024px)，套用 rscon_scale 並銷毀暫存 .npy。"""
    orig_h, orig_w = img_meta["height"], img_meta["width"]
    sh, sw, scale_y, scale_x = compute_stitch_canvas_shape(orig_h, orig_w, stitch_max_dim)
    tiles = img_meta["tiles"]
    rscon_scale = float(img_meta.get("rscon_scale", 1.0))

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
            return amap * rscon_scale

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

    sigma = max(0.8, 4.0 * min(scale_y, scale_x))
    stitched_amap = cv2.GaussianBlur(stitched_amap, (0, 0), sigmaX=sigma)

    if delete_tiles:
        for p in consumed_tile_npys:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
    return stitched_amap * rscon_scale


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
    """雙百分位門檻判定：僅計算 Top 5% (P95) 與 Top 3% (P97)，固定以 Top 5% 作為輸出紅框之有效門檻。"""
    scores_arr = np.array(pool_scores, dtype=np.float32)
    if scores_arr.size == 0:
        fallback = float(min_threshold)
        return fallback, f"無樣本保底門檻 ({fallback:.4f})"

    total_n = int(scores_arr.size)
    thresh_top5 = float(np.percentile(scores_arr, 95.0))
    thresh_top3 = float(np.percentile(scores_arr, 97.0))

    top5_count = int(np.sum(scores_arr >= thresh_top5))
    top5_pct = (top5_count / total_n) * 100.0
    top3_count = int(np.sum(scores_arr >= thresh_top3))
    top3_pct = (top3_count / total_n) * 100.0

    if top3_count == 0:
        top3_cut_info = f"高於全體最高分 ({float(np.max(scores_arr)):.4f})，切出 0 張 (0.00%)"
    else:
        top3_cut_info = (
            f"切在 Score = {thresh_top3:.4f} (位於 Rank {top3_count}/{total_n} 之後，"
            f"共 {top3_count} 張 >= 該門檻, 佔 {top3_pct:.2f}%)"
        )

    desc = (
        f"採用 [Top 5% (P95)] = {thresh_top5:.4f} 作為畫框門檻 "
        f"(切出 {top5_count}/{total_n} 張, {top5_pct:.2f}%)\n"
        f"           ├─ Top 5% (P95) 門檻 : {thresh_top5:.4f} (輸出 Bounding Box 基準)\n"
        f"           └─ Top 3% (P97) 參考 : {top3_cut_info}"
    )
    return thresh_top5, desc


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

    # 關鍵：若 Step 0 有對該圖進行旋轉對齊，輸出前將熱力圖逆向轉回原圖角度 (-align_angle)
    align_angle = float(meta.get("align_angle", 0.0))
    if abs(align_angle) > 1e-3:
        fill_med = float(np.median(active_amap))
        active_amap = rotate_image_or_amap(active_amap, -align_angle, border_value=fill_med, is_amap=True)

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
    """多核心平行縫合、逆向轉回原圖角度與篩選畫框。"""
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
        f"\n[Step 3.1] 啟動 RAM 多核心平行縫合、逆轉原圖角度與碎片銷毀模式 (共 {total_imgs} 張原圖 | "
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
                    stage_title="Step 3.1 RAM縫合與逆轉",
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
                        stage_title=f"Step 3.1 平行縫合逆轉(x{eff_workers})",
                        current=completed,
                        total=total_imgs,
                        start_t=t_stitch_loop_start,
                        extra_info=f"最新分數={res['max_score']:.4f}",
                    )

    safe_rmtree(tile_amap_dir)
    log_and_print(
        f"[🧹 碎片清理完成] 所有熱力圖已於 RAM 內完成重建並逆轉回原圖角度，已自動銷毀全部暫存碎片 .npy -> {os.path.abspath(tile_amap_dir)}",
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

        # active_amap 已逆轉回原圖方向，直接將紅框座標映回 1:1 原圖解析度
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

        # 直接讀取未旋轉的原始圖片作為底圖
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
        ang_tag = f" | Align={meta.get('align_angle', 0.0):.1f}deg" if abs(meta.get("align_angle", 0.0)) > 1e-3 else ""
        banner_txt = (
            f"[{mode_tag}] Orig: {fname}{ang_tag} | Score: {max_score:.4f} (Raw: {raw_max_score:.4f}, Filt: {filtered_max_score:.4f}, "
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

    # Step 0 方向對齊參數
    parser.add_argument(
        "--align_mode", type=str, default="full360",
        choices=["full360", "discrete90", "off"],
        help="Step 0 方向對齊模式：full360=任意角度對齊第一張圖, discrete90=僅90度倍數, off=關閉",
    )

    # MuSc-V2 核心互評超參數
    parser.add_argument("--r_list", type=str, default="1,3,5", help="MuSc-V2 局部鄰域聚合尺度清單 (例如 1,3,5)")
    parser.add_argument("--feature_layers", type=str, default="", help="指定提取的 Transformer 層數 (留空自動依骨幹深度配置)")
    parser.add_argument("--mutual_ratio", type=float, default=0.30, help="MuSc-V2 互評機制取最低距離的比例 (預設 0.30 即 30%%)")
    parser.add_argument("--rscon_weight", type=float, default=0.15, help="MuSc-V2 全晶圓級 RsCon 重評開關 (>0 啟用 k=7 WMO，0 關閉)")
    parser.add_argument("--max_peer_tiles", type=int, default=400, help="單座標互評時每個 Query 最多抽樣比較的他片圖塊數 (設 0 則全體 N×N 互評)")

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
    parser.add_argument("--infer_batch_size", type=int, default=64, help="提取特徵時的 Batch Size")
    parser.add_argument("--num_workers", type=int, default=default_workers, help="對齊、切塊、讀圖與縫合之多核心 Worker 數")
    parser.add_argument("--stitch_max_dim", type=int, default=1024, help="縫合與濾波畫布最大邊長 (預設 1024，設 0 則退回原圖解析度)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "off"], help="GPU 混合精度設定")

    # mask 參數
    parser.add_argument("--pre_mask", action="store_true", help="推論前先用圓形 Mask 蓋掉背景")
    parser.add_argument("--pre_mask_margin", type=float, default=0.0, help="圓形 Mask 內縮比例 (例如 0.02 代表半徑內縮 2%%，負值為外擴)")
    parser.add_argument("--pre_mask_fill", type=int, default=0, help="背景填補像素值 (預設 0=純黑)")
    parser.add_argument("--preview_mask", type=int, default=0, help="大於 0 時僅抽樣 N 張輸出 Mask 預覽圖並直接結束，不執行推論")
    parser.add_argument("--mask_cx", type=float, default=-1.0, help="手動指定圓心 X (0~1 為寬度比例，>1 為像素，-1 為自動)")
    parser.add_argument("--mask_cy", type=float, default=-1.0, help="手動指定圓心 Y (0~1 為高度比例，>1 為像素，-1 為自動)")
    parser.add_argument("--mask_r", type=float, default=-1.0, help="手動指定半徑 R (0~1 為短邊比例，>1 為像素，-1 為自動)")
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
        f"[無損加速設定] Align={args.align_mode} | BatchSize={args.infer_batch_size} | Workers={args.num_workers} | "
        f"StitchCanvas={args.stitch_max_dim}px | AMP={args.amp_dtype}\n"
        f"{'=' * 75}",
        log_files,
    )

    if args.preview_mask > 0:
        run_mask_preview(
            pool_dir=args.pool_dir,
            out_dir=run_out_dir,
            num_samples=args.preview_mask,
            wafer_edge_mode=args.wafer_edge_mode,
            category=args.category,
            pre_mask_margin=args.pre_mask_margin,
            fill_value=args.pre_mask_fill,
            manual_cx=args.mask_cx,
            manual_cy=args.mask_cy,
            manual_r=args.mask_r,
            log_files=log_files,
        )
        return

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
        f"al{args.align_mode}_mtr{args.min_tiling_res}_t{args.tile_size}_ov{args.tile_overlap}"
        f"_bg{args.bg_std_thresh:g}_dt{int(args.disable_tiling)}"
        f"_pm{int(args.pre_mask)}_pmm{args.pre_mask_margin:g}_pmf{args.pre_mask_fill}"
        f"_cx{args.mask_cx:g}_cy{args.mask_cy:g}_r{args.mask_r:g}"
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
            f"對齊與切塊 PNG ({total_test_tiles} 塊)"
            if manifest.get("large_img_count", 0) > 0
            else f"整張原圖直接推論 ({num_test_images} 張，防呆跳過切塊)"
        )
        log_and_print(
            f"\n[⚡ 測試池清單快取命中] 測試圖片池與參數未變動，直接沿用既有 {cache_mode_note} (0.00 秒)",
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
            pre_mask=args.pre_mask,
            pre_mask_margin=args.pre_mask_margin,
            pre_mask_fill=args.pre_mask_fill,
            manual_cx=args.mask_cx,
            manual_cy=args.mask_cy,
            manual_r=args.mask_r,
            align_mode=args.align_mode,
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
        f"Step 0+1 方向對齊與切塊 ({total_test_tiles} 塊)"
        if manifest.get("large_img_count", 0) > 0
        else f"Step 0+1 方向對齊與檢查 ({num_test_images} 張全圖)"
    )

    infer_done_summary = (
        f"\n{'=' * 75}\n"
        f" [✅ MuSc-V2 推論完成 — 整體時間與所有檔案儲存位置總覽]\n"
        f"{'-' * 75}\n"
        f"  [1] 整體推論階段總時間 (Total Inference Time) : {format_duration(total_infer_time)}\n"
        f"      ├─ {step1_label:<38} : {format_duration(test_tile_sec)}"
        f"{' (Cached)' if used_png_tile_cache else ''}\n"
        f"      ├─ Step 2 MuSc-V2 同座標互評與全晶圓RsCon (BS={args.infer_batch_size}) : {format_duration(model_infer_sec)}\n"
        f"      ├─ Step 3 RAM 縫合、逆轉原圖與{'未濾波' if args.raw_score_mode else '防誤報濾波'}畫框 (Canvas<={args.stitch_max_dim}px) : {format_duration(stitch_vis_sec)}\n"
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