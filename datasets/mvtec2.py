import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def preprocess_wafer_image(img_bgr, ref_template=None):
    """
    三階段晶圓規範化 (Wafer Canonicalization)：
    - 若三個環境變數皆為 '0' (例如在公司外測試 MVTec-2 can)，直接回傳原圖。
    - 若任一開關啟用，統一將畫布轉為 S x S 正方形，確保晶圓永遠維持正圓形且轉 90 度不變形。
    """
    if img_bgr is None:
        return img_bgr

    align_on = os.environ.get('AUTO_ALIGN_WAFER', '0') == '1'
    photo_on = os.environ.get('PHOTOMETRIC_NORM', '0') == '1'
    rot90_on = os.environ.get('FIX_90DEG_ROTATION', '0') == '1'

    # 若三個開關都沒開 (公司外測一般資料集)，完全不動原圖直接回傳
    if not (align_on or photo_on or rot90_on):
        return img_bgr

    H, W = img_bgr.shape[:2]
    S = min(H, W)  # 統一使用正方形畫布邊長 S，避免 Resize(518, 518) 時正圓被壓成橢圓
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 【步驟 1：自動抓取晶圓圓心、統一半徑為 0.45*S 並輸出至 SxS 正方形畫布】 ---
    if align_on:
        corner_mean = np.mean([blurred[0, 0], blurred[0, -1], blurred[-1, 0], blurred[-1, -1]])
        thresh_type = cv2.THRESH_BINARY_INV if corner_mean > 127 else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(largest_cnt)

            if r >= S * 0.25:
                inner_mask = np.zeros_like(gray)
                cv2.circle(inner_mask, (int(cx), int(cy)), int(r * 0.85), 255, thickness=-1)
                edges = cv2.Canny(blurred, 50, 150)
                edges = cv2.bitwise_and(edges, edges, mask=inner_mask)

                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                        minLineLength=int(r * 0.15), maxLineGap=10)
                angle_deg = 0.0
                if lines is not None and len(lines) > 0:
                    angles = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        deg_mod = ((deg + 45.0) % 90.0) - 45.0
                        if abs(deg_mod) < 20.0:
                            angles.append(deg_mod)
                    if len(angles) > 0:
                        angle_deg = float(np.median(angles))

                target_r = S * 0.45  # 對齊後晶圓半徑固定為 0.45*S (即直徑占畫面 90%)
                scale = target_r / (r + 1e-8)
                M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
                M[0, 2] += (S / 2.0) - cx
                M[1, 2] += (S / 2.0) - cy

                img_bgr = cv2.warpAffine(
                    img_bgr, M, (S, S),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # [修正 1] 防呆：無論是否成功抓到外圓，確保進入步驟 2 前畫布一定已經是 S x S 正方形，避免長方形原圖報 IndexError
    if img_bgr.shape[0] != S or img_bgr.shape[1] != S:
        y0, x0 = (H - S) // 2, (W - S) // 2
        img_bgr = img_bgr[y0:y0 + S, x0:x0 + S]
        gray = gray[y0:y0 + S, x0:x0 + S]

    # 建立晶圓內部取樣遮罩 (半徑 0.42*S)，僅供統計亮度與旋轉相似度使用
    roi_mask = np.zeros((S, S), dtype=np.uint8)
    cv2.circle(roi_mask, (S // 2, S // 2), int(S * 0.42), 1, thickness=-1)

    # --- 【步驟 2：前景光度正規化 (只用 roi_mask 算統計量，完全不塗黑外圍，避免感受野擴張時碰到人工黑邊！)】 ---
    if photo_on:
        roi_pixels = gray[roi_mask == 1].astype(np.float32)
        if roi_pixels.size > 0:
            med = np.median(roi_pixels)
            q75, q25 = np.percentile(roi_pixels, [75, 25])
            robust_std = max((q75 - q25) / 1.349, 1e-4)

            norm_gray = (gray.astype(np.float32) - med) / robust_std * 40.0 + 128.0
            norm_gray = np.clip(norm_gray, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)
            gray = norm_gray


    # --- 【步驟 3：離散 90 度方向配準 (解決晶圓轉向 90/180/270 度)】 ---
    if ref_template is not None and rot90_on:
        small_gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        small_mask = cv2.resize(roi_mask, (128, 128), interpolation=cv2.INTER_NEAREST) > 0

        best_k = 0
        best_corr = -1e9
        ref_vec = ref_template[small_mask]
        ref_vec = ref_vec - np.mean(ref_vec)

        for k in range(4):
            cand = np.rot90(small_gray, k=k)
            cand_vec = cand[small_mask]
            cand_vec = cand_vec - np.mean(cand_vec)
            corr = np.dot(cand_vec, ref_vec) / (np.linalg.norm(cand_vec) * np.linalg.norm(ref_vec) + 1e-8)
            if corr > best_corr:
                best_corr = corr
                best_k = k

        if best_k > 0:
            img_bgr = np.ascontiguousarray(np.rot90(img_bgr, k=best_k))

    return img_bgr


class MVTec2Dataset(Dataset):
    ref_template = None

    def __init__(self, data_path, class_name, is_train=False, transform=None):
        self.data_path = data_path
        self.class_name = class_name
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.masks = []

        target_folder = os.path.join(self.data_path, self.class_name)

        if os.path.exists(target_folder):
            for root, dirs, files in os.walk(target_folder):
                if 'ground_truth' in root:
                    continue
                for file_name in sorted(files):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        full_path = os.path.join(root, file_name)
                        self.image_paths.append(full_path)
                        self.labels.append(0)
                        self.masks.append(None)
        else:
            print(f"[警告] 找不到資料夾：{target_folder}")

        # 若有開啟晶圓前處理，才初始化第一張晶圓作為 0 度基準模板
        if len(self.image_paths) > 0 and os.environ.get('FIX_90DEG_ROTATION', '0') == '1':
            first_bgr = cv2.imread(self.image_paths[0])
            first_aligned = preprocess_wafer_image(first_bgr, ref_template=None)
            first_gray = cv2.cvtColor(first_aligned, cv2.COLOR_BGR2GRAY)
            MVTec2Dataset.ref_template = cv2.resize(first_gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            MVTec2Dataset.ref_template = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_bgr = cv2.imread(img_path)
        img_bgr = preprocess_wafer_image(img_bgr, ref_template=MVTec2Dataset.ref_template)
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)

        dummy_mask = torch.zeros((1, img.shape[1], img.shape[2]))

        return {
            "image": img,
            "image_path": img_path,
            "mask": dummy_mask,
            "is_anomaly": 0
        }
