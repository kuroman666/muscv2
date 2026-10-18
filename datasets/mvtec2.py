#新增的檔案
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MVTec2Dataset(Dataset):
    def __init__(self, data_path, class_name, is_train=False, transform=None):
        self.data_path = data_path
        self.class_name = class_name
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.masks = []
        
        target_folder = os.path.join(self.data_path, self.class_name)
        
        # ====== [進公司需確認處：自動相容公司外(多層子目錄)與公司內(單層目錄)] ======
        # 若在公司外只想測試特定子資料夾 (如 test_private_mixed)，可在 .sh 將 CLASS_NAME 設為 "can/test_private_mixed"
        if os.path.exists(target_folder):
            for root, dirs, files in os.walk(target_folder):
                # 排除 ground_truth 遮罩資料夾，避免把黑白遮罩當成待測圖片讀進來
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
        # ============================================================================

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # 自動跟隨 transform 後的圖片長寬 (支援 224, 392, 518 等任意尺寸)
        dummy_mask = torch.zeros((1, img.shape[1], img.shape[2]))
        
        return {
            "image": img,
            "image_path": img_path,
            "mask": dummy_mask,
            "is_anomaly": 0
        }
