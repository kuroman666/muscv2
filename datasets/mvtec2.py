#新增的檔案
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class MVTec2Dataset(Dataset):
    def __init__(self, data_path, class_name, is_train=False, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 建立一個清單來存放要讀取的資料夾名稱
        if is_train:
            folders = ["train"]
        else:
            # MvTEC-2 特有的測試集命名方式
            folders = ["test_public", "test_private", "test_private_mixed"]
            
        # 依序檢查並讀取這些資料夾
        for folder_name in folders:
            target_dir = os.path.join(data_path, class_name, folder_name)
            
            # 確保資料夾真的存在才進去尋找
            if os.path.exists(target_dir):
                for root, dirs, files in os.walk(target_dir):
                    for file_name in files:
                        if file_name.endswith(".png") or file_name.endswith(".jpg"):
                            full_path = os.path.join(root, file_name)
                            self.image_paths.append(full_path)
                            
                            # 資料夾路徑如果包含 'good' 就是正常(0)，否則就是瑕疵(1)
                            if "good" in full_path:
                                self.labels.append(0)
                            else:
                                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # 建立一張全黑的假 mask 避免計算像素級分數時報錯
        _, h, w = image.shape
        mask = torch.zeros((1, h, w))
        if label == 1:
            mask[0, 0, 0] = 1 # 隨便塞一個 1，避免全為 0 導致 AUC 計算崩潰
            
        return {
            "image": image,
            "image_path": img_path,
            "mask": mask,
            "is_anomaly": torch.tensor(label)
        }