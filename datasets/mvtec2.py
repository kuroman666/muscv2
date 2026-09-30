#新增的檔案
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MVTec2Dataset(Dataset):
    def __init__(self, data_path, class_name, is_train=False, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.mask_paths = [] 
        
        if is_train:
            folders = ["train"]
        else:
            folders = ["test_public", "test_private", "test_private_mixed"]
            
        for folder_name in folders:
            # 例如： data/mvtec-2/can/test_public
            target_dir = os.path.join(data_path, class_name, folder_name)
            
            if os.path.exists(target_dir):
                for root, dirs, files in os.walk(target_dir):
                    
                    # 【關鍵修改 1】如果當前路徑包含 ground_truth，就直接跳過，不要把它當成測試圖片讀取
                    if "ground_truth" in root:
                        continue
                        
                    for file_name in files:
                        if file_name.endswith(".png") or file_name.endswith(".jpg"):
                            full_path = os.path.join(root, file_name)
                            self.image_paths.append(full_path)
                            
                            # 判斷是否為正常圖片
                            if "good" in full_path:
                                self.labels.append(0)
                                self.mask_paths.append(None) 
                            else:
                                self.labels.append(1)
                                
                                # 【關鍵修改 2】推導正確的 Mask 路徑
                                # 此時的 root 會是類似： .../can/test_public/bad
                                # 我們用 os.path.basename 抓出最後面的 'bad' (瑕疵類別)
                                defect_category = os.path.basename(root)
                                
                                # 組合成截圖中的正確路徑： .../can/test_public/ground_truth/bad
                                mask_dir = os.path.join(target_dir, "ground_truth", defect_category)
                                
                                # MVTec 通常會把 Mask 檔名加上 _mask.png
                                mask_name = file_name.replace(".png", "_mask.png")
                                mask_path = os.path.join(mask_dir, mask_name)
                                
                                # 防呆機制：如果加了 _mask 的檔案不存在，就直接用原檔名找找看
                                if not os.path.exists(mask_path):
                                    mask_path = os.path.join(mask_dir, file_name)
                                
                                self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        if self.transform:
            image = self.transform(image)
            
        _, h, w = image.shape
        
        # 建立正確的 Mask
        if label == 0:
            mask = torch.zeros((1, h, w))
        else:
            if mask_path is not None and os.path.exists(mask_path):
                # 讀取真實的 Mask，轉成灰階 (L) 並調整大小
                mask_img = Image.open(mask_path).convert('L')
                mask_img = mask_img.resize((w, h))
                mask_tensor = transforms.ToTensor()(mask_img)
                mask = (mask_tensor > 0.5).float()
            else:
                mask = torch.zeros((1, h, w))
            
        return {
            "image": image,
            "image_path": img_path,
            "mask": mask,
            "is_anomaly": torch.tensor(label)
        }