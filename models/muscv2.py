import os
import sys
import datetime
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append('./models/backbone')
import models.backbone._backbones as _backbones
from models.modules._SNAMD import SNAMD
from models.modules._MSM import MSM_single, MSM_multiple, patch2point, point2patch
from models.modules._RsCon import RsCon
from utils.tools import organized_pc_to_unorganized_pc
from utils.pointnet2_utils import interpolating_points
from utils.metrics import compute_metrics
from openpyxl import Workbook
from tqdm import tqdm
import random
import cv2
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MuScV2():
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda:{}".format(cfg['device']) if torch.cuda.is_available() else "cpu")
        self.path = cfg['datasets']['data_path']
        self.dataset = cfg['datasets']['dataset_name']
        self.vis = cfg['testing']['vis']
        self.vis_type = cfg['testing']['vis_type']
        self.save_excel = cfg['testing']['save_excel']

        # the categories to be tested
        self.categories = cfg['datasets']['class_name']
        if isinstance(self.categories, str):
            if self.categories.lower() == 'all':
                self.categories = self.load_datasets(get_categories=True)
            else:
                self.categories = [self.categories]
        self.model_name = cfg['models']['backbone_name']

        # test pattern (only 2D modal, only 3D modal, or 2D+3D multimodal)
        if len(self.model_name) == 1:
            if self.model_name[0] in ['point-mae']:
                self.modality = '3d'
            else:
                self.modality = '2d'
        else:
            self.modality = '2d+3d'

        self.image_size = cfg['datasets']['img_resize']
        self.features_list = [l+1 for l in cfg['models']['feature_layers']]
        self.r_list = cfg['models']['r_list']

        # 優先讀取 .sh 腳本建立好的【資料集名稱_台北時間】資料夾路徑
        if 'EXP_OUTPUT_DIR' in os.environ:
            self.output_dir = os.environ['EXP_OUTPUT_DIR']
        else:
            raw_path = cfg['datasets']['data_path']
            dataset_id = raw_path.strip('/').split('/')[-1]
            taipei_tz = datetime.timezone(datetime.timedelta(hours=8))
            current_time = datetime.datetime.now(taipei_tz).strftime("%Y%m%d_%H%M%S")
            folder_name = f"{dataset_id}_{current_time}"
            self.output_dir = os.path.join(cfg['testing']['output_dir'], self.dataset, '-'.join(self.model_name), 'imagesize{}'.format(self.image_size), folder_name) + '/'

        os.makedirs(self.output_dir, exist_ok=True)


    def load_backbone(self):
        if self.modality in ['2d', '2d+3d']:
            from models.backbone.dino_vision_transformer import vit_large
            
            # 1. 建立 DINOv2 Large 模型
            self.dino_model = vit_large(patch_size=14, img_size=518)
           
            # 2. 讀取權重檔
            weight = torch.load('./dinov2_vitl14_reg4_pretrain.pth', map_location='cpu')
            
            # 3. 將權重寫入模型
            self.dino_model.load_state_dict(weight, strict=False)

            # 4. 放到 GPU 
            self.dino_model.to(self.device)

        if self.modality in ['3d', '2d+3d']:
            from models.backbone.point_transformer import PointTransformer
            self.point_transformer = PointTransformer(group_size=128, num_group=1024).to(self.device)
            self.point_transformer.load_model_from_ckpt('./pointmae_pretrain.pth')
            
            
    def load_datasets(self, category='', get_categories=False):
        if self.dataset == 'mvtec_3d':
            import datasets.mvtec3d as mvtec3d
            from datasets.mvtec3d import _CLASSNAMES as _CLASSNAMES_mvtec3d
            if get_categories:
                return _CLASSNAMES_mvtec3d
            test_dataset = mvtec3d.MVTec3dDataset(source=self.path, split=mvtec3d.DatasetSplit.TEST,
                                            classname=category, resize=self.image_size, imagesize=self.image_size)
        elif self.dataset == 'eyecandies':
            import datasets.eyecandies as eyecandies
            from datasets.eyecandies import _CLASSNAMES as _CLASSNAMES_eyecandies
            if get_categories:
                return _CLASSNAMES_eyecandies
            test_dataset = eyecandies.EyecandiesDataset(source=self.path, split=eyecandies.DatasetSplit.TEST,
                                            classname=category, resize=self.image_size, imagesize=self.image_size)
        elif self.dataset in ['mvtec_2', 'TMUN87']:
            if get_categories: 
                return [category]
                
            import torchvision.transforms as transforms
            from datasets.mvtec2 import MVTec2Dataset
            
            my_transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            test_dataset = MVTec2Dataset(
                data_path=self.path, 
                class_name=category, 
                is_train=False, 
                transform=my_transforms
            )

        return test_dataset


    def visualization(self, image_path_list, gt_list, pr_px, category):
        # 1. 嚴格遵循 HUST-SLOW/MuSc-V2 官方論文之 normalization01 熱力圖計算方式
        def normalization01(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        if pr_px.ndim == 2:
            pr_px = np.expand_dims(pr_px, axis=0)

        # 複製一份用來進行論文標準 [0, 1] 正規化 (支援 single_norm 與 whole_norm)
        norm_pr_px = pr_px.copy()
        if self.vis_type == 'single_norm':
            for i in range(norm_pr_px.shape[0]):
                norm_pr_px[i] = normalization01(norm_pr_px[i])
        elif self.vis_type == 'whole_norm':
            norm_pr_px = normalization01(norm_pr_px)

        # 2. 讀取 .sh 腳本設定的 THRESHOLD、ALPHA 與 WAFER_MASK_RATIO
        threshold = float(os.environ.get('THRESHOLD', 0.5))
        alpha = float(os.environ.get('ALPHA', 0.6))
        wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))
        print(f"正在產生視覺化圖片 [模式: {self.vis_type} | Threshold={threshold} (0~1尺度) | Alpha={alpha} | Wafer_ROI={wafer_mask_ratio}]...")
        
        save_path = os.path.join(self.output_dir, 'vis', category)
        os.makedirs(save_path, exist_ok=True)

        for i, img_path in enumerate(image_path_list):
            # --- 【第 1 張：原圖】 ---
            orig_img = cv2.imread(img_path)
            orig_img = cv2.resize(orig_img, (self.image_size, self.image_size))

            # 取得該張圖片經論文公式正規化後的 [0, 1] 熱力圖矩陣
            norm_map = norm_pr_px[i]
            if norm_map.shape[0] != self.image_size:
                norm_map = cv2.resize(norm_map, (self.image_size, self.image_size))

            # 依論文 cvt2heatmap 計算出對應色彩與超過門檻之瑕疵遮罩 (defect_mask)
            norm_map_uint8 = np.uint8(norm_map * 255)
            paper_heatmap = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)
            defect_mask = norm_map >= threshold

            # --- 【第 2 張：單獨瑕疵圖 (不顯示背景 Heatmap，僅單獨呈現抓出的瑕疵)】 ---
            isolated_defect_img = np.zeros_like(orig_img)
            blended_defect = cv2.addWeighted(orig_img, 1 - alpha, paper_heatmap, alpha, 0)
            isolated_defect_img[defect_mask] = blended_defect[defect_mask]

            # --- 【第 3 張：匡出瑕疵的圖 (紅框標示瑕疵 + 細綠圈標示晶圓有效檢測區)】 ---
            boxed_defect_img = orig_img.copy()
            mask_uint8 = np.uint8(defect_mask * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # 在原圖上畫出紅色外框 (BGR: 0, 0, 255)，線條粗細為 2
                cv2.rectangle(boxed_defect_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 若有啟用晶圓圓形遮罩 (WAFER_MASK_RATIO > 0)，畫出細綠線圓圈代表「有效晶圓檢測區」
            if wafer_mask_ratio > 0.0:
                center = (self.image_size // 2, self.image_size // 2)
                radius = int(self.image_size * 0.5 * wafer_mask_ratio)
                # 畫出細綠線圓圈 (若想改用虛線，可將下行註解，並取消註解下方的 for 迴圈)
                cv2.circle(boxed_defect_img, center, radius, (0, 255, 0), 1, cv2.LINE_AA)
                # for angle in range(0, 360, 6):
                #     cv2.ellipse(boxed_defect_img, center, (radius, radius), 0, angle, angle + 3, (0, 255, 0), 1, cv2.LINE_AA)

            # 3. 儲存圖片：「左：原圖 | 中：單獨瑕疵圖 | 右：匡出瑕疵的圖」
            img_name = os.path.basename(img_path)
            name_no_ext, ext = os.path.splitext(img_name)
            
            # 取得上一層資料夾名稱 (例如 good 或 bad)，避免同名檔案互相覆蓋！
            parent_dir = os.path.basename(os.path.dirname(img_path))

            combined_img = np.hstack((orig_img, isolated_defect_img, boxed_defect_img))
            cv2.imwrite(os.path.join(save_path, f"{parent_dir}_{name_no_ext}_comparison{ext}"), combined_img)

        
        print(f"視覺化完成！三合一圖片 (原圖 | 單獨瑕疵圖 | 匡出瑕疵的圖) 已儲存至: {save_path}")

            

    def make_category_data(self, category):
        print(f'Category: {category}')
        setup_seed(self.seed)
        self.load_backbone()
        test_dataset = self.load_datasets(category)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        gt_list = []
        img_masks = []
        image_path_list = []
        patch_tokens_list = []
        point_tokens_list = []
        pc_list = []
        pc_center_list = []
        nonzero_indices_list = []
        nonzero_indices_center_list = []
        nonzero_indices_center_neighbor_list = []
        curvatures_list = []

        print(f"資料集總共讀到 {len(test_dataset)} 張圖片！")

        # extract features
        print('Features extracting...')
        for image_info in tqdm(test_dataloader):
            image = image_info["image"]
           image_path_list.extend(image_info["image_path"])
            img_masks.append(image_info["mask"].cpu())
            gt_list.extend(list(image_info["is_anomaly"].numpy()))
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.modality != '3d':
                    # 2D modal feature extraction
                    input_image = image.to(torch.float).to(self.device)
                    patch_tokens_all = self.dino_model.get_intermediate_layers(x=input_image, n=max(self.features_list))
                    
                    # 【關鍵修正】若 Backbone (如 DINOv2) 輸出不含 CLS token (長度已為完全平方數，如 37x37=1369)
                    # 在最前面補 1 個 dummy token (變成 1370)，讓 _SNAMD.py 切掉 index 0 後剛好還原為 1369！
                    patch_tokens = []
                    for l in self.features_list:
                        pt = patch_tokens_all[l-1].cpu() # (B, L, C)
                        if int(np.sqrt(pt.shape[1])) ** 2 == pt.shape[1]:
                            dummy_cls = torch.zeros((pt.shape[0], 1, pt.shape[2]), dtype=pt.dtype)
                            pt = torch.cat([dummy_cls, pt], dim=1) # (B, L+1, C)
                        patch_tokens.append(pt)

                    patch_tokens_list.append(patch_tokens) # (B, L+1, C)
                    
                # 特徵提取完畢後，釋放 DINO backbone 佔用的 GPU 記憶體
                if self.modality in ['2d', '2d+3d']:
                    del self.dino_model
                    torch.cuda.empty_cache()

               # Similar Neighborhood Aggregation with Multiple Degrees (SNAMD)
               print('Features aggregating...')
               SNAMD_r = SNAMD(device=self.device, r_list=self.r_list)
               Z_2d_layers = {}
               Z_3d_layers = {}
               if self.modality != '3d':
                   # aggregate 2D features
                   for im in range(len(patch_tokens_list)):
                       with torch.no_grad(), torch.cuda.amp.autocast():
                           # 【修正拼字】patch_tokens (有 s)、self.device (無 s)、_embed_2d (有底線)、True
                           patch_tokens = [p.to(self.device) for p in patch_tokens_list[im]]
                           features = SNAMD_r._embed_2d(patch_tokens) # (B, L, layer, C)
                           features /= features.norm(dim=-1, keepdim=True)
                           for l in range(features.shape[2]):
                               if str(l) not in Z_2d_layers.keys():
                                   Z_2d_layers[str(l)] = []
                           Z_2d_layers[str(l)].append(features[:, :, l, :].cpu())
                    del patch_tokens_list, SNAMD_r
                    torch.cuda.empty_cache()
                if len(point_tokens_list) != 0:
                    for im in range(len(point_tokens_list)):
                       with torch.no_grad(), torch.cuda.amp.autocast():
                           patch_tokens = point_tokens_list[im]
                           patch_tokens = [p/p.norm(dim=-1, keepdim=True) for p in patch_tokens]
                           center = pc_center_list[im]
                           curvatures = curvatures_list[im]
                           features = SNAMD_r._embed_3d(patch_tokens, center, curvatures)
                           for l in range(features.shape[2]):
                               if str(l) not in Z_3d_layers.keys():
                                   Z_3d_layers[str(l)] = []
                               Z_3d_layers[str(l)].append(features[:, :, l, :])

        # Multimodal Mutual Scoring
        layer_keys = Z_2d_layers.keys() if self.modality != '3d' else Z_3d_layers.keys()
        if self.modality == '2d+3d':
            nonzero_indices_center_list = torch.cat(nonzero_indices_center_list, 0)
            nonzero_indices_center_neighbor_list = torch.cat(nonzero_indices_center_neighbor_list, 0)
            patch2point_list = patch2point(Z_2d_layers[list(Z_2d_layers.keys())[0]][0].shape[1], nonzero_indices_center_list, self.image_size)
            point2patch_list = point2patch(Z_2d_layers[list(Z_2d_layers.keys())[0]][0].shape[1], nonzero_indices_center_list, self.image_size)
        else:
            patch2point_list = None
            point2patch_list = None
        anomaly_maps_2d = torch.tensor([]).float().to(self.device)
        anomaly_maps_3d = torch.tensor([]).float().to(self.device)
       
        for l in layer_keys:
            print(f'Mutual scoring in layer{l}...')
            if self.modality != '3d':
                # 確保每個元素的裝置一致後再 cat，並放上 GPU
                Z_2d = torch.cat([z.to(self.device) for z in Z_2d_layers[l]], dim=0)
            if self.modality != '2d':
                Z_3d = torch.cat([z.to(self.device) for z in Z_3d_layers[l]], dim=0)
            if self.modality == '2d':
                anomaly_maps_2d_msm = MSM_single(Z=Z_2d, device=self.device)
                # 強制將 MSM_single 回傳的結果對齊到 self.device (cuda:0)，解決 wrapper_CUDA_cat 報錯
                anomaly_maps_2d = torch.cat((anomaly_maps_2d, anomaly_maps_2d_msm.to(self.device).unsqueeze(0)), dim=0)
            if self.modality == '3d':
                anomaly_maps_3d_msm = MSM_single(Z=Z_3d, device=self.device)
                anomaly_maps_3d = torch.cat((anomaly_maps_3d, anomaly_maps_3d_msm.to(self.device).unsqueeze(0)), dim=0)
            elif self.modality == '2d+3d':
                anomaly_maps_2d_msm, anomaly_maps_3d_msm = MSM_multiple(Z_2d=Z_2d, Z_3d=Z_3d, device=self.device, patch2point_list=patch2point_list, point2patch_list=point2patch_list)
                anomaly_maps_3d = torch.cat((anomaly_maps_3d, anomaly_maps_3d_msm.to(self.device).unsqueeze(0)), dim=0)
                anomaly_maps_2d = torch.cat((anomaly_maps_2d, anomaly_maps_2d_msm.to(self.device).unsqueeze(0)), dim=0)
            if self.modality != '3d':
                anomaly_maps_2d = torch.mean(anomaly_maps_2d, 0)
            if self.modality != '2d':
                anomaly_maps_3d = torch.mean(anomaly_maps_3d, 0)
                 
        # calculate anomaly-salient features
        anomaly_salient_feat_2d = []
        anomaly_salient_feat_3d = []
        if self.modality in ['2d', '2d+3d']:
            B, L = anomaly_maps_2d.shape
            layers_2d = list(Z_2d_layers.keys())
            # 確保 features_2d 與索引都在同一個 device 上運算完再轉回 cpu
            features_2d = torch.cat([z.to(self.device) for z in Z_2d_layers[layers_2d[-2]]], dim=0)
            amap_max_idx = torch.max(anomaly_maps_2d, dim=-1)[1]
            batch_idx = torch.arange(B, device=self.device)
            anomaly_salient_feat_2d.append(features_2d[batch_idx, amap_max_idx, :].cpu())
            H = int(np.sqrt(L))
            anomaly_maps_2d = F.interpolate(anomaly_maps_2d.view(B, 1, H, H),
                                            size=self.image_size, mode='bilinear', align_corners=True).squeeze()     
         if self.modality in ['3d', '2d+3d']:
            from utils.tools import KNNGaussianBlur
            B, L = anomaly_maps_3d.shape
            layers_3d = list(Z_3d_layers.keys())
            features_3d = torch.cat(Z_3d_layers[layers_3d[-2]], dim=0) if len(layers_3d) != 0 else None
            amap_max_idx = torch.max(anomaly_maps_3d, dim=-1)[1]
            anomaly_salient_feat_3d.append(features_3d[torch.arange(B), amap_max_idx, :].cpu())
            KnnBlur = KNNGaussianBlur(4)
            anomaly_maps_3d = anomaly_maps_3d.unsqueeze(0).unsqueeze(-1)
            anomaly_maps_3d_resized = []
            for img_i in tqdm(range(B)):
                anomaly_map = anomaly_maps_3d[:, img_i, :, :]
                anomaly_map = interpolating_points(pc_list[img_i].permute(0,2,1), pc_center_list[img_i], anomaly_map)
                resized_anomaly_map = torch.zeros((self.image_size * self.image_size), dtype=anomaly_map.dtype)
                resized_anomaly_map[nonzero_indices_list[img_i]] = anomaly_map.cpu()
                resized_anomaly_map = resized_anomaly_map.reshape(1, 1, self.image_size, self.image_size)
                resized_anomaly_map = KnnBlur(resized_anomaly_map)
                resized_anomaly_map = resized_anomaly_map.reshape(1, self.image_size, self.image_size)
                anomaly_maps_3d_resized.append(torch.tensor(resized_anomaly_map))
            anomaly_maps_3d = torch.cat(anomaly_maps_3d_resized, dim=0).squeeze()

        if self.modality == '2d':
            anomaly_maps = anomaly_maps_2d.cpu()
            anomaly_salient_feats = torch.cat(anomaly_salient_feat_2d, dim=0).cpu()
        elif self.modality == '3d':
            anomaly_maps = anomaly_maps_3d.cpu()
            anomaly_salient_feats = torch.cat(anomaly_salient_feat_3d, dim=0).cpu()
        if self.modality == '2d+3d':
            B, _, _ = anomaly_maps_2d.shape
            anomaly_maps = anomaly_maps_2d.cpu() + anomaly_maps_3d.cpu()
            anomaly_salient_feat_2d = torch.cat(anomaly_salient_feat_2d, dim=0).cpu()
            anomaly_salient_feat_3d = torch.cat(anomaly_salient_feat_3d, dim=0).cpu()
            anomaly_salient_feats = torch.cat([anomaly_salient_feat_2d, anomaly_salient_feat_3d], dim=1)

        # # 1. 取得熱力圖像素矩陣 (確保維度為: 圖片數, 高, 寬)
        # pr_px = np.array(anomaly_maps).squeeze()
        # if pr_px.ndim == 2:
        #     pr_px = np.expand_dims(pr_px, axis=0)

        # # ====== [新增：整張 Wafer 專用圓形 ROI 遮罩 (濾除四角背景與晶圓邊緣反光)] ======
        # wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))
        # if wafer_mask_ratio > 0.0:
        #     H_px, W_px = pr_px.shape[1], pr_px.shape[2]
        #     center = (W_px // 2, H_px // 2)
        #     radius = int(min(H_px, W_px) * 0.5 * wafer_mask_ratio)
            
        #     # 建立圓形遮罩：圓內為 1，圓外(四角背景與晶圓極邊緣)為 0
        #     circle_mask = np.zeros((H_px, W_px), dtype=np.uint8)
        #     cv2.circle(circle_mask, center, radius, 1, thickness=-1)
            
        #     # 將圓外區域的分數統一壓平為整批資料的最低背景值，避免邊框反光霸佔 Anomaly_Score
        #     bg_min_val = pr_px.min()
        #     pr_px[:, circle_mask == 0] = bg_min_val
        #     print(f"已套用整張 Wafer 圓形 ROI 遮罩 (保留中心半徑比例: {wafer_mask_ratio}，有效半徑: {radius} px)")
        # # ==============================================================================

        # # 2. 依論文 Max 定義計算異常分數 (此時已排除晶圓背景干擾)
        # B = pr_px.shape[0]
        # pr_sp = pr_px.reshape(B, -1).max(axis=1)

        # import pandas as pd
        # df_scores = pd.DataFrame({
        #     'Image_Path': image_path_list,
        #     'Anomaly_Score': pr_sp
        # })

        # df_scores = df_scores.sort_values(by='Anomaly_Score', ascending=False)
        # csv_path = os.path.join(self.output_dir, f"{category}_defect_scores.csv")
        # df_scores.to_csv(csv_path, index=False)
        # print(f"已將所有圖片的異常分數排序並儲存至: {csv_path}")

        # # 3. 繪製三合一視覺化圖 (原圖 | 單獨瑕疵圖 | 匡出瑕疵的圖)
        # self.visualization(image_path_list, gt_list, pr_px, category)

        # return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

                # 1. 取得熱力圖像素矩陣
        pr_px = np.array(anomaly_maps).squeeze()
        if pr_px.ndim == 2:
            pr_px = np.expand_dims(pr_px, axis=0)

        # 2. 補回論文標準高斯平滑 (濾除雷射貼紙與表面單點極端反光雜訊)
        from scipy.ndimage import gaussian_filter
        for i in range(pr_px.shape[0]):
            pr_px[i] = gaussian_filter(pr_px[i], sigma=4)

        # 3. 整張 Wafer 專用圓形 ROI 遮罩 (WAFER_MASK_RATIO > 0 時生效)
        wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))
        if wafer_mask_ratio > 0.0:
            H_px, W_px = pr_px.shape[1], pr_px.shape[2]
            center = (W_px // 2, H_px // 2)
            radius = int(min(H_px, W_px) * 0.5 * wafer_mask_ratio)
            circle_mask = np.zeros((H_px, W_px), dtype=np.uint8)
            cv2.circle(circle_mask, center, radius, 1, thickness=-1)
            pr_px[:, circle_mask == 0] = pr_px.min()

        # 4. 計算原始最高分 ac_score，並執行修正對角線歸零問題後的 RsCon 二次計分
        B = pr_px.shape[0]
        ac_score = pr_px.reshape(B, -1).max(axis=1)
        
        if B > 8:
            feats_norm = F.normalize(anomaly_salient_feats, p=2, dim=-1)
            dist_matrix = torch.cdist(feats_norm, feats_norm, p=2).numpy()
            np.fill_diagonal(dist_matrix, np.inf) # 排除自己跟自己比較，解決原本歸零的 Bug
            sorted_indices = np.argsort(dist_matrix, axis=1)
            rs_scores = np.zeros_like(ac_score)
            for k in [1, 7]:
                k_neighbors = sorted_indices[:, :k]
                neighbor_scores = ac_score[k_neighbors].mean(axis=1)
                rs_scores += (ac_score - neighbor_scores * 0.3)
            pr_sp = rs_scores / 2.0
        else:
            pr_sp = ac_score

        # 5. 輸出 CSV 與視覺化圖片
        import pandas as pd
        df_scores = pd.DataFrame({
            'Image_Path': image_path_list,
            'Anomaly_Score': pr_sp
        })

        df_scores = df_scores.sort_values(by='Anomaly_Score', ascending=False)
        csv_path = os.path.join(self.output_dir, f"{category}_defect_scores.csv")
        df_scores.to_csv(csv_path, index=False)
        print(f"已將所有圖片的異常分數排序並儲存至: {csv_path}")

        self.visualization(image_path_list, gt_list, pr_px, category)

        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)




    def main(self):
        for category in self.categories:
            self.make_category_data(category=category)
        print("所有類別檢測完畢！CSV 與視覺化圖片已產出。")
