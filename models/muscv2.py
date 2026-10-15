import os
import sys
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

         # 1. 從設定檔取得您傳入的完整資料路徑 (例如: ./data/categorized_dataset/TMUN87/)
         raw_path = cfg['datasets']['data_path']

         # 2. 用 strip('/') 去除字串最右邊可能多出來的斜線
         # 然後用 split('/') 把字串依照斜線切成陣列，並用 [-1] 抓取最後一個元素 (即 TMUN87)
         dataset_id = raw_path.strip('/').split('/')[-1]

         # 3. 將抓出來的名稱 dataset_id 加到路徑的最尾端
         self.output_dir = os.path.join(cfg['testing']['output_dir'], self.dataset, '-'.join(self.model_name), 'imagesize{}'.format(self.image_size), dataset_id) + '/'

         os.makedirs(self.output_dir, exist_ok=True)

         # 1. 取得資料集名稱 (例如 TMUN87)
        raw_path = cfg['datasets']['data_path']
        dataset_id = raw_path.strip('/').split('/')[-1]

        # 2. 強制設定為台北時間 (UTC+8)
        import datetime
        taipei_tz = datetime.timezone(datetime.timedelta(hours=8))
        current_time = datetime.datetime.now(taipei_tz).strftime("%Y%m%d_%H%M%S")

        # 3. 組合出包含「資料集名稱_台北時間」的輸出資料夾
        self.image_size = cfg['datasets']['img_resize']
        self.features_list = [l+1 for l in cfg['models']['feature_layers']]
        self.r_list = cfg['models']['r_list']

        # 優先讀取 .sh 腳本建立好的【資料集名稱_台北時間】資料夾路徑，確保 .txt 與圖表 100% 放在同一個資料夾
        if 'EXP_OUTPUT_DIR' in os.environ:
            self.output_dir = os.environ['EXP_OUTPUT_DIR']
        else:
            # 防呆備案：如果沒有透過 .sh 腳本執行，才由 Python 自行建立路徑
            raw_path = cfg['datasets']['data_path']
            dataset_id = raw_path.strip('/').split('/')[-1]
            import datetime
            taipei_tz = datetime.timezone(datetime.timedelta(hours=8))
            current_time = datetime.datetime.now(taipei_tz).strftime("%Y%m%d_%H%M%S")
            folder_name = f"{dataset_id}_{current_time}"
            self.output_dir = os.path.join(cfg['testing']['output_dir'], self.dataset, '-'.join(self.model_name), 'imagesize{}'.format(self.image_size), folder_name) + '/'

        os.makedirs(self.output_dir, exist_ok=True)


    # def load_backbone(self):
    #     if self.modality in ['2d', '2d+3d']:
    #         self.dino_model = _backbones.load(self.model_name[0])
    #         self.dino_model.to(self.device)
    #     if self.modality in ['3d', '2d+3d']:
    #         from models.backbone.point_transformer import PointTransformer
    #         self.point_transformer = PointTransformer(group_size=128, num_group=1024).to(self.device)
    #         self.point_transformer.load_model_from_ckpt('./pointmae_pretrain.pth')

    #改
    def load_backbone(self):
        if self.modality in ['2d', '2d+3d']:
            # 從作者組裝好的 vision_transformer 檔案匯入 vit_large
            from models.backbone.dino_vision_transformer import vit_large
            
            # 1. 建立 DINOv2 Large 模型
            # 注意：這份程式碼沒有支援 num_register_tokens 參數，所以要拿掉
            self.dino_model = vit_large(patch_size=14, img_size=518)
           
            # 2. 讀取權重檔
            weight = torch.load('./dinov2_vitl14_reg4_pretrain.pth', map_location='cpu')
            
            # 3. 將權重寫入模型 (strict=False 可以略過 registers 等不匹配的權重)
            self.dino_model.load_state_dict(weight, strict=False)

            # 4. 放到 GPU 
            self.dino_model.to(self.device)

            if self.modality in ['3d', '2d+3d']:
            from models.backbone.point_transformer import PointTransformer
            self.point_transformer = PointTransformer(group_size=128, num_group=1024).to(self.device)
            self.point_transformer.load_model_from_ckpt('./pointmae_pretrain.pth')
    #改
            
            
    def load_datasets(self, category='', get_categories=False):
        # dataloader
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

        #改
        # --- 新增這段 MvTEC-2 讀取邏輯 ---
        elif self.dataset == 'mvtec_2':
            if get_categories: 
                return ['can']
                
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
        # -----------------------------------
        
        #改
        # --- 新增這段 MvTEC-2 讀取邏輯 ---
        elif self.dataset == 'TMUN87':
            if get_categories: 
                return ['TMUN87']

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
        # -----------------------------------


        return test_dataset


    def visualization(self, image_path_list, gt_list, pr_px, category):
        import cv2
        
        # 防呆：確保只有一張圖時，維度也是 (數量, 高, 寬)
        if pr_px.ndim == 2:
            pr_px = np.expand_dims(pr_px, axis=0)
            
        # 1. 取得整批資料的最大與最小分數，用來將熱力圖正規化到 0~255
        max_score = pr_px.max()
        min_score = pr_px.min()

        # ==========================================
        # 2. 自動從 .sh 腳本讀取 Threshold 與 Alpha (若未設定則預設為 0.35 與 0.6)
        # ==========================================
        threshold = float(os.environ.get('THRESHOLD', 0.35)) # 瑕疵門檻值：建議設在你的正常背景值(0.234)與最高分(0.500)之間
        alpha = float(os.environ.get('ALPHA', 0.6)) # 瑕疵凸顯透明度：0.0 為完全透明，1.0 為完全不透明 (建議 0.5 ~ 0.7)
        print(f"正在產生視覺化圖片 (Threshold={threshold}, Alpha={alpha})...")
        
        # 設定圖片輸出的資料夾路徑
        save_path = os.path.join(self.output_dir, 'vis', category)
        os.makedirs(save_path, exist_ok=True)

        for i, img_path in enumerate(image_path_list):
            # --- 【第 1 張：原圖】 ---
            orig_img = cv2.imread(img_path)
            orig_img = cv2.resize(orig_img, (self.image_size, self.image_size))

            # 取得單張圖片的分數矩陣，並確保尺寸與原圖一致
            score_map = cv2.resize(pr_px[i], (self.image_size, self.image_size))

            # --- 【第 3 張：Heatmap 原圖】 ---
            # 將分數轉換為 0~255 的色階，並套用 JET 色彩 (藍->綠->黃->紅)
            norm_map = (score_map - min_score) / (max_score - min_score + 1e-8)
            norm_map_uint8 = np.uint8(norm_map * 255)
            raw_heatmap = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)
            
            # (選用) 如果你希望「Heatmap 原圖」也帶有一點點原圖輪廓，可以把下面這行取消註解：
            # raw_heatmap = cv2.addWeighted(orig_img, 0.3, raw_heatmap, 0.7, 0)

            # --- 【第 2 張：設定 Threshold 的半透明瑕疵凸顯圖】 ---
            # 先複製一份乾淨的原圖當底
            thresh_overlay = orig_img.copy()
            # 計算原圖與熱力圖依照 alpha 比例混合後的畫面
            blended = cv2.addWeighted(orig_img, 1 - alpha, raw_heatmap, alpha, 0)

            # 找出分數超過 threshold 的像素位置
            defect_mask = score_map >= threshold
            # 只有超過 threshold 的異常位置才塗上半透明熱力圖，其餘維持清晰原圖
            thresh_overlay[defect_mask] = blended[defect_mask]
            
            # ==========================================
            # 3. 儲存圖片 (預設將三張圖左右並排拼接成一張，最方便報告對照)
            # ==========================================
            img_name = os.path.basename(img_path)
            name_no_ext, ext = os.path.splitext(img_name)

            # 方式 A：將「左：原圖 | 中：瑕疵凸顯圖 | 右：Heatmap原圖」左右拼成一張圖存檔
            combined_img = np.hstack((orig_img, thresh_overlay, raw_heatmap))
            cv2.imwrite(os.path.join(save_path, f"{name_no_ext}_comparison{ext}"), combined_img)
            
            # 方式 B：如果你希望三張圖各自獨立存成三個檔案，可以把下面三行取消註解：
            # cv2.imwrite(os.path.join(save_path, f"{name_no_ext}_1_orig{ext}"), orig_img)
            # cv2.imwrite(os.path.join(save_path, f"{name_no_ext}_2_thresh{ext}"), thresh_overlay)
            # cv2.imwrite(os.path.join(save_path, f"{name_no_ext}_3_heatmap{ext}"), raw_heatmap)
          
        print(f"視覺化完成！圖片已儲存至: {save_path}")
            

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
        #改
        print(f"資料集總共讀到 {len(test_dataset)} 張圖片！")

        # extract features
        print('Features extracting...')
        for image_info in tqdm(test_dataloader):
            image = image_info["image"]
            image_path_list.extend(image_info["image_path"])
            img_masks.append(image_info["mask"])
            gt_list.extend(list(image_info["is_anomaly"].numpy()))
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.modality != '3d':
                    # 2D modal feature extraction
                    input_image = image.to(torch.float).to(self.device)evice)
                  
                    # 1. 取得特徵並強制要求回傳 cls_token
                    patch_tokens_raw = self.dino_model.get_intermediate_layers(x=input_image, n=max(self.features_list), return_class_token=True)

                    # 2. 將 cls_token 重新拼接到 patch_tokens 的最前面，還原成 [B, 257, 1024] 的形狀
                    patch_tokens_all = [torch.cat((cls_tok.unsqueeze(1), p_tok), dim=1) for p_tok, cls_tok in patch_tokens_raw]
                    
                    # 3. 提取指定的特徵層
                    patch_tokens = [patch_tokens_all[l-1] for l in self.features_list]
                    patch_tokens_list.append(patch_tokens) # (B, L+1, C)
                if self.modality != '2d':
                    # 3D modal feature extraction
                    organized_pc = image_info['point_cloud'].to(self.device)
                    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0)
                    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
                    # remove the background
                    nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1), as_tuple=False).squeeze()
                    xyz = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1).contiguous()  # (B, 3, N)
                    pc_list.append(xyz)
                    point_tokens, center, ori_idx, center_idx, curvatures = self.point_transformer(xyz)
                    curvatures_list.append(curvatures)  # curvature
                    pc_center_list.append(center)   # center point
                    nonzero_indices_list.append(nonzero_indices)    # foreground points
                    nonzero_indices_center_list.append(nonzero_indices[center_idx.squeeze()].unsqueeze(0))  # foreground center points
                    nonzero_indices_center_neighbor = nonzero_indices[ori_idx.squeeze()]    # neighborhood points of each center point
                    nonzero_indices_center_neighbor_list.append(nonzero_indices_center_neighbor.unsqueeze(0))
                    point_tokens_list.append(point_tokens)

        # Similar Neighborhood Aggregation with Multiple Degrees (SNAMD)
        print('Features aggregating...')
        SNAMD_r = SNAMD(device=self.device, r_list=self.r_list)
        Z_2d_layers = {}
        Z_3d_layers = {}
        if self.modality != '3d':
            # aggregate 2D features
            for im in range(len(patch_tokens_list)):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    patch_tokens = patch_tokens_list[im]
                    features = SNAMD_r._embed_2d(patch_tokens) # (B, L, layer, C)
                    features /= features.norm(dim=-1, keepdim=True)
                    for l in range(features.shape[2]):
                        if str(l) not in Z_2d_layers.keys():
                            Z_2d_layers[str(l)] = []
                        Z_2d_layers[str(l)].append(features[:, :, l, :])
        if len(point_tokens_list) != 0:
            # aggregate 3D features
            for im in range(len(point_tokens_list)):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    patch_tokens = point_tokens_list[im]
                    patch_tokens = [p/p.norm(dim=-1, keepdim=True) for p in patch_tokens]
                    center = pc_center_list[im]
                    curvatures = curvatures_list[im]
                    features = SNAMD_r._embed_3d(patch_tokens, center, curvatures)
                    for l in range(features.shape[2]):
                        # save the aggregated features
                        if str(l) not in Z_3d_layers.keys():
                            Z_3d_layers[str(l)] = []
                        Z_3d_layers[str(l)].append(features[:, :, l, :])

        # Multimodal Mutual Scoring
        layer_keys = Z_2d_layers.keys() if self.modality != '3d' else Z_3d_layers.keys()
        if self.modality == '2d+3d':
            # establish the cross-modal project list
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
            # different layers
            print(f'Mutual scoring in layer{l}...')
            if self.modality != '3d':
                Z_2d = torch.cat(Z_2d_layers[l], dim=0).to(self.device)   # (N, L, C)
            if self.modality != '2d':
                Z_3d = torch.cat(Z_3d_layers[l], dim=0).to(self.device) # (N, L, C)
            if self.modality == '2d':
                # mutual scoring for only 2D data
                anomaly_maps_2d_msm = MSM_single(Z=Z_2d, device=self.device)
                anomaly_maps_2d = torch.cat((anomaly_maps_2d, anomaly_maps_2d_msm.unsqueeze(0)), dim=0)
            if self.modality == '3d':
                # mutual scoring for only 3D data
                anomaly_maps_3d_msm = MSM_single(Z=Z_3d, device=self.device)
                anomaly_maps_3d = torch.cat((anomaly_maps_3d, anomaly_maps_3d_msm.unsqueeze(0)), dim=0)
            elif self.modality == '2d+3d':
                # mutual scoring for 2D+3D data
                anomaly_maps_2d_msm, anomaly_maps_3d_msm = MSM_multiple(Z_2d=Z_2d, Z_3d=Z_3d, device=self.device, patch2point_list=patch2point_list, point2patch_list=point2patch_list)
                anomaly_maps_3d = torch.cat((anomaly_maps_3d, anomaly_maps_3d_msm.unsqueeze(0)), dim=0)
                anomaly_maps_2d = torch.cat((anomaly_maps_2d, anomaly_maps_2d_msm.unsqueeze(0)), dim=0)
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
            features_2d = torch.cat(Z_2d_layers[layers_2d[-2]], dim=0)
            amap_max_idx = torch.max(anomaly_maps_2d, dim=-1)[1]
            anomaly_salient_feat_2d.append(features_2d[torch.arange(B), amap_max_idx, :].cpu())
            H = int(np.sqrt(L))
            # interpolate
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
            # interpolate
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

        B = anomaly_maps.shape[0]   # the number of unlabeled test images
        ac_score = np.array(anomaly_maps).reshape(B, -1).max(-1)
        
        #改
        # 取得最原始的異常分數 (不要經過 RsCon 處理)
        B = anomaly_maps.shape[0] 
        ac_score = np.array(anomaly_maps).reshape(B, -1).max(-1)

        # 1. 直接把 ac_score 拿來當作最終預測分數
        pr_sp = ac_score
        
        # 確保陣列格式正確 (單張圖片時的防呆)
        if pr_sp.ndim == 0:
            pr_sp = np.expand_dims(pr_sp, axis=0)

        # 2. 使用基礎的 Pandas 存成 CSV
        import pandas as pd
        df_scores = pd.DataFrame({
            'Image_Path': image_path_list,
            'Anomaly_Score': pr_sp
        })

        # 依分數由高到低排序，排在最上面的就是最有問題的圖片
        df_scores = df_scores.sort_values(by='Anomaly_Score', ascending=False)
        csv_path = os.path.join(self.output_dir, f"{category}_defect_scores.csv")
        df_scores.to_csv(csv_path, index=False)
        print(f"已將所有圖片的異常分數排序並儲存至: {csv_path}")

        # 3. 繪製熱力圖，方便你肉眼確認瑕疵位置
        pr_px = np.array(anomaly_maps).squeeze()
        self.vis_type = 'whole_norm'
        self.visualization(image_path_list, gt_list, pr_px, category)

        # 4. 回傳假的指標數值給主程式，避免報錯
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
        #改


    def main(self):
        auroc_sp_ls = []
        f1_sp_ls = []
        ap_sp_ls = []
        auroc_px_ls = []
        f1_px_ls = []
        ap_px_ls = []
        aupro_ls = []
        for category in self.categories:
            image_metric, pixel_metric = self.make_category_data(category=category,)
            auroc_sp, f1_sp, ap_sp = image_metric
            auroc_px, f1_px, ap_px, aupro = pixel_metric
            auroc_sp_ls.append(auroc_sp)
            f1_sp_ls.append(f1_sp)
            ap_sp_ls.append(ap_sp)
            auroc_px_ls.append(auroc_px)
            f1_px_ls.append(f1_px)
            ap_px_ls.append(ap_px)
            aupro_ls.append(aupro)
        # mean
        auroc_sp_mean = sum(auroc_sp_ls) / len(auroc_sp_ls)
        f1_sp_mean = sum(f1_sp_ls) / len(f1_sp_ls)
        ap_sp_mean = sum(ap_sp_ls) / len(ap_sp_ls)
        auroc_px_mean = sum(auroc_px_ls) / len(auroc_px_ls)
        f1_px_mean = sum(f1_px_ls) / len(f1_px_ls)
        ap_px_mean = sum(ap_px_ls) / len(ap_px_ls)
        aupro_mean = sum(aupro_ls) / len(aupro_ls)

        for i, category in enumerate(self.categories):
            print(category)
            print('image-level, auroc:{}, f1:{}, ap:{}'.format(auroc_sp_ls[i]*100, f1_sp_ls[i]*100, ap_sp_ls[i]*100))
            print('pixel-level, auroc:{}, f1:{}, ap:{}, aupro:{}'.format(auroc_px_ls[i]*100, f1_px_ls[i]*100, ap_px_ls[i]*100, aupro_ls[i]*100))
        print('mean')
        print('image-level, auroc:{}, f1:{}, ap:{}'.format(auroc_sp_mean*100, f1_sp_mean*100, ap_sp_mean*100))
        print('pixel-level, auroc:{}, f1:{}, ap:{}, aupro:{}'.format(auroc_px_mean*100, f1_px_mean*100, ap_px_mean*100, aupro_mean*100))
        
        # save in excel
        if self.save_excel:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "MuSc_results"
            sheet.cell(row=1,column=2,value='auroc_px')
            sheet.cell(row=1,column=3,value='f1_px')
            sheet.cell(row=1,column=4,value='ap_px')
            sheet.cell(row=1,column=5,value='aupro')
            sheet.cell(row=1,column=6,value='auroc_sp')
            sheet.cell(row=1,column=7,value='f1_sp')
            sheet.cell(row=1,column=8,value='ap_sp')
            for col_index in range(2):
                for row_index in range(len(self.categories)):
                    if col_index == 0:
                        sheet.cell(row=row_index+2,column=col_index+1,value=self.categories[row_index])
                    else:
                        sheet.cell(row=row_index+2,column=col_index+1,value=auroc_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+2,value=f1_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+3,value=ap_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+4,value=aupro_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+5,value=auroc_sp_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+6,value=f1_sp_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+7,value=ap_sp_ls[row_index]*100)
                    if row_index == len(self.categories)-1:
                        if col_index == 0:
                            sheet.cell(row=row_index+3,column=col_index+1,value='mean')
                        else:
                            sheet.cell(row=row_index+3,column=col_index+1,value=auroc_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+2,value=f1_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+3,value=ap_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+4,value=aupro_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+5,value=auroc_sp_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+6,value=f1_sp_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+7,value=ap_sp_mean*100)
            print(self.output_dir)
            workbook.save(os.path.join(self.output_dir, 'results.xlsx'))
