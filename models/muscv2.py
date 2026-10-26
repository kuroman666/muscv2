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
from datasets.mvtec2 import preprocess_wafer_image, MVTec2Dataset
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
            self.dino_model = vit_large(patch_size=14, img_size=518)
            weight = torch.load('./dinov2_vitl14_reg4_pretrain.pth', map_location='cpu')
            self.dino_model.load_state_dict(weight, strict=False)
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
            from torchvision.transforms import InterpolationMode

            my_transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
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
        def normalization01(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        if pr_px.ndim == 2:
            pr_px = np.expand_dims(pr_px, axis=0)

        # norm_pr_px 僅用來產生 0~255 的鮮豔熱力圖色彩 (COLORMAP_JET)
        norm_pr_px = pr_px.copy()
        if self.vis_type == 'single_norm':
            for i in range(norm_pr_px.shape[0]):
                norm_pr_px[i] = normalization01(norm_pr_px[i])
        elif self.vis_type == 'whole_norm':
            norm_pr_px = normalization01(norm_pr_px)

        # 讀取單一門檻 THRESHOLD (直接對應 CSV 的 Anomaly_Score 尺度！)
        threshold = float(os.environ.get('THRESHOLD', 0.35))
        alpha = float(os.environ.get('ALPHA', 0.6))
        wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))

        # 統計有多少張圖片的最高分超過此 THRESHOLD (即會被畫上紅框的圖片數)
        img_max_scores = pr_px.reshape(pr_px.shape[0], -1).max(axis=1)
        flagged_count = int((img_max_scores >= threshold).sum())
        print(f"正在產生視覺化圖片 [單一門檻 Threshold={threshold} (直接對應 CSV Anomaly_Score) | 標註異常圖數: {flagged_count}/{len(image_path_list)} 張]...", flush=True)
        if flagged_count == 0:
            print(f"[提醒] 目前設定的 THRESHOLD={threshold} 高於本批資料最高分 ({img_max_scores.max():.4f})，不會畫出任何紅框！請參考上方統計建議值調降。", flush=True)

        save_path = os.path.join(self.output_dir, 'vis', category)
        os.makedirs(save_path, exist_ok=True)

        for i, img_path in enumerate(image_path_list):
            # --- 【第 1 張：原圖】 ---
            raw_bgr = cv2.imread(img_path)
            if self.dataset in ['mvtec_2', 'TMUN87']:
                raw_bgr = preprocess_wafer_image(raw_bgr, ref_template=MVTec2Dataset.ref_template)
            orig_img = cv2.resize(raw_bgr, (self.image_size, self.image_size))

            # 取得上色用的 [0, 1] 矩陣 norm_map，以及與 CSV Anomaly_Score 同尺度的分數矩陣 raw_map
            norm_map = norm_pr_px[i]
            raw_map = pr_px[i]
            if norm_map.shape[0] != self.image_size:
                norm_map = cv2.resize(norm_map, (self.image_size, self.image_size))
                raw_map = cv2.resize(raw_map, (self.image_size, self.image_size))

            norm_map_uint8 = np.uint8(norm_map * 255)
            paper_heatmap = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)

            # [關鍵修改] 直接使用與 CSV Anomaly_Score 同尺度的 raw_map 判斷是否超過門檻！
            # [升級版] 連通域遲滯門檻 (Hysteresis Thresholding)：
            # 僅當連通區塊的最高分 >= threshold 時，才完整保留該瑕疵延伸至 0.90*threshold 的山腰輪廓，防止長條刮痕斷成碎框！
            defect_mask = np.zeros_like(raw_map, dtype=bool)
            if raw_map.max() >= threshold:
                num_labels, labels = cv2.connectedComponents(np.uint8(raw_map >= threshold * 0.90))
                for lbl in range(1, num_labels):
                    if raw_map[labels == lbl].max() >= threshold:
                        defect_mask[labels == lbl] = True


            # --- 【第 2 張：單獨瑕疵圖】 ---
            isolated_defect_img = np.zeros_like(orig_img)
            blended_defect = cv2.addWeighted(orig_img, 1 - alpha, paper_heatmap, alpha, 0)
            isolated_defect_img[defect_mask] = blended_defect[defect_mask]

            # --- 【第 3 張：匡出瑕疵的圖】 ---
            boxed_defect_img = orig_img.copy()
            mask_uint8 = np.uint8(defect_mask * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(boxed_defect_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if wafer_mask_ratio > 0.0:
                center = (self.image_size // 2, self.image_size // 2)
                radius = int(self.image_size * 0.5 * wafer_mask_ratio)
                cv2.circle(boxed_defect_img, center, radius, (0, 255, 0), 1, cv2.LINE_AA)

            img_name = os.path.basename(img_path)
            name_no_ext, ext = os.path.splitext(img_name)
            rel_dir = os.path.relpath(os.path.dirname(img_path), self.path)
            safe_prefix = rel_dir.replace(os.sep, '_').strip('.')

            combined_img = np.hstack((orig_img, isolated_defect_img, boxed_defect_img))
            cv2.imwrite(os.path.join(save_path, f"{safe_prefix}_{name_no_ext}_comparison{ext}"), combined_img)

        print(f"視覺化完成！三合一圖片已儲存至: {save_path}", flush=True)


    def make_category_data(self, category):
        print(f'Category: {category}', flush=True)
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

        print(f"資料集總共讀到 {len(test_dataset)} 張圖片！", flush=True)

        # 1. Extract features (已修復縮排：確保所有圖片都抽完特徵後才釋放 dino_model)
        print('Features extracting...', flush=True)
        for image_info in tqdm(test_dataloader):
            image = image_info["image"]
            image_path_list.extend(image_info["image_path"])
            img_masks.append(image_info["mask"].cpu())
            gt_list.extend(list(image_info["is_anomaly"].numpy()))
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.modality != '3d':
                    input_image = image.to(torch.float).to(self.device)
                    patch_tokens_all = self.dino_model.get_intermediate_layers(x=input_image, n=max(self.features_list))

                    patch_tokens = []
                    for l in self.features_list:
                        pt = patch_tokens_all[l-1].cpu()
                        if int(np.sqrt(pt.shape[1])) ** 2 == pt.shape[1]:
                            dummy_cls = torch.zeros((pt.shape[0], 1, pt.shape[2]), dtype=pt.dtype)
                            pt = torch.cat([dummy_cls, pt], dim=1)
                        patch_tokens.append(pt)

                    patch_tokens_list.append(patch_tokens)

                if self.modality != '2d':
                    organized_pc = image_info['point_cloud'].to(self.device)
                    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0)
                    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
                    nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1), as_tuple=False).squeeze()
                    xyz = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1).contiguous()
                    pc_list.append(xyz)
                    point_tokens, center, ori_idx, center_idx, curvatures = self.point_transformer(xyz)
                    curvatures_list.append(curvatures)
                    pc_center_list.append(center)
                    nonzero_indices_list.append(nonzero_indices)
                    nonzero_indices_center_list.append(nonzero_indices[center_idx.squeeze()].unsqueeze(0))
                    nonzero_indices_center_neighbor = nonzero_indices[ori_idx.squeeze()]
                    nonzero_indices_center_neighbor_list.append(nonzero_indices_center_neighbor.unsqueeze(0))
                    point_tokens_list.append(point_tokens)

        if self.modality in ['2d', '2d+3d']:
            del self.dino_model
            torch.cuda.empty_cache()

        # 2. Similar Neighborhood Aggregation with Multiple Degrees (SNAMD) (已修復迴圈縮排)
        print('Features aggregating...', flush=True)
        SNAMD_r = SNAMD(device=self.device, r_list=self.r_list)
        Z_2d_layers = {}
        Z_3d_layers = {}
        if self.modality != '3d':
            for im in range(len(patch_tokens_list)):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    patch_tokens = [p.to(self.device) for p in patch_tokens_list[im]]
                    features = SNAMD_r._embed_2d(patch_tokens)
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

        # 3. Multimodal Mutual Scoring (已修復 torch.mean 縮排位置)
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
            print(f'Mutual scoring in layer{l}...', flush=True)
            if self.modality != '3d':
                Z_2d = torch.cat([z.to(self.device) for z in Z_2d_layers[l]], dim=0)
            if self.modality != '2d':
                Z_3d = torch.cat([z.to(self.device) for z in Z_3d_layers[l]], dim=0)
            if self.modality == '2d':
                anomaly_maps_2d_msm = MSM_single(Z=Z_2d, device=self.device)
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

        # 4. Calculate anomaly-salient features
        anomaly_salient_feat_2d = []
        anomaly_salient_feat_3d = []
        if self.modality in ['2d', '2d+3d']:
            B, L = anomaly_maps_2d.shape
            H = int(np.sqrt(L))
            layers_2d = list(Z_2d_layers.keys())
            features_2d = torch.cat([z.to(self.device) for z in Z_2d_layers[layers_2d[-2]]], dim=0)

            # [新增修正] 若有設定 WAFER_MASK_RATIO，在挑選 amap_max_idx 前先排除圓外區域，確保 RsCon 抽到圓內瑕疵特徵
            wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))
            if wafer_mask_ratio > 0.0:
                grid_mask = np.zeros((H, H), dtype=np.uint8)
                cv2.circle(grid_mask, (H // 2, H // 2), int(H * 0.5 * wafer_mask_ratio), 1, thickness=-1)
                valid_flat = torch.from_numpy(grid_mask.reshape(-1) == 1).to(self.device)
                masked_amap = anomaly_maps_2d.clone()
                masked_amap[:, ~valid_flat] = masked_amap.min()
                amap_max_idx = torch.max(masked_amap, dim=-1)[1]
            else:
                amap_max_idx = torch.max(anomaly_maps_2d, dim=-1)[1]

            batch_idx = torch.arange(B, device=self.device)
            anomaly_salient_feat_2d.append(features_2d[batch_idx, amap_max_idx, :].cpu())
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
                anomaly_map = interpolating_points(pc_list[img_i].permute(0, 2, 1), pc_center_list[img_i], anomaly_map)
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

        # 5. 嚴格遵循原論文後處理流程 (僅先遮除晶圓圓外背景避免高斯模糊暈染，不額外竄改圓內像素分佈)
        pr_px = np.array(anomaly_maps).squeeze()
        if pr_px.ndim == 2:
            pr_px = np.expand_dims(pr_px, axis=0)

        B, H_px, W_px = pr_px.shape
        wafer_mask_ratio = float(os.environ.get('WAFER_MASK_RATIO', 0.0))

        circle_mask = np.ones((H_px, W_px), dtype=np.uint8)
        if wafer_mask_ratio > 0.0:
            center = (W_px // 2, H_px // 2)
            radius = int(min(H_px, W_px) * 0.5 * wafer_mask_ratio)
            circle_mask.fill(0)
            cv2.circle(circle_mask, center, radius, 1, thickness=-1)
            for i in range(B):
                pr_px[i, circle_mask == 0] = np.median(pr_px[i, circle_mask == 1])

        from scipy.ndimage import gaussian_filter
        for i in range(B):
            pr_px[i] = gaussian_filter(pr_px[i], sigma=4)

        if wafer_mask_ratio > 0.0:
            pr_px[:, circle_mask == 0] = pr_px.min()

        # 6. 原論文 RsCon 二次計分
        ac_score = pr_px.reshape(B, -1).max(axis=1)

        if B > 8:
            feats_norm = F.normalize(anomaly_salient_feats, p=2, dim=-1)
            dist_matrix = torch.cdist(feats_norm, feats_norm, p=2).numpy()
            np.fill_diagonal(dist_matrix, np.inf)
            sorted_indices = np.argsort(dist_matrix, axis=1)
            rs_scores = np.zeros_like(ac_score)
            for k in [1, 7]:
                k_neighbors = sorted_indices[:, :k]
                neighbor_scores = ac_score[k_neighbors].mean(axis=1)
                rs_scores += (ac_score - neighbor_scores * 0.3)
            pr_sp = rs_scores / 2.0
        else:
            pr_sp = ac_score

        # [關鍵同步] 將 RsCon 的計分比例同步映射回像素熱力圖 pr_px，使每張圖的最大像素值 100% 等於 CSV 的 Anomaly_Score (pr_sp)
        scale_factor = (pr_sp / (ac_score + 1e-8)).reshape(B, 1, 1)
        pr_px = pr_px * scale_factor

        # 印出無 Ground Truth 時的統計分布與建議門檻，方便直接參考設定 THRESHOLD
        s_mean, s_std = np.mean(pr_sp), np.std(pr_sp)
        p95, p90 = np.percentile(pr_sp, 95), np.percentile(pr_sp, 90)
        print("----------------------------------------", flush=True)
        print(f"【Anomaly_Score 統計分佈與無監督門檻建議】", flush=True)
        print(f" - 分數範圍 (Min ~ Max) : {pr_sp.min():.4f} ~ {pr_sp.max():.4f} (平均 Mean={s_mean:.4f}, 標準差 Std={s_std:.4f})", flush=True)
        print(f" - 建議門檻 1 [統計離群 Mean + 1.5*Std] : {s_mean + 1.5*s_std:.4f}", flush=True)
        print(f" - 建議門檻 2 [嚴格離群 Mean + 2.0*Std] : {s_mean + 2.0*s_std:.4f}", flush=True)
        print(f" - 建議門檻 3 [預估前  5% 異常 (P95)]   : {p95:.4f}", flush=True)
        print(f" - 建議門檻 4 [預估前 10% 異常 (P90)]   : {p90:.4f}", flush=True)
        print("----------------------------------------", flush=True)

        # 7. 輸出 CSV 與視覺化圖片
        import pandas as pd
        df_scores = pd.DataFrame({
            'Image_Path': image_path_list,
            'Anomaly_Score': pr_sp
        })

        df_scores = df_scores.sort_values(by='Anomaly_Score', ascending=False)
        csv_path = os.path.join(self.output_dir, f"{category}_defect_scores.csv")
        df_scores.to_csv(csv_path, index=False)
        print(f"已將所有圖片的異常分數排序並儲存至: {csv_path}", flush=True)

        if self.vis:
            self.visualization(image_path_list, gt_list, pr_px, category)

        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)


    def main(self):
        for category in self.categories:
            self.make_category_data(category=category)
        print("所有類別檢測完畢！CSV 與視覺化圖片已產出。", flush=True)
