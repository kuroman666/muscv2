device=0
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py \
--device 0 \
--data_path ./data/mvtec-2 \
--dataset_name mvtec_2 \
--class_name can \
--backbone_name dino_vitbase8 \
--feature_layers 3 7 11 \
--img_resize 224 \
--r_list 1 3 5 \
--output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm