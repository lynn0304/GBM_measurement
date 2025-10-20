# python train_hrnet.py train \
#   --images hrnet_dataset/train/images \
#   --masks  hrnet_dataset/train/masks \
#   --val-images hrnet_dataset/val/images \
#   --val-masks  hrnet_dataset/val/masks \
#   --epochs 1000 --imgsz 640 --batch 8 --lr 3e-4 \
#   --weights hrnet_result_new/hrnetv2_w18_seg.pt

python train_hrnet.py predict \
  --weights hrnetv2_w18_seg.pt \
  --source slide_1014 \
  --outdir pred_hrnet_1014 \
  --imgsz 640 --overlay \
  --min-obj 300 \
  --hole-area 300 \
  --band-width 2