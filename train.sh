echo $1
CUDA_VISIBLE_DEVICES=3 python train.py --load_path ./checkpoint --anomaly_label $1