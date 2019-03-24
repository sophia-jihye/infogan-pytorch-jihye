echo $1
echo $2
CUDA_VISIBLE_DEVICES=$1 python train.py --trainYn true --load_path ./checkpoint --anomaly_label $2
