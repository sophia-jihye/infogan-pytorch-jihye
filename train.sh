echo $1
echo $2
CUDA_VISIBLE_DEVICES=$1 python train.py --trainYn true --load_path ./checkpoint --anomaly_label $2 --dis_c_dim 8 --num_con_c 3
