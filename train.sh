echo $1	# DEVICE
echo $2 # ANOMALY_LABEL
echo $3 #number of continuous latent variables
CUDA_VISIBLE_DEVICES=$1 python train.py --trainYn true --load_path ./checkpoint --anomaly_label $2 --dis_c_dim 9 --num_con_c $3
