echo $1	#DEVICES
echo $2 #anomaly_label
echo $3 #anonum
echo $4 #filename
echo $5 #epoch
echo $6 #lambda_r
echo $7 #lambda_dis
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_final110_MNIST_3_d1c3_beta0.5 --trainYn false --anomaly_label $2 --anonum $3 --base_score 0 --filename $4 --lambda_res $6 --lambda_disc $7 --lambda_cdis 0.0 --lambda_ccon 0.0 --sim_num 5 --dis_c_dim 9 --num_con_c 2 --show true

