echo $1	#DEVICES
echo $2 #anomaly_label
echo $3 #anonum
echo $4 #filename
echo $5 #epoch
echo $6 #dis_c_dim
echo $7 #num_con_c
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_epoch$5_MNIST_$2_d1c2_beta0.5 --trainYn false --anomaly_label $2 --anonum $3 --base_score 0 --filename $4 --lambda_res 1.0 --lambda_disc 0.0 --lambda_cdis 0.0 --lambda_ccon 0.0 --sim_num 50 --dis_c_dim $6 --num_con_c $7

