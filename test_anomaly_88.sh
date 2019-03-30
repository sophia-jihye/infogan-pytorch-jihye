echo $1	#DEVICES
echo $2 #anomaly_label
echo $3 #anonum
echo $4 #filename
echo $5 #epoch
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_epoch$5_MNIST_$2_d1c2_beta0.5 --trainYn false --anomaly_label $2 --anonum $3 --base_score 0 --filename $4 --lambda_res 1.0 --lambda_disc 1.0 --lambda_cdis 1.0 --lambda_ccon 0.0 --sim_num 50

