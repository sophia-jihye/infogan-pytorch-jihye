echo $1	#DEVICES
echo $2 #anonum
echo $3 #filename
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_final110_MNIST_0_d9c2 --trainYn false --anomaly_label 0 --anonum $2 --filename $3 --lambda_res 1.0 --lambda_disc 0.0 --lambda_cdis 0.0 --lambda_ccon 0.0 --sim_num 5 --dis_c_dim 9 --num_con_c 2

