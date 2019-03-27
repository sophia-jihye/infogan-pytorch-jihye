echo $1	#DEVICES
echo $2 #anomaly_label
echo $3 #epoch
echo $4 #anonum
echo $5 #filename
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_epoch$3_MNIST_$2_d1c2_beta0.5 --trainYn false --anomaly_label $2 --anonum $4 --base_score 0 --filename $5

