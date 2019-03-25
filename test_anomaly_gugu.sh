echo $1
echo $2
echo $3
echo $4
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_epoch100_MNIST_$2_d1c2 --trainYn false --anomaly_label $2 --anonum 0.002 --base_score $3 --filename $4

