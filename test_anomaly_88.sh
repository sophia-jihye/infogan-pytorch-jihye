echo $1
echo $2
echo $3
CUDA_VISIBLE_DEVICES=$1 python mnist_anogan.py --load_path ./checkpoint/model_final500_MNIST_$2_d1c2 --trainYn false --anomaly_label $2 --anonum 0.1 --base_score 0 --filename $3

