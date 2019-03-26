echo $1
echo $2
echo $3
python mnist_generate.py --anomaly_label $1 --load_path ./checkpoint/model_epoch$2_MNIST_$1_d1c2 --filename $3
