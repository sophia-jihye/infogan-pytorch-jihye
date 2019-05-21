echo $1	# DEVICE
echo $2 # filename
CUDA_VISIBLE_DEVICESES=$1 python mnist_generate.py --anomaly_label 3 --load_path ./checkpoint/model_epoch100_MNIST_0_d1c2 --filename $2
