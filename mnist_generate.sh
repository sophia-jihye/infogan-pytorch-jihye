echo $1	# DEVICE
echo $2 # filename
CUDA_VISIBLE_DEVICESES=$1 python mnist_generate.py --anomaly_label 0 --load_path ./checkpoint/model_final110_MNIST_0_d9c2 --filename $2
