echo $1	# DEVICE
echo $2 # epoch
CUDA_VISIBLE_DEVICESES=$1 python mnist_generate.py --anomaly_label 0 --load_path ./checkpoint/model_epoch$2_MNIST_0_d1c2_beta0.5 --filename e$2
