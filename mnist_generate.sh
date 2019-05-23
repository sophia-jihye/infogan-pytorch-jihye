echo $1	# DEVICE
echo $2 # filename
echo $3 # model_name
CUDA_VISIBLE_DEVICESES=$1 python mnist_generate.py --anomaly_label 0 --load_path ./checkpoint/$3 --filename $2
