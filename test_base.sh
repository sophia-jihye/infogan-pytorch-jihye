echo $1
echo $2
python mnist_anogan.py --load_path /home/dmlab/jihye/GIT/InfoGAN-PyTorch/checkpoint/model_final10_MNIST_$1_d1c2 --trainYn true --anomaly_label $1 --basenum 500 --filename $2