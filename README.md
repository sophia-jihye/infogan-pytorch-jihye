### InfoAnoGAN


#### References
* Implementation Tips
    - [Natsu6767/InfoGAN-PyTorch](https://github.com/Natsu6767/InfoGAN-PyTorch)
* Paper
    - [Chen, Xi, et al. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." Advances in neural information processing systems. 2016.](http://papers.nips.cc/paper/6399-infogan-interpretable-representation)
    - [Schlegl, Thomas, et al. "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery." International Conference on Information Processing in Medical Imaging. Springer, Cham, 2017.](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)     

#### Train
> CUDA_VISIBLE_DEVICES=$1 python train.py --trainYn true --load_path ./checkpoint --anomaly_label $2 --dis_c_dim 9 --num_con_c 2

    