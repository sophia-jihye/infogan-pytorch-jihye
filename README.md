## InfoAnoGAN


### References
* Implementation Tips
    - [Natsu6767/InfoGAN-PyTorch](https://github.com/Natsu6767/InfoGAN-PyTorch)
* Paper
    - [Chen, Xi, et al. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." Advances in neural information processing systems. 2016.](http://papers.nips.cc/paper/6399-infogan-interpretable-representation)
    - [Schlegl, Thomas, et al. "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery." International Conference on Information Processing in Medical Imaging. Springer, Cham, 2017.](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)     

### Train: Refer to `./train.sh`
> CUDA_VISIBLE_DEVICES=$DEVICENUM python train.py --trainYn true --load_path ./checkpoint --anomaly_label $ANOMALYLABEL --dis_c_dim 9 --num_con_c 2

* Results after Training
    - `./result/$ANOMALYLABEL_epoch$EPOCHNUM_Training_Images.png` : Raw images for training except for anomaly label.
    - `./result/$ANOMALYLABEL-Epoch$EPOCHNUM.png` : $EPOCHNUM includes 1, $save_epoch in config.py. Generated images during training.
    - `./result/Epoch_$EPOCHNUM_MNIST.png` : Generated images from the model which completed total training procedures.
    - `./result/Loss Curve_$ANOMALYLABEL.png` : Generator and discriminator loss during training.
        - D_loss = loss_real + loss_fake
        - G_loss = gen_loss + discrete_variable_loss + continuous_variable_loss
    - `./checkpoint/model_epoch$EPOCHNUM_MNIST_$ANOMALYLABEL_d$DISCDIMc$CONTINUOUSVARNUM` : $EPOCHNUM includes $save_epoch in config.py. Trained model during training.
    - `./checkpoint/model_final$EPOCHNUM_MNIST_$ANOMALYLABEL_d$DISCDIMc$CONTINUOUSVARNUM` : Trained model which completed total training procedures.
  
<br>
    
### Semantic Features of Training dataset: Refer to `./mnist_generate.sh`
> CUDA_VISIBLE_DEVICESES=$DEVICENUM python mnist_generate.py --anomaly_label $ANOMALYLABEL --load_path ./checkpoint/$MODELNAME --filename $FILENAME

* Results for Semantic Feature Representation (In case of 1 discrete variable and 2 continuous variable)
    - `./result/$ANOMALYLABEL_c12_$FILENAME.png` : y-axis represents c1 (discrete) variable, x-axis represents c2 (continuous) variable.
    - `./result/$ANOMALYLABEL_c13_$FILENAME.png` : y-axis represents c1 (discrete) variable, x-axis represents c3 (continuous) variable.
    
### Anomaly Detection: Refer to `./test_anomaly.sh`
> CUDA_VISIBLE_DEVICES=$DEVICENUM python mnist_anogan.py --load_path ./checkpoint/$MODELNAME --trainYn false --anomaly_label $ANOMALYLABEL --anonum $ANONUM --filename $FILENAME --lambda_res 1.0 --lambda_disc 0.0 --lambda_cdis 0.0 --lambda_ccon 0.0 --sim_num 5 --dis_c_dim 9 --num_con_c 2

* $ANONUM : 0.001 ~ 1
* Results after Anomaly Detection
    - `./result/$ANOMALYLABEL-$ANONUM-$FILENAME_prc.png`
