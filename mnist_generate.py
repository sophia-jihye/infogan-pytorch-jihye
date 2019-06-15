import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from config import params as params_config
from models.mnist_model import Generator

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

state_dict = torch.load(params_config['load_path'])
params = state_dict['params']
anomaly_label = params_config['anomaly_label']
filename = params_config['filename']
dis_c_dim = params['dis_c_dim']

# Create the generator network.
num_z_c = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
netG = Generator(num_z_c).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
# print(netG)

c = np.linspace(-2, 2, params['dis_c_dim']).reshape(1, -1)
c = np.repeat(c, dis_c_dim, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

dis_c_dim_squared = dis_c_dim * dis_c_dim
zeros = torch.zeros(dis_c_dim_squared, 1, 1, 1, device=device)

idx = np.arange(params['dis_c_dim']).repeat(dis_c_dim)
dis_c = torch.zeros(dis_c_dim_squared, params['dis_c_dim'], 1, 1, device=device)
dis_c[torch.arange(0, dis_c_dim_squared), idx] = 1.0

z = torch.randn(dis_c_dim_squared, 62, 1, 1, device=device)

if(params['num_con_c']==2):
    # Discrete latent code.
    c1 = dis_c.view(dis_c_dim_squared, -1, 1, 1)
    # Continuous latent code.
    c2 = torch.cat((c, zeros), dim=1)
    c3 = torch.cat((zeros, c), dim=1)

    # To see variation along first item (Vertically) and last item (Horizontally).
    noise1 = torch.cat((z, c1, c2), dim=1)
    print('---noise1: ', noise1.shape)

    # Generate image.
    with torch.no_grad():
        generated_img1 = netG(noise1).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=dis_c_dim, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('./result/%d_c12_%s' % (anomaly_label, filename))
    plt.close('all')

    # To see variation along c3 (Horizontally) and c1 (Vertically)
    noise2 = torch.cat((z, c1, c3), dim=1)
    print('---noise2: ', noise1.shape)

    # Generate image.
    with torch.no_grad():
        generated_img2 = netG(noise2).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=dis_c_dim, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('./result/%d_c13_%s' % (anomaly_label, filename))
    plt.close('all')
elif(params['num_con_c']==3):
    # Discrete latent code.
    c1 = dis_c.view(dis_c_dim_squared, -1, 1, 1)
    # Continuous latent code.
    c2 = torch.cat((c, zeros, zeros), dim=1)
    c3 = torch.cat((zeros, c, zeros), dim=1)
    c4 = torch.cat((zeros, zeros, c), dim=1)

    # To see variation along first item (Vertically) and last item (Horizontally).
    noise1 = torch.cat((z, c1, c2), dim=1)
    print('---noise1: ', noise1.shape)

    # Generate image.
    with torch.no_grad():
        generated_img1 = netG(noise1).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=dis_c_dim, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('./result/%d_c12_%s' % (anomaly_label, filename))
    plt.close('all')

    # To see variation along c3 (Horizontally) and c1 (Vertically)
    noise2 = torch.cat((z, c1, c3), dim=1)
    print('---noise2: ', noise1.shape)

    # Generate image.
    with torch.no_grad():
        generated_img2 = netG(noise2).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=dis_c_dim, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('./result/%d_c13_%s' % (anomaly_label, filename))
    plt.close('all')

    # To see variation along c4 (Horizontally) and c1 (Vertically)
    noise3 = torch.cat((z, c1, c4), dim=1)
    print('---noise3: ', noise3.shape)

    # Generate image.
    with torch.no_grad():
        generated_img3 = netG(noise3).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img3, nrow=dis_c_dim, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig('./result/%d_c14_%s' % (anomaly_label, filename))
    plt.close('all')
