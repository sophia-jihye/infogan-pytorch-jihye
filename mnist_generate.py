import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from config import params as params_config

from models.mnist_model import Generator

# Load the checkpoint file
state_dict = torch.load(params_config['load_path'])

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

anomaly_label = params_config['anomaly_label']
filename = params_config['filename']
temp_dim = params['dis_c_dim']


# Create the generator network.
num_z_c = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
netG = Generator(num_z_c).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

c = np.linspace(-2, 2, params['dis_c_dim']).reshape(1, -1)
c = np.repeat(c, temp_dim, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

temp_100 = temp_dim * params['dis_c_dim']
zeros = torch.zeros(temp_100, 1, 1, 1, device=device)

idx = np.arange(params['dis_c_dim']).repeat(temp_dim)
dis_c = torch.zeros(temp_100, params['dis_c_dim'], 1, 1, device=device)
dis_c[torch.arange(0, temp_100), idx] = 1.0

z = torch.randn(temp_100, 62, 1, 1, device=device)

# Discrete latent code.
c1 = dis_c.view(temp_100, -1, 1, 1)
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
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=temp_dim, padding=2, normalize=True), (1, 2, 0)))
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
plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=temp_dim, padding=2, normalize=True), (1, 2, 0)))
plt.savefig('./result/%d_c13_%s' % (anomaly_label, filename))
plt.close('all')
