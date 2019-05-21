import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from dataloader import get_data
from utils import *
from config import params

anomaly_label = params['anomaly_label']
trainYn = params['trainYn']
batch_size = params['batch_size']

if (params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif (params['dataset'] == 'CELL'):
    from models.cell_model import Generator, Discriminator, DHead, QHead
elif (params['dataset'] == 'SVHN'):
    from models.svhn_model import Generator, Discriminator, DHead, QHead
elif (params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif (params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

if params['dataset'] == 'MNIST':
    dataloader = get_data(params['dataset'], batch_size, anomaly_label, trainYn)
else:
    dataloader = get_data(params['dataset'], batch_size)

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if (params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    # params['dis_c_dim'] = 10
    # params['num_con_c'] = 2
elif (params['dataset'] == 'CELL'):
    params['num_z'] = 124
    params['num_dis_c'] = 3
    params['dis_c_dim'] = 10
    params['num_con_c'] = 3
    params['datainfo'] = 'normal '
elif (params['dataset'] == 'SVHN'):
    params['num_z'] = 124
    params['num_dis_c'] = 4
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif (params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif (params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

temp_dim = params['dis_c_dim']

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[: temp_dim * temp_dim], nrow=temp_dim, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('./result/%d_epoch%d_Training_Images' % (anomaly_label, params['num_epochs']))
plt.close('all')

# Initialise the network.
num_z_c = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
if (params['dataset'] == 'MNIST' or params['dataset'] == 'CELL'):
    netG = Generator(num_z_c).to(device)
else:
    netG = Generator().to(device)

netG.apply(weights_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

if (params['dataset'] == 'MNIST' or params['dataset'] == 'CELL'):
    netQ = QHead(params['num_con_c']).to(device)
else:
    netQ = QHead().to(device)

netQ.apply(weights_init)
print(netQ)

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

# Adam optimiser is used.
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['learning_rate'],
                    betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'],
                    betas=(params['beta1'], params['beta2']))

# Fixed Noise
temp_100 = temp_dim * params['dis_c_dim']
z = torch.randn(temp_100, params['num_z'], 1, 1, device=device)
fixed_noise = z
if (params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(temp_dim)
    dis_c = torch.zeros(temp_100, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0, temp_100), i, idx] = 1.0

    dis_c = dis_c.view(temp_100, -1, 1, 1)

    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if (params['num_con_c'] != 0):
    con_c = torch.rand(temp_100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label = 1
fake_label = 0

# List variables to store results pf training.
img_list = []
G_losses = []
D_losses = []

print("-" * 25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d\nbeta1: %s\nbeta2: %s'.format(
    params['dataset']) % (
          params['num_epochs'], batch_size, len(dataloader), str(params['beta1']), str(params['beta2'])))
print("-" * 25)

start_time = time.time()
iters = 0

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        # Updating discriminator and DHead
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size,), real_label, device=device)
        output1 = discriminator(real_data)
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label)
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'],
                                  b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label)
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        # Updating Generator and QHead
        optimG.zero_grad()

        # Fake data treated as real.
        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(params['num_dis_c']):
            dis_loss += criterionQ_dis(q_logits[:, j * temp_dim: j * temp_dim + temp_dim], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if (params['num_con_c'] != 0):
            con_loss = criterionQ_con(
                noise[:, params['num_z'] + params['num_dis_c'] * params['dis_c_dim']:].view(-1, params['num_con_c']),
                q_mu, q_var) * 0.1

        # Net loss for generator.
        G_loss = gen_loss + dis_loss + con_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()

        # Check progress of training.
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (epoch + 1, params['num_epochs'], i, len(dataloader),
                 D_loss.item(), G_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=temp_dim, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if ((epoch + 1) == 1 or (epoch + 1) % params['save_epoch'] == 0):
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=temp_dim, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig("./result/%d-Epoch%d" % (anomaly_label, epoch + 1))
        plt.close('all')

    # Save network weights.
    if (params['dataset'] == 'CELL'):
        if (epoch + 1) % params['save_epoch'] == 0:
            torch.save({
                'netG': netG.state_dict(),
                'discriminator': discriminator.state_dict(),
                'netD': netD.state_dict(),
                'netQ': netQ.state_dict(),
                'optimD': optimD.state_dict(),
                'optimG': optimG.state_dict(),
                'params': params
            }, 'checkpoint/model_%d_epoch%d_{}_{}_d{}c{}'.format(params['dataset'], params['datainfo'],
                                                                 params['num_dis_c'],
                                                                 params['num_con_c']) % (anomaly_label, epoch + 1))
    else:
        if (epoch + 1) % params['save_epoch'] == 0:
            torch.save({
                'netG': netG.state_dict(),
                'discriminator': discriminator.state_dict(),
                'netD': netD.state_dict(),
                'netQ': netQ.state_dict(),
                'optimD': optimD.state_dict(),
                'optimG': optimG.state_dict(),
                'params': params
            }, 'checkpoint/model_epoch%d_{}_{}_d{}c{}'.format(params['dataset'], anomaly_label,
                                                                     params['dis_c_dim'],
                                                                     params['num_con_c']) % (epoch + 1))

training_time = time.time() - start_time
print("-" * 50)
print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
print("-" * 50)

# Generate image to check performance of trained generator.
with torch.no_grad():
    gen_data = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=temp_dim, padding=2, normalize=True), (1, 2, 0)))
plt.savefig("./result/Epoch_%d_{}".format(params['dataset']) % (params['num_epochs']))

# Save network weights.
if (params['dataset'] == 'CELL'):
    torch.save({
        'netG': netG.state_dict(),
        'discriminator': discriminator.state_dict(),
        'netD': netD.state_dict(),
        'netQ': netQ.state_dict(),
        'optimD': optimD.state_dict(),
        'optimG': optimG.state_dict(),
        'params': params
    }, 'checkpoint/model_final_{}_{}_d{}c{}_beta{}'.format(params['dataset'], params['datainfo'], params['num_dis_c'],
                                                           params['num_con_c'], params['beta1']))
else:
    torch.save({
        'netG': netG.state_dict(),
        'discriminator': discriminator.state_dict(),
        'netD': netD.state_dict(),
        'netQ': netQ.state_dict(),
        'optimD': optimD.state_dict(),
        'optimG': optimG.state_dict(),
        'params': params
    }, 'checkpoint/model_final{}_{}_{}_d{}c{}'.format(params['num_epochs'], params['dataset'], anomaly_label,
                                                             params['dis_c_dim'],
                                                             params['num_con_c']))

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./result/Loss Curve_%d" % (anomaly_label))

# Animation showing the improvements of the generator.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# anim.save('./result/infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
# plt.show()
