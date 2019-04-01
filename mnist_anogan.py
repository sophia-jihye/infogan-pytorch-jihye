import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import params
from dataloader import get_data
import argparse
from models.mnist_model import Generator, Discriminator, DHead, QHead
from utils import *
import csv
from evaluations import do_prc
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


print('load_path:', params['load_path'])
state_dict = torch.load(params['load_path'])
filename = params['filename']
basenum = params['basenum']
trainYn = params['trainYn']

if (params['trainYn'] == False):
    anonum = params['anonum']
    base_score = params['base_score']

print('trainYn: ', trainYn)

if (params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 9
    params['num_con_c'] = 2

lambda_res = params['lambda_res']
lambda_disc = params['lambda_disc']
lambda_cdis = params['lambda_cdis']
lambda_ccon = params['lambda_ccon']
sim_num = params['sim_num']

temp_dim = params['dis_c_dim']

# restore models: generator, discriminator, netQ
discriminator = Discriminator().to(device)
discriminator.load_state_dict(state_dict['discriminator'])

num_z_c = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
netG = Generator(num_z_c).to(device)
netG.load_state_dict(state_dict['netG'])

netQ = QHead(params['num_con_c']).to(device)
netQ.load_state_dict(state_dict['netQ'])

netD = DHead().to(device)
netD.load_state_dict(state_dict['netD'])

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()


def res_loss(x, Gz):
    abs_sub = abs(x - Gz)
    return sum(abs_sub)


def get_rand_z_c():
    temp_100 = temp_dim * params['dis_c_dim']
    idx = np.arange(params['dis_c_dim']).repeat(temp_dim)
    zeros = torch.zeros(temp_100, 1, 1, 1, device=device)

    c = np.linspace(-2, 2, params['dis_c_dim']).reshape(1, -1)
    c = np.repeat(c, temp_dim, 0).reshape(-1, 1)
    c = torch.from_numpy(c).float().to(device)
    c = c.view(-1, 1, 1, 1)

    dis_c = torch.zeros(temp_100, params['dis_c_dim'], 1, 1, device=device)
    dis_c[torch.arange(0, temp_100), idx] = 1.0

    z = torch.randn(temp_100, 60, 1, 1, device=device)

    # Discrete latent code.
    c1 = dis_c.view(temp_100, -1, 1, 1)

    # Continuous latent code.
    c2 = torch.cat((c, zeros), dim=1)
    c3 = torch.cat((zeros, c), dim=1)

    rand_z_c = torch.cat((z, c1, c2, c3), dim=1)

    return rand_z_c


# Get random z_c, and iterate 500 times to pick one which has minimum loss
# among those 500 candidates.
def get_most_similar_zc(x):
    z_c = get_rand_z_c()
    Gz = netG(z_c)
    min_loss = res_loss(x, Gz)
    result_z_c = z_c

    for _ in range(sim_num):  # 500
        z_c = get_rand_z_c()
        Gz = netG(z_c)
        loss = res_loss(x, Gz)
        if (torch.sum(loss) < torch.sum(min_loss)):
            min_loss = loss
            result_z_c = z_c

    # TEMP TEMP FOR VISUALIZATION
    # with torch.no_grad():
    #     generated_img1 = netG(result_z_c).detach().cpu()
    # # Display the generated image.
    # fig = plt.figure(figsize=(10, 10))
    # plt.axis("off")
    # plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1, 2, 0)))
    # plt.savefig('./result/result_z_c {}'.format(label))
    # plt.close('all')
    return result_z_c, Gz.to(device)


def anomaly_score(test_img):
    x = test_img.reshape((1, 1, 28, 28))

    z_c, Gz = get_most_similar_zc(x)
    # Gz = netG(z_c)

    # res_loss
    sub_res_loss = torch.sum(res_loss(x, Gz))
    print('sub_res_loss', sub_res_loss.item())

    # discriminator_loss
    output = discriminator(Gz)
    label = torch.full((81,), 1, device=device)
    probs_fake = netD(output).view(-1)
    sub_discriminator_loss = criterionD(probs_fake, label)
    print('sub_discriminator_loss', sub_discriminator_loss.item())

    # c_dis_loss
    q_logits, q_mu, q_var = netQ(output)
    idx = np.zeros((81, 81))
    target = torch.LongTensor(idx).to(device)
    temp_dim = 9
    c_dis_loss = 0
    for j in range(params['num_dis_c']):
        c_dis_loss += criterionQ_dis(q_logits[:, j * temp_dim: j * temp_dim + temp_dim], target[j])
    print('c_dis_loss=', c_dis_loss.item())

    # c_con_loss
    c_con_loss = 0
    if (params['num_con_c'] != 0):
        c_con_loss = criterionQ_con(
            z_c[:, params['num_z'] + params['num_dis_c'] * params['dis_c_dim']:].view(-1, params['num_con_c']),
            q_mu, q_var) * 0.1
    print('c_con_loss=', c_con_loss.item())

    a_score = lambda_res * sub_res_loss.detach().cpu() + lambda_disc * sub_discriminator_loss.detach().cpu() + lambda_cdis * c_dis_loss.detach().cpu() + lambda_ccon * c_con_loss.detach().cpu()
    print('a_score=', a_score)
    return a_score


def show_img(test_img, filename):
    ### plt view
    sample_batch = next(iter(test_img))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[: temp_dim * temp_dim], nrow=temp_dim, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('./result/img/raw-test-image-%s' % filename)
    plt.close('all')


# main
anomaly_label = params['anomaly_label']

if (trainYn == False):
    f = open('./result/test-%d-%.2f-%s.csv' % (anomaly_label, anonum, filename), 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['label', 'idx', 'score', 'anomalyYN'])

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    import time

    begin = time.time()
    inference_time = []

    scores = []
    testy = np.zeros((0,))
    for label in range(10):
        dataloader = get_data(params['dataset'], params['batch_size'], label, trainYn)

        if (anonum != 1.0):
            sample_test = round(1000 * anonum)
            rand_idx_list = np.random.choice(len(dataloader), sample_test)  # 500

        cnt = -1
        scores_temp = []

        if (anonum != 1.0):
            testy_temp = np.zeros((sample_test,))
        else:
            testy_temp = np.zeros((len(dataloader),))

        for idx, item in enumerate(dataloader):
            if (anonum != 1.0):
                if (idx >= sample_test):
                    break
            cnt += 1

            if (anonum != 1.0):
                print(cnt, '/', len(rand_idx_list))
            else:
                print(cnt, '/', len(dataloader))

            item = item[0].to(device)
            aScore = anomaly_score(item)

            scores_temp.append(aScore)
            if (label == anomaly_label):
                print('anomaly_lael data')
                testy_temp[cnt] = 1

        consumed_time = time.time() - begin

        scores = np.concatenate((scores, scores_temp))
        testy = np.concatenate((testy, testy_temp))
        print('scores.shape=', scores.shape)
        print('testy.shape=', testy.shape)

    prc_auc = do_prc(scores, testy,
                     file_name=r'%d-%.2f_prc_%s' % (anomaly_label, anonum, filename),
                     directory=r'result/')
    print("Testing | PRC AUC = {:.4f}".format(prc_auc))
    print('consumed_time=', consumed_time)

    # f1_score_val = f1_score(testy, scores, average='weighted')
    # print("f1_score=", f1_score_val)
    # csv_writer.writerow(['f1-score', f1_score_val])

    # print('fp=', fp)
    # print('tp=', tp)
    # print('tn=', tn)
    # print('fn=', fn)
    # if fp == 0:
    #     fp = 0.01
    # if tp == 0:
    #     tp = 0.01
    # if tn == 0:
    #     tn = 0.01
    # if fn == 0:
    #     fn = 0.01
    # precision = tp / (tp + fp)
    # sensitivity = tp / (tp + fn)
    # specificity = tn / (tn + fp)
    # f1score = 2 * tp / (2 * tp + fp + fn)
    # csv_writer.writerow([])
    # csv_writer.writerow(['false-positive(label7, anomaly)', fp])
    # csv_writer.writerow(['true-positive(label8-6, anomaly)', tp])
    # csv_writer.writerow(['false-negative(label7, normal)', fn])
    # csv_writer.writerow(['true-negative(label8-6, normal)', tn])
    # csv_writer.writerow(['precision', precision])
    # csv_writer.writerow(['sensitivity', sensitivity])
    # csv_writer.writerow(['specificity', specificity])
    # csv_writer.writerow(['f1-score', f1score])
    f.close()
    # print('precision=', precision)
    # print('sensitivity=', sensitivity)
    # print('specificity=', specificity)
    # print('f1score=', f1score)

# if (trainYn == True):  # test_base
#     f = open('./result/base-%d-%s.csv' % (anomaly_label, filename), 'w', newline='')
#     csv_writer = csv.writer(f)
#     print('Use training data to calculate base-anomaly score.')
#     print('anomaly_label: ', anomaly_label)
#     dataloader = get_data(params['dataset'], params['batch_size'], anomaly_label, trainYn)
#
#     rand_idx_list = np.random.choice(len(dataloader), basenum)  # 500
#
#     aScore_list = []
#     for idx, item in enumerate(dataloader):
#         if (idx in rand_idx_list):
#             item = item[0]
#             aScore = anomaly_score(item)
#             aScore_list.append(aScore)
#
#     csv_writer.writerow(['mean', np.average(aScore_list)])
#     csv_writer.writerow(['max', max(aScore_list).item()])
#     csv_writer.writerow(['stdvar', np.std(aScore_list)])
#     csv_writer.writerow(['mean + 1 sigma ', np.average(aScore_list) + np.std(aScore_list)])
#     csv_writer.writerow(['mean + 2 sigma ', np.average(aScore_list) + 2 * np.std(aScore_list)])
#     f.close()
#     print('mean+1sigma', np.average(aScore_list) + np.std(aScore_list))
