import matplotlib.pyplot as plt
import torchvision.utils as vutils
from config import params
from dataloader import get_data
import argparse
from models.mnist_model import Generator, Discriminator, DHead, QHead
from utils import *
import csv
from evaluations import do_prc

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
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

# restore models: generator, discriminator, netQ
discriminator = Discriminator().to(device)
discriminator.load_state_dict(state_dict['discriminator'])
# print(discriminator)

num_z_c = params['num_z'] + params['num_dis_c'] * params['dis_c_dim'] + params['num_con_c']
netG = Generator(num_z_c).to(device)
netG.load_state_dict(state_dict['netG'])
# print(netG)

netQ = QHead(params['num_con_c']).to(device)
netQ.load_state_dict(state_dict['netQ'])
# print(netQ)

netD = DHead().to(device)
netD.load_state_dict(state_dict['netD'])


# print(netD)


def res_loss(x, Gz):
    abs_sub = np.abs(x - Gz)
    return sum(abs_sub)


def get_rand_z_c():
    idx = np.arange(10).repeat(10)
    zeros = torch.zeros(100, 1, 1, 1, device=device)

    c = np.linspace(-2, 2, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)
    c = torch.from_numpy(c).float().to(device)
    c = c.view(-1, 1, 1, 1)

    dis_c = torch.zeros(100, 10, 1, 1, device=device)
    dis_c[torch.arange(0, 100), idx] = 1.0

    z = torch.randn(100, 60, 1, 1, device=device)

    # Discrete latent code.
    c1 = dis_c.view(100, -1, 1, 1)

    # Continuous latent code.
    c2 = torch.cat((c, zeros), dim=1)
    c3 = torch.cat((zeros, c), dim=1)

    rand_z_c = torch.cat((z, c1, c2, c2), dim=1)

    return rand_z_c


# Get random z_c, and iterate 500 times to pick one which has minimum loss
# among those 500 candidates.
def get_most_similar_zc(x):
    z_c = get_rand_z_c()
    Gz = netG(z_c).detach().cpu()
    min_loss = res_loss(x, Gz)
    result_z_c = z_c

    for _ in range(5):  # 500
        z_c = get_rand_z_c()
        Gz = netG(z_c).detach().cpu()
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
    return result_z_c


def anomaly_score(test_img):
    x = test_img.reshape((1, 1, 28, 28))

    z_c = get_most_similar_zc(x)
    Gz = netG(z_c).detach().cpu()
    a_score = torch.sum(res_loss(x, Gz))
    # print("===")
    # print(label, ': %d' % a_score)
    return a_score


def show_img(test_img, filename):
    ### plt view
    sample_batch = next(iter(test_img))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[: 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('./result/img/raw-test-image-%s' % filename)
    plt.close('all')


# main
anomaly_label = params['anomaly_label']
if (trainYn == True):  # test_base
    f = open('./result/base-%d-%s.csv' % (anomaly_label, filename), 'w', newline='')
    csv_writer = csv.writer(f)
    print('Use training data to calculate base-anomaly score.')
    print('anomaly_label: ', anomaly_label)
    dataloader = get_data(params['dataset'], params['batch_size'], anomaly_label, trainYn)

    rand_idx_list = np.random.choice(len(dataloader), basenum)  # 500

    aScore_list = []
    for idx, item in enumerate(dataloader):
        if (idx in rand_idx_list):
            item = item[0]
            aScore = anomaly_score(item)
            aScore_list.append(aScore)

    csv_writer.writerow(['mean', np.average(aScore_list)])
    csv_writer.writerow(['max', max(aScore_list).item()])
    csv_writer.writerow(['stdvar', np.std(aScore_list)])
    csv_writer.writerow(['mean + 1 sigma ', np.average(aScore_list) + np.std(aScore_list)])
    csv_writer.writerow(['mean + 2 sigma ', np.average(aScore_list) + 2 * np.std(aScore_list)])
    f.close()
    print('mean+1sigma', np.average(aScore_list) + np.std(aScore_list))

if (trainYn == False):
    f = open('./result/test-%d-%d-%s.csv' % (anomaly_label, base_score, filename), 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['label', 'idx', 'score', 'anomalyYN'])

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    scores = []
    testy = np.zeros((0,))
    for label in range(10):
        dataloader = get_data(params['dataset'], params['batch_size'], label, trainYn)

        sample_test = round(1000 * anonum)
        rand_idx_list = np.random.choice(len(dataloader), sample_test)  # 500
        cnt = -1
        scores_temp = []
        testy_temp = np.zeros((sample_test,))
        for idx, item in enumerate(dataloader):
            if (idx >= sample_test):
                break
            cnt += 1
            print(cnt, '/', len(rand_idx_list))

            item = item[0]
            # show_img(item, str(label) + '-' + str(idx))
            aScore = anomaly_score(item)
            scores_temp.append(aScore)
            if (label == anomaly_label):
                print('anomaly_lael data')
                testy_temp[cnt] = 1
            # if (idx in rand_idx_list):
            #     cnt += 1
            #     print(cnt, '/', len(rand_idx_list))
            #
            #     item = item[0]
            #     # show_img(item, str(label) + '-' + str(idx))
            #     aScore = anomaly_score(item)
            #     scores_temp.append(aScore)
            #     if (label == anomaly_label):
            #         print('anomaly_lael data')
            #         testy_temp[cnt] = 1
        scores = np.concatenate((scores, scores_temp))
        testy = np.concatenate((testy, testy_temp))
        print('scores.shape=', scores.shape)
        print('testy.shape=', testy.shape)

        # if (aScore > base_score):
        #     print(str(label) + '-' + str(idx), ' => anomaly // aScore=', aScore.item())
        #     csv_writer.writerow([label, idx, aScore.item(), 'y'])
        #     if (label == anomaly_label):
        #         tp += 1
        #     else:
        #         fp += 1
        # else:
        #     csv_writer.writerow([label, idx, aScore.item(), 'n'])
        #     if (label != anomaly_label):
        #         tn += 1
        #     else:
        #         fn += 1

    prc_auc = do_prc(scores, testy,
                     file_name=r'%d-%.2f_prc_%s' % (anomaly_label, anonum, filename),
                     directory=r'result/')
    print("Testing | PRC AUC = {:.4f}".format(prc_auc))

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
