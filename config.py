import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('--anomaly_label', type=int)
parser.add_argument('--filename', type=str)
parser.add_argument('--base_score', type=int)
parser.add_argument('--trainYn', type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--basenum', type=int)
parser.add_argument('--anonum', type=float)
args = parser.parse_args()

# Dictionary storing network parameters.
params = {
    'load_path': args.load_path,  # 128 for train, 1 for anogan
    # 'batch_size': 1,  # 128 for train, 1 for anogan
    'num_epochs': 500,  # 500   # Number of epochs to train for.
    'learning_rate': 2e-4,  # Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,  # 0.999
    'save_epoch': 100,  # After how many epochs to save checkpoints and generate test output.
    'dataset': 'MNIST'}  # Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!

if (params['dataset'] == 'MNIST'):
    params['trainYn'] = args.trainYn  # True , False
    params['anomaly_label'] = args.anomaly_label
    params['filename'] = args.filename
    params['basenum'] = args.basenum

    if (params['trainYn'] == True):
        params['batch_size'] = 128
    elif (params['trainYn'] == False):  # test-anomaly
        params['base_score'] = args.base_score
        params['anonum'] = args.anonum
        params['batch_size'] = 1
# if (params['dataset'] == 'CELL'):
#     params['beta1'] = 0.0  # 0.1 for class 0 and 1
#     params['beta2'] = 0.999
#     params['num_epochs'] = 100
#     params['save_epoch'] = 5
#     params['learning_rate'] = 2e-3
#     params['trainYn'] = False  # True , False
#     params['test_label'] = 1
