import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Directory containing the data.
root = 'data/'


def get_data(dataset, batch_size, label, trainYn):
    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root + 'mnist/', train=trainYn,
                              download=True, transform=transform)

        if trainYn == True:
            # contain labels except for anomaly_label
            dataset_labels = dataset.train_labels.tolist()
            dataset_temp = []
            for i in range(len(dataset)):
                if dataset_labels[i] != label:
                    dataset_temp.append(dataset.__getitem__(i))
            dataset = dataset_temp
            print('[anomaly_label ', label, '] # of data: ', len(dataset))

        elif trainYn == False:
            dataset_labels = dataset.test_labels.tolist()
            dataset_temp = []
            for i in range(len(dataset)):
                if dataset_labels[i] == label:
                    dataset_temp.append(dataset.__getitem__(i))
            dataset = dataset_temp
            print('[label ', label, '] # of data: ', len(dataset))

        dataloader = dataset

    # Get cell dataset.
    elif dataset == 'CELL':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])
        if trainYn == True:
            dataset = dsets.ImageFolder(root=root + 'cell/',
                                        transform=transform)
        elif trainYn == False:
            dataset = dsets.ImageFolder(root=root + 'cell-test/',
                                        transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root + 'svhn/', split='train',
                             download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root + 'fashionmnist/', train='train',
                                     download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root + 'celeba/', transform=transform)

    # Create dataloader.
    if dataset != 'MNIST':
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

    return dataloader
