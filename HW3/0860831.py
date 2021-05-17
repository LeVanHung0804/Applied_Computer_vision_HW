from datasets import CIFAR10_truncated
import argparse
from torchvision import transforms
import torch.utils.data as data
from model import *
from utils import *
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="./data/cifar10", help="Data directory")
parser.add_argument('--train_bs', default=128, type=int, help='training batch size')
parser.add_argument('--test_bs', default=100, type=int, help='testing batch size')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--seed', default=25, type=int, help='random seed')
parser.add_argument('--augmentation', default=False, type=int, help='Data Augmentation')
args = parser.parse_args()

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def get_dataloader(datadir, train_bs, test_bs, is_augmentation = False, dataidxs=None):

    if is_augmentation != True:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        print(" Don't use data augmentation!")
        train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = CIFAR10_truncated(datadir, train=False, transform=transform_test, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,pin_memory=True)

    else:
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        print(" Using data augmentation")
        print(" Data augmentation for only training set")

        train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = CIFAR10_truncated(datadir, train=False, transform=transform_test, download=True)
        #train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,pin_memory=True)
        # For unbalanced dataset, create a weightes sampler
        targets = train_ds.target
        class_count = np.unique(targets, return_counts= True)[1]
        #class_count = [1865, 2677, 3602, 916, 4354, 4061, 892, 1718, 3417, 152]
        class_count = np.array(class_count)
        weight = 1. / class_count
        samples_weight = weight[targets]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs,sampler=sampler,pin_memory=True)

    return train_dl, test_dl

if __name__ == '__main__':
    dataidxs = []
    #load the index of imbalanced CIFAR-10 from dataidx.txt
    with open("dataidx.txt", "r") as f:
        for line in f:
            dataidxs.append(int(line.strip()))

    #Set random number seed equal 25
    torch.manual_seed(args.seed)

    #get the training/testing data loader
    train_dl, test_dl = get_dataloader(args.datadir, args.train_bs, args.test_bs, args.augmentation , dataidxs)

    # plot sub image
    #plot_grid_images(data_dl=test_dl)
    # Using GPU

    device = check_cuda()
    train_ld = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    # Load model
    model = to_device(VGG('VGG16'), device)

    # Train model
    # history = [evaluate(model, test_dl)]
    history_training, history_testing = [], []

    history =model_fit(epochs=20, model=model, training_set=train_ld, validation_set=test_dl)

    history_training += history[0]

    history_testing +=history[1]

    result = evaluate(model, test_dl)
    print(result)

    # plot_losses(history)

    plot_accuracies(history_training,history_testing)

    accuracy_in_each_class(model, test_dl)

