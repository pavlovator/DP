import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from torch.utils.data import DataLoader
from utils_CV1 import *


class CNN_CV1_100(nn.Module):
    def __init__(self, direction):
        super(CNN_CV1_100, self).__init__()
        self.direction = direction
        self.num_outputs = {0: 17, 45: 20, 90: 19, 135: 8, 180: 15, 225: 9, 270: 23, 315: 20}
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels, out_channels, kernel_size
        self.batch_norm1 = torch.nn.BatchNorm2d(6) # num of input_channels
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm2 = torch.nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.batch_norm3 = torch.nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 7, 120)
        self.batch_norm4 = torch.nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_outputs[direction])

    def forward(self, x):
        # input shape: (batch_size, channels, height, width) : (b, 3, 100, 177)
        x = self.pool1(self.batch_norm1(F.relu(self.conv1(x)))) # out: (b, 6, 24, 43)
        x = self.pool2(self.batch_norm2(F.relu(self.conv2(x)))) # out: (b, 16, 10, 19)
        x = self.pool3(self.batch_norm3(F.relu(self.conv3(x)))) # out: (b, 32, 3, 7)
        x = x.view(-1, 32 * 3 * 7)
        x = self.batch_norm4(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


if torch.cuda.is_available():
    dev = "cuda:0"
    print('Running on GPU')
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)
for direction in range(0, 360, 45):
    batch_size = 32
    num_epochs = 150
    learning_rate = 0.0001
    train_set = ScaledDatasetC("uniform_train/*", height=100, direction=direction)

    train_val_splits = [len(train_set) - round(len(train_set) * 0.2),
                        round(len(train_set) * 0.2)]  # lengths of splits to be produced
    history = {"epoch": [], "train_loss": [], "validation_loss": [], "train_acc_arrows": [],
               "validation_acc_arrows": [], "train_acc_image": [], "validation_acc_image": []}

    print('Loading dataset for direction {:}'.format(direction))
    train_set, val_set = torch.utils.data.random_split(train_set, train_val_splits)
    print("Length of: Train_set: {:} & Validation_set {:}".format(len(train_set), len(val_set)))
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    net = CNN_CV1_100(direction)
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, history, verbose_interval=2)

    save_history(history, "history_models/100px/CNN_CV1_dir_{:}_100px.json".format(direction))
    save_model(num_epochs, batch_size, net, optimizer, criterion, "trained_models/100px/CNN_CV1_dir_{:}_100px.pt".format(direction))

    test_set = ScaledDatasetC("uniform_test/*", height=100, direction=direction)
    test(net, test_set, device, "test_models/100px/CNN_CV1_dir_{:}_100px.txt".format(direction))