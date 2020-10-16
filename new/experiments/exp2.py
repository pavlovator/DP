import torch
from utils.dataloaders import UnifiedDataset
from utils.configs import OutputConfig
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.history import History
from nets.CNNUni import CNNUni, train, test
import torch.nn as nn
from utils.eval import test_prevailing_visibility


if torch.cuda.is_available():
    if torch.cuda.device_count() == 2:
        dev = "cuda:1"
    else:
        dev = "cuda:0"
    print('Running on GPU {:}'.format(dev))
else:
    dev = "cpu"
    print('Running on CPU')

device = torch.device(dev)

batch_size = 64
learning_rate = 0.000007
num_epochs = 150
criterion = nn.MSELoss()
img_height = 40

verbose_interval = 1
output_config_file = "../utils/files/output_config.json"
xml_folder = "../data/train_set/*"
img_folder = "../data/scaled_pics/"
out_conf = OutputConfig(output_config_file)
output_size = out_conf.max_output_length()
train_set = UnifiedDataset(xmls_folder=xml_folder, height=img_height, img_folder=img_folder, config=out_conf)
img_shape = tuple(train_set.X.shape[2:])

train_val_splits = [len(train_set) - round(len(train_set) * 0.1), round(len(train_set) * 0.1)]
train_set, val_set = torch.utils.data.random_split(train_set, train_val_splits)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
history = History(num_epochs)
print("Dataset loaded, with image shape of {:}".format(img_shape))
net = CNNUni(input_shape=img_shape, latent_dim=128, output_dim=output_size)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, verbose_interval, history)

xml_folder = "../data/test_set/*"
test_set = UnifiedDataset(xmls_folder=xml_folder, height=img_height, img_folder=img_folder, config=out_conf)
labels_orig, outputs_orig, dates, directions = net.real_output(test_set, device, out_conf.output_lengths())

test(labels_orig, outputs_orig)
test_prevailing_visibility(dates, directions, outputs_orig, labels_orig, out_conf, "/home/filip/Documents/Univerzita/DP/src/data/xmls/")


'''
from utils.plotting import plot_results
for i in range(len(dates)):
    plot_results(date=dates[i], direction=directions[i], output=outputs_orig[i], label=labels_orig[i], config=out_conf,
                 mode=5, save="out/{:}_{:}.png".format(directions[i], dates[i]))
'''