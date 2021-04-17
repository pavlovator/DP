import torch
from utils.dataloaders2 import VNET1Dataset
from utils.configs import OutputConfig
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.history import History
from utils.plotting import plot_history
from models.VNET import VNET, train, test
import torch.nn as nn
from utils.eval import class_report, test_prevailing_visibility, test_visibility, test_visibility_intervals, test_prevailing_visibility_intervals


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

#set hyper-parameters
batch_size = 32
learning_rate = 0.00001
num_epochs = 50
criterion = nn.MSELoss()
img_height = 250

#training and validation set
verbose_interval = 1
output_config_file = "../utils/files/output_config.json"
xml_folder_train = "../../data/train_set/*"
xml_folder_val = "../../data/validation_set/*"
img_folder = "../../data/scaled_pics250/"
out_conf = OutputConfig(output_config_file)
output_size = out_conf.max_output_length()
train_set = VNET1Dataset(xmls_folder=xml_folder_train, height=img_height, img_folder=img_folder, config=out_conf)
val_set = VNET1Dataset(xmls_folder=xml_folder_val, height=img_height, img_folder=img_folder, config=out_conf)
img_shape = tuple(train_set[0][1].shape[1:3])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
history = History(num_epochs)
print("Dataset loaded, with image shape of {:}".format(img_shape))
net = VNET(input_shape=img_shape, latent_dim=128, output_dim=output_size, in_channels=3)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, verbose_interval, history)
plot_history(history, "VNET1")


xml_folder = "../../data/test_set/*"
test_set = VNET1Dataset(xmls_folder=xml_folder, height=img_height, img_folder=img_folder, config=out_conf)
labels_orig, outputs_orig, dates, directions = net.real_output(test_set, device, out_conf.output_lengths())



test(labels_orig, outputs_orig)
class_report(labels_orig, outputs_orig)
print('Testing (Prevailing)Visibility')
test_visibility(dates, directions, outputs_orig, labels_orig, out_conf, "../../data/xmls/")
test_visibility_intervals(dates, directions, outputs_orig, labels_orig, out_conf, "../../data/xmls/")
test_prevailing_visibility(dates, directions, outputs_orig, labels_orig, out_conf, "../../data/xmls/")
test_prevailing_visibility_intervals(dates, directions, outputs_orig, labels_orig, out_conf, "../../data/xmls/")