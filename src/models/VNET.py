import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.eval import unified_to_original, accuracy_per_pointer, accuracy_per_image
import numpy as np


class VNET(nn.Module):
    def __init__(self, input_shape, latent_dim, output_dim, hidden_dims=None, in_channels=3):
        '''
        :param input_shape: input shape of image
        :param latent_dim:
        :param output_dim: number of landmarks
        :param hidden_dims: convolutional layers
        :param in_channels: number of channels e.g. 3 for rgb image
        '''
        super(VNET, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_dim = output_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        conv_modules = []
        in_channels_tmp = in_channels
        for h_dim in hidden_dims:
            conv_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.conv_layers = nn.Sequential(*conv_modules)

        input = torch.zeros(1, in_channels_tmp, *input_shape)
        output = self.conv_layers(input)
        self.conv_layers.zero_grad()
        _, dims, height, width = output.shape
        self.output_conv_dim = dims * height * width
        self.fc1 = nn.Linear(self.output_conv_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)

    def forward(self, input):
        '''
        foward pass of network
        :param input:
        :return: output
        '''
        z = self.conv_layers(input)
        z = z.view(-1, self.output_conv_dim)
        z = F.relu(self.fc1(z))
        y = torch.sigmoid(self.fc2(z))
        return y

    def training_step(self, inputs, labels, optimizer, criterion, device):
        '''
        Forward and backward pass of network
        :param inputs:
        :param labels:
        :param optimizer:
        :param criterion:
        :param device:
        :return: loss of batch and outputs
        '''
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss, outputs

    def testing_step(self, inputs, labels, criterion, device):
        '''
        only forward pass without calculating of gradients
        :param inputs:
        :param labels:
        :param criterion:
        :param device:
        :return:
        '''
        with torch.no_grad():
            self.train(False)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            self.train(True)
        return loss, outputs

    def real_output(self, out_set, device, output_sizes):
        '''
        Output of network compatible for further analysis.
        Every element is computed separately due to memory reasons.
        :param out_set:
        :param device:
        :param output_sizes:
        :return: transformed labels and outputs to proper output sizes with dates and directions
        '''
        indexes, inputs, labels = [], [] ,[]
        for j in range(len(out_set)):
            idx, inp, lab = out_set[j]
            indexes.append(idx)
            inputs.append(inp)
            labels.append(lab)
        indexes = torch.stack(indexes)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        new_labels, new_outputs = [], []
        with torch.no_grad():
            self.train(False)
            for i in range(len(indexes)):
                single_input = inputs[i:i+1]
                single_label = labels[i:i+1]
                single_input, single_label = single_input.to(device), single_label.to(device)
                single_output = self(single_input)
                new_labels.append(single_label.cpu().numpy())
                new_outputs.append(single_output.cpu().numpy())
        labels = np.concatenate(new_labels)
        outputs = np.concatenate(new_outputs)
        dates, directions = out_set.get_date_direction(indexes)
        labels_orig = unified_to_original(labels, directions, output_sizes)
        outputs_orig = unified_to_original(outputs, directions, output_sizes)
        return labels_orig, outputs_orig, dates, directions


def train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, verbose_interval, history):
    for epoch in range(0, num_epochs):
        for i, data in enumerate(train_loader):
            _, inputs, labels = data
            loss, outputs = net.training_step(inputs, labels, optimizer, criterion, device)
            del inputs
        if epoch % verbose_interval == 0:
            val_loss, val_inputs, val_labels = 0, [], []
            for j in range(len(val_set)):
                _, vi, vl = val_set[j]
                val_inputs.append(vi)
                val_labels.append(vl)
                if j %32 == 0:
                    val_inputs = torch.stack(val_inputs)
                    val_labels = torch.stack(val_labels)
                    val_loss_part, val_outputs = net.testing_step(val_inputs, val_labels, criterion, device)
                    if val_loss == 0:
                        val_loss = val_loss_part
                    else:
                        val_loss += val_loss_part
                    val_inputs, val_labels = [], []
            history.update_loss(loss, val_loss/(len(val_set)//32), epoch)
            history.print_last()


def test(labels_orig, outputs_orig):
    app = accuracy_per_pointer(outputs_orig, labels_orig)
    api = accuracy_per_image(outputs_orig, labels_orig)
    print("Testing Accuracy per Image: {:.2f}% | Accuracy per Pointer: {:.2f}%".format(api, app))



