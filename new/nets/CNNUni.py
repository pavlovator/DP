import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.eval import unified_to_original, accuracy_per_pointer, accuracy_per_image
import numpy as np


class CNNUni(nn.Module):
    def __init__(self, input_shape, latent_dim, output_dim, hidden_dims=None):
        super(CNNUni, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_dim = output_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        conv_modules = []
        in_channels = 3
        for h_dim in hidden_dims:
            conv_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1), #using stride=2 to 1, transform nn.MaxPool2d(2, 2)
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.conv_layers = nn.Sequential(*conv_modules)

        input = torch.zeros(1, 3, *input_shape)
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
        indexes, inputs, labels = out_set[:]
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
        '''#old version
        indexes, inputs, labels = out_set[:]
        with torch.no_grad():
            self.train(False)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
        dates, directions = out_set.get_date_direction(indexes)
        labels = labels.cpu().numpy()
        outputs = outputs.cpu().numpy()
        labels_orig = unified_to_original(labels, directions, output_sizes)
        outputs_orig = unified_to_original(outputs, directions, output_sizes)
        return labels_orig, outputs_orig, dates, directions
        '''


def train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, verbose_interval, history):
    for epoch in range(0, num_epochs):
        for i, data in enumerate(train_loader):
            _, inputs, labels = data
            loss, outputs = net.training_step(inputs, labels, optimizer, criterion, device)
        if epoch % verbose_interval == 0:
            _, val_inputs, val_labels = val_set[:]
            val_loss, val_outputs = net.testing_step(val_inputs, val_labels, criterion, device)
            history.update_loss(loss, val_loss, epoch)
            history.print_last()


def test(labels_orig, outputs_orig):
    app = accuracy_per_pointer(outputs_orig, labels_orig)
    api = accuracy_per_image(outputs_orig, labels_orig)
    print("Testing Accuracy per Image: {:.2f}% | Accuracy per Pointer: {:.2f}%".format(api, app))



