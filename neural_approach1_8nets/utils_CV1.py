from utils import *
from torch.utils.tensorboard import SummaryWriter

def train(net, optimizer, criterion, val_set, device, num_epochs, train_loader, history, verbose_interval):
    net_name = net._get_name() + "_direction:_{:}".format(net.direction)
    tb_writer = SummaryWriter()
    for epoch in range(1, num_epochs+1):
        train_loss, train_outputs, all_labels, all_outputs = 0.0, 0.0, [], []
        for i, data in enumerate(train_loader):
            #get data to GPU if possible
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            #backward pass
            optimizer.step()
            train_loss += loss.item()
            all_labels.append(labels)
            all_outputs.append(outputs)
        if epoch % verbose_interval == 0:
            val_loss, val_acc_arrows, val_acc_img = validation(net, val_set, criterion, device)
            train_loss /= len(train_loader)
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            train_acc_arrows, train_acc_img = compute_arrows_image_accuracy(all_outputs, all_labels)
            print('epoch: {:} | train_loss: {:.4f} | val_loss: {:.4f} | train_acc_arrows: {:.2f} | '
                  'val_acc_arrows: {:.2f} | train_acc_img: {:.2f} | val_acc_img: {:.2f}'
                  .format(epoch, train_loss, val_loss, train_acc_arrows, val_acc_arrows, train_acc_img, val_acc_img))
            update_history_CV1(history, epoch, train_loss, val_loss, train_acc_arrows, val_acc_arrows, train_acc_img, val_acc_img)
            update_tb_writer(tb_writer, net_name, epoch, train_loss, val_loss, train_acc_arrows, val_acc_arrows, train_acc_img, val_acc_img)

def validation(net, val_set, criterion, device):
    with torch.no_grad():
        net.train(False)
        inputs, labels = val_set[:]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_acc_arrows, val_acc_img = compute_arrows_image_accuracy(outputs, labels)
        net.train(True)
        return loss.item() / len(val_set), val_acc_arrows, val_acc_img


def test(net, test_set, device, path):
    with torch.no_grad():
        net.train(False)
        inputs, labels = test_set[:]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        acc_arrows, acc_img = compute_arrows_image_accuracy(outputs, labels)
        print("Testing: acc_img: {:.2f} | acc_arrows: {:.2f}".format(acc_img, acc_arrows))
        save_test_C(outputs, labels, path)
