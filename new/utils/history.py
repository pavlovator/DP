class History:
    def __init__(self, n_epochs):
        self.train_losses = []
        self.validation_losses = []
        self.epochs = []
        self.n_epochs = n_epochs

    def update_loss(self, train_loss, val_loss, epoch):
        '''
        update losses corresponding to given epoch
        :param train_loss:
        :param val_loss:
        :param epoch:
        :return:
        '''
        self.train_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        self.epochs.append(epoch)

    def print_last(self):
        print("Epoch: {:} | train loss: {:} | validation loss: {:}"
              .format(self.epochs[-1], self.train_losses[-1], self.validation_losses[-1]))
