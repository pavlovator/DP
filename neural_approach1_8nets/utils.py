import torch
import json
from matplotlib import pyplot as plt

def update_history_RV1(history, ep, t_l, v_l, t_mae, v_mae):
    history["epoch"].append(ep)
    history["train_loss"].append(t_l)
    history["validation_loss"].append(v_l)
    history["train_MAE"].append(t_mae)
    history["validation_MAE"].append(v_mae)


def update_history_CV1(history, ep, t_l, v_l, t_a_a, v_a_a, t_a_i, v_a_i):
    history["epoch"].append(ep)
    history["train_loss"].append(t_l)
    history["validation_loss"].append(v_l)
    history["train_acc_arrows"].append(t_a_a)
    history["validation_acc_arrows"].append(v_a_a)
    history["train_acc_image"].append(t_a_i)
    history["validation_acc_image"].append(v_a_i)


def compute_arrows_image_accuracy(outputs, labels):
    outputs = torch.round(outputs)
    false_arrows_count = torch.abs(labels - outputs).sum().item()
    all_arrows_count = outputs.shape[0] * outputs.shape[1]
    bad_counts_per_image = torch.sum((outputs - labels), axis=1)
    false_image_count = len(torch.nonzero(bad_counts_per_image))
    all_image_counts = len(bad_counts_per_image)
    return (1 - false_arrows_count / all_arrows_count)*100, (1 - false_image_count / all_image_counts)*100


def save_history(history, path):
    with open(path, 'w') as fp:
        json.dump(history, fp)

def load_history(path):
   with open(path) as f_in:
       return(json.load(f_in))

def save_model(epochs, batch_size, model, optimizer, loss, path):
    torch.save({
        'num_epochs': epochs,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def save_test_C(outputs, labels, path):
    outputs = torch.round(outputs).tolist()
    labels = labels.tolist()
    with open(path, 'w') as file:
        print("outputs labels", file=file)
        for y_hat, y in zip(outputs, labels):
            y_hat = "".join(list(map(lambda x: str(int(x)), y_hat)))
            y = "".join(list(map(lambda x: str(int(x)), y)))
            line = "{:} {:}".format(y_hat, y)
            print(line, file=file)


def save_test_R(outputs, labels, path):
    outputs = outputs.squeeze()
    labels = labels.squeeze()
    with open(path, 'w') as file:
        print("outputs labels", file=file)
        for y_hat, y in zip(outputs, labels):
            print("{:} {:}".format(y_hat, y), file=file)


def plot_classifier_history(path_history, path_plot, direction):
    history = load_history(path_history)
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(10, 5))
    epochs = history["epoch"]
    fig.suptitle('Direction: {:}'.format(direction), size=9)
    axs[0].plot(epochs, history["train_loss"])
    axs[0].plot(epochs, history["validation_loss"], '--')
    axs[0].set_xlabel("epochs", fontsize=8)
    axs[0].set_ylabel("MSE loss", fontsize=8)
    axs[0].legend(("train loss", "validation loss"), prop={'size': 7})
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[1].plot(epochs, history["train_acc_arrows"])
    axs[1].plot(epochs, history["validation_acc_arrows"], '--')
    axs[1].plot(epochs, history["train_acc_image"])
    axs[1].plot(epochs, history["validation_acc_image"], '--')
    axs[1].set_xlabel("epochs", fontsize=8)
    axs[1].set_ylabel("% accuracy", fontsize=8)
    axs[1].legend(("train acc arrows", "val acc arrows", "train acc img", "val acc img"), loc=4, prop={'size': 7})
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout(pad=1.5)
    plt.savefig(path_plot, dpi=300)
    plt.show()


def get_distances_to_arrows(direction):
    switch = False
    distances = []
    with open("arrows.txt") as file:
        for row in file:
            indicator, value = row.strip().split(' ')
            value = int(value)
            if switch == True and indicator == 'a':
                break
            elif switch == True:
                distances.append(value)
            if indicator == 'a' and value == direction:
                switch = True
    distances.sort(reverse=True)
    return distances


def arrows_code_to_distance(arrows, distances):
    for ar, dist in zip(arrows, distances):
        if ar.item() == 1:
            return dist
    return 0


def compute_distance_AE(outputs, labels, distances):
    AE = 0
    for out_arrows, lab_arrows in zip(outputs, labels):
        out_dis = arrows_code_to_distance(out_arrows, distances)
        lab_dis = arrows_code_to_distance(lab_arrows, distances)
        AE += abs(lab_dis - out_dis)
    return AE

def print_test(test_file, direction):
    with open(test_file) as file:
        file.readline()
        outputs = []
        labels = []
        for row in file:
            output, label = row.strip().split(' ')
            output = list(map(float, list(map(int, list(output)))))
            label = list(map(float, list(map(int, list(label)))))
            outputs.append(torch.tensor(output))
            labels.append(torch.tensor(label))
    labels = torch.stack(labels)
    outputs = torch.stack(outputs)
    distances = get_distances_to_arrows(direction)
    AE = compute_distance_AE(outputs, labels, distances)
    MAE = AE / len(labels)
    acc_arrows, acc_img = compute_arrows_image_accuracy(outputs, labels)
    print("Testing direction: {:}: acc_arrows: {:.2f} | acc_img: {:.2f} | MAE: {:}".format(direction, acc_arrows, acc_img, MAE))

def update_tb_writer(tb_writer, net_name, step, train_loss, val_loss, train_acc_arrows, val_acc_arrows, train_acc_img, val_acc_img):
    tb_writer.add_scalars('Loss {:}'.format(net_name), {'Train loss':train_loss, 'Validation loss':val_loss}, step)
    tb_writer.add_scalars('Accuracy arrows {:}'.format(net_name), {'Train accuracy arrows':train_acc_arrows, 'Validation accuracy arrows':val_acc_arrows}, step)
    tb_writer.add_scalars('Accuracy image {:}'.format(net_name), {'Train accuracy image':train_acc_img, 'Validation accuracy image':val_acc_img}, step)