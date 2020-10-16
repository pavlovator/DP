import numpy as np
import glob
from utils.xml import XML


def accuracy_per_image(outputs, labels):
    '''
    Metric accuracy per image. Outputs and labels are converted to its proper sizes
    :param outputs: list of numpy probability arrays
    :param labels: list of numpy binary arrays
    :return: percentage 0 - 100
    '''
    good = 0
    num_of_samples = len(outputs)
    for o, l in zip(outputs, labels):
        o_binary = np.round(o)
        good += (o_binary == l).all()
    return (good / num_of_samples)*100


def accuracy_per_pointer(outputs, labels):
    '''
    Metric accuracy per pointer. Outputs and labels are converted to its proper sizes
    :param outputs: list of numpy probability arrays
    :param labels: list of numpy binary arrays
    :return: percentage 0 - 100
    '''
    bad = 0
    num_of_samples = 0
    for o, l in zip(outputs, labels):
        num_of_samples += len(o)
        o_binary = np.round(o)
        bad += np.sum(np.abs(o_binary - l))
    return (1 - bad / num_of_samples)*100


def prevailing_visibility_MAE(struct_lab, struct_out):
    AE = 0
    N = len(struct_lab)
    for date in struct_lab:
        directions_lab = struct_lab[date]
        directions_out = struct_out[date]
        prevailing_lab = sorted(list(directions_lab.values()))[4]
        prevailing_out = sorted(list(directions_out.values()))[4] #prevailing je piata najmensia
        AE += np.abs(prevailing_lab - prevailing_out)
    return AE/ N


def prevailing_visibility_acc(struct_lab, struct_out):
    acc = 0
    N = len(struct_lab)
    for date in struct_lab:
        directions_lab = struct_lab[date]
        directions_out = struct_out[date]
        prevailing_lab = sorted(list(directions_lab.values()))[4]
        prevailing_out = sorted(list(directions_out.values()))[4]  # prevailing je piata najmensia
        #print(prevailing_lab, prevailing_out, sorted(list(directions_lab.values())), sorted(list(directions_out.values())))
        acc += prevailing_lab == prevailing_out
    return acc / N

def test_prevailing_visibility(dates, directions, outputs, labels, config, xml_folder):
    struct_out = get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder)
    struct_lab = get_prevailing_visibility_struct(dates, directions, labels, config, xml_folder)
    MAE = prevailing_visibility_MAE(struct_out, struct_lab)
    acc = prevailing_visibility_acc(struct_out, struct_lab)
    print("Testing Prevailing visibility,  MAE: {:} | acc: {:.2f}".format(MAE, acc*100))

def get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder):
    '''
    :param dates: 
    :param directions: 
    :param outputs: 
    :param labels: 
    :param config: 
    :return: 
    '''
    distances_struct_outputs = {date:{direction:None for direction in range(0, 360, 45)} for date in dates}
    for i in range(len(dates)):
        xml_file = xml_folder + str(dates[i]) + ".xml"
        xml = XML(xml_file)
        distances_struct_outputs[dates[i]][directions[i]] = xml.get_output_prevailing_visibility(config, directions[i], outputs[i])
    return distances_struct_outputs


def unified_to_original(Y, directions, output_sizes):
    '''
    convert np.array of outputs or labels (Y = MxN) to list of np.arrays of size M
    with corresponding lengths, depending on direction
    :param Y: np array of MxN
    :param directions:
    :param output_sizes: dictionary of format (key, value) = ("direction", output_size)
    :return:list of np arrays
    '''
    result = []
    for idx, direction in enumerate(directions):
        single_y = Y[idx]
        result.append(single_y[:output_sizes[str(direction)]])
    return result
