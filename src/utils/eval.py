import numpy as np
import glob
from utils.xml import XML
import pandas as pd
from sklearn.metrics import classification_report

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


def class_report(labels, outputs):
    labels_list, outputs_list = [], []
    for o, l in zip(outputs, labels):
        o_binary = np.round(o)
        for i in range(len(o_binary)):
            outputs_list.append(o_binary[i])
            labels_list.append(l[i])
    print(classification_report(labels_list, outputs_list, digits=4))
    perf_measure_matrix(labels_list, outputs_list)


def perf_measure_matrix(labels, outputs):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(outputs)):
        if labels[i]==outputs[i]==1:
           TP += 1
        if outputs[i]==1 and labels[i]!=outputs[i]:
           FP += 1
        if labels[i]==outputs[i]==0:
           TN += 1
        if outputs[i]==0 and labels[i]!=outputs[i]:
           FN += 1
    print("TP = {:}, FP = {:}, TN = {:}, FN = {:}".format(TP, FP, TN, FN))


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
        if prevailing_out > 10000:
            prevailing_out = 10000
        if prevailing_lab > 10000:
            prevailing_lab = 10000
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
    print("Testing Prevailing visibility,  MAE: {:} m. | acc: {:.2f} %".format(MAE, acc*100))


def test_prevailing_visibility_intervals(dates, directions, outputs, labels, config, xml_folder):
    intervals_lab = {0:[], 600:[], 1500:[], 5000:[], 10000:[]}
    intervals_out = {0:[], 600:[], 1500:[], 5000:[], 10000:[]}
    struct_out = get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder)
    struct_lab = get_prevailing_visibility_struct(dates, directions, labels, config, xml_folder)
    for date in struct_lab:
        directions_lab = struct_lab[date]
        directions_out = struct_out[date]
        prevailing_lab = sorted(list(directions_lab.values()))[4]
        prevailing_out = sorted(list(directions_out.values()))[4]  # prevailing je piata najmensia
        if prevailing_out > 10000:
            prevailing_out = 10000
        if prevailing_lab > 10000:
            prevailing_lab = 10000
        if prevailing_lab < 600:
            intervals_lab[0].append(prevailing_lab)
            intervals_out[0].append(prevailing_out)
        elif prevailing_lab >= 600 and prevailing_lab < 1500:
            intervals_lab[600].append(prevailing_lab)
            intervals_out[600].append(prevailing_out)
        elif prevailing_lab >= 1500 and prevailing_lab < 5000:
            intervals_lab[1500].append(prevailing_lab)
            intervals_out[1500].append(prevailing_out)
        elif prevailing_lab >= 5000 and prevailing_lab < 10000:
            intervals_lab[5000].append(prevailing_lab)
            intervals_out[5000].append(prevailing_out)
        elif prevailing_lab >= 10000:
            intervals_lab[10000].append(prevailing_lab)
            intervals_out[10000].append(prevailing_out)
    print("------Testing prevailing visibility of intervals--------")
    compute_visibility_interval(intervals_out, intervals_lab)



def get_real_out_lab(struct_out, struct_lab):
    real_out = []
    real_lab = []
    for date in struct_lab:
        directions_lab = struct_lab[date]
        directions_out = struct_out[date]
        for dir in directions_out:
            d_out, d_lab = directions_out[dir], directions_lab[dir]
            if d_out > 10000:
                d_out = 10000
            if d_lab > 10000:
                d_lab = 10000
            real_out.append(d_out)
            real_lab.append(d_lab)
    return np.array(real_out), np.array(real_lab)


def test_visibility(dates, directions, outputs, labels, config, xml_folder):
    struct_out = get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder)
    struct_lab = get_prevailing_visibility_struct(dates, directions, labels, config, xml_folder)
    real_out, real_lab = get_real_out_lab(struct_out, struct_lab)
    MAE = np.abs(np.sum(real_out - real_lab))/len(real_lab)
    acc = (np.sum(real_out == real_lab)/len(real_lab))
    print("Testing visibility,  MAE: {:} m. | acc: {:.2f} %".format(MAE, acc*100))


def compute_visibility_interval(intervals_out, intervals_lab):
    for interval in intervals_lab:
        labs = np.array(intervals_lab[interval])
        outs = np.array(intervals_out[interval])
        MAE = np.abs(np.sum(outs - labs)) / len(outs)
        acc = (np.sum(outs == labs) / len(outs))
        print("Interval {:}m.  MAE: {:} m. | acc: {:.2f}% | instances: {:}".format(interval, MAE, acc * 100,len(outs)))


def test_visibility_intervals(dates, directions, outputs, labels, config, xml_folder):
    intervals_lab = {0:[], 600:[], 1500:[], 5000:[], 10000:[]}
    intervals_out = {0:[], 600:[], 1500:[], 5000:[], 10000:[]}
    struct_out = get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder)
    struct_lab = get_prevailing_visibility_struct(dates, directions, labels, config, xml_folder)
    real_out, real_lab = get_real_out_lab(struct_out, struct_lab)
    for i, rl in enumerate(real_lab):
        if rl < 600:
            intervals_lab[0].append(real_lab[i])
            intervals_out[0].append(real_out[i])
        elif rl >= 600 and rl < 1500:
            intervals_lab[600].append(real_lab[i])
            intervals_out[600].append(real_out[i])
        elif rl >= 1500 and rl < 5000:
            intervals_lab[1500].append(real_lab[i])
            intervals_out[1500].append(real_out[i])
        elif rl >= 5000 and rl < 10000:
            intervals_lab[5000].append(real_lab[i])
            intervals_out[5000].append(real_out[i])
        elif rl >= 10000:
            intervals_lab[10000].append(real_lab[i])
            intervals_out[10000].append(real_out[i])
    print("------Testing visibility of intervals--------")
    compute_visibility_interval(intervals_out, intervals_lab)


def metar_MAE(struct_out, struct_metar):
    AE = 0
    N = 0
    for date in struct_out:
        if date in struct_metar:
            directions_out = struct_out[date]
            metar_vis = struct_metar[date]
            observer_vis = sorted(list(directions_out.values()))[4]  # prevailing je piata najmensia
            if observer_vis > 10000:
                observer_vis = 10000
            AE += abs(metar_vis - observer_vis)
            N += 1
        else:
            pass
    return AE / N


def test_metar(dates, directions, outputs, config, xml_folder, metar_file):
    struct_out = get_prevailing_visibility_struct(dates, directions, outputs, config, xml_folder)
    struct_metar = get_prevailing_visibility_metar_struct(metar_file)
    MAE = metar_MAE(struct_out, struct_metar)
    print("Testing Observer to Metar visibility,  MAE: {:.2f} m.".format(MAE))



def get_prevailing_visibility_metar_struct(metar_file):
    df = pd.read_csv(metar_file)
    struct = dict()
    for date, pv in zip(df['date'], df['p_vis']):
        date = pd.to_datetime(date) - pd.offsets.Minute(10)
        key_date = date.strftime("%Y%m%d%H%M")
        struct[int(key_date)] = pv
    return struct

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