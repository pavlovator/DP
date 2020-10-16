import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.xml import XML
import numpy as np


def plot_raw_photo(date, direction, only_marked=True):
    '''
    Plot xml file as it is. With NO constraints on desired outputs or configuration file.
    :param date:
    :param direction:
    :param only_marked:
    :return:
    '''
    xml_file_name = "../data/xmls/{:}.xml".format(date)
    img_file_name = "../data/pics/dir_{:}_date_{:}.jpg".format(direction, date)
    xml = XML(xml_file_name)
    img = plt.imread(img_file_name)
    attributes = xml.get_pointer_attributes(direction)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img)
    for idx, viz in enumerate(attributes['visibility']):
        if viz == 'VISIBILITY_NOT_SET' and not only_marked:
            ax.scatter(float(attributes['x'][idx]), float(attributes['y'][idx]), s=200, marker='o', facecolors='none', edgecolors='black')
        if viz != 'VISIBILITY_NOT_SET':
            if viz == 'VISIBILITY_VISIBLE':
                color = 'green'
            else:
                color = 'red'
            ax.scatter(float(attributes['x'][idx]), float(attributes['y'][idx]), s=200, marker='o', facecolors='none', edgecolors=color)
    ax.set(title='Markers')
    plt.show()

def plot_results(date, direction, output, label, config, mode, save = "", show = False):
    '''
    Mode 1,2,3,4 and 5 plots "label", "output", "output with probability", "output and label",
    "output and label with probability" respectively.
    :param date:
    :param direction:
    :param output: numpy array
    :param label: numpy array
    :param config: OutputConfig
    :param mode: int
    :param save: if not empty string save picture as
    :return:
    '''
    def _plot_mode():
        output_binary = np.round(output)
        for idx, (x, y) in enumerate(xy_arr):
            x, y = float(x), float(y)
            output_color = color_fx(output_binary[idx])
            label_color = color_fx(label[idx])
            box_text = ""
            if mode == 1:
                box_color = label_color
            elif mode == 2:
                box_color = output_color
            elif mode == 3:
                box_color = output_color
                label_color = output_color
                box_text = round(output[idx], 2)
            elif mode == 4:
                box_color = label_color
                label_color = output_color
                box_text = " "*3
            elif mode == 5:
                box_color = label_color
                label_color = output_color
                box_text = round(output[idx], 2)
            rect_pos = (x - w // 2, y - h // 2 + 9)
            rect = patches.Rectangle(rect_pos, w, h, linewidth=1, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
            if mode != 1 and mode != 2:
                ax.text(x - w // 2, y - h // 2, box_text, fontsize=6,
                        bbox={'facecolor': label_color, 'alpha': 1, 'edgecolor': label_color, 'pad': 0})

    color_fx = lambda x: 'r' if (x == 0) else 'g'
    img_file_name = "../data/pics/dir_{:}_date_{:}.jpg".format(direction, date)
    w, h = 60, 60
    img = plt.imread(img_file_name)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    xy_arr = config.get_output_positions(direction)
    _plot_mode()
    if save != "":
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close('all')

#from utils.configs import OutputConfig
#out = np.random.random(17)
#lab = np.round(np.random.random(17))
#plot_results(date=201911010620, direction= 0, output=out, label=lab, config = OutputConfig("files/output_config.json"), mode=5, save = "moje.png", show=False)