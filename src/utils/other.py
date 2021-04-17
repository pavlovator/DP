import numpy as np
from scipy import signal
import glob
from PIL import Image
import PIL
from shutil import copyfile
import random
from utils.plotting import plot_results
from utils.configs import OutputConfig
import utils.dataloaders2
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import signal


def gkern(kernlen=31, std=3): #pouzity
    '''
    :param kernlen: int
    :param std: int
    :return:
    '''
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def map_of_gaussians(relative_positions, width, height, kernel, std):
    '''
    :param relative_positions: (x,y) relative positions of landmarks
    :param width: real width of input
    :param height: real height of image
    :param kernel: size of gauss kernel
    :param std:
    :return: guass map
    '''
    if kernel % 2 != 1:
        raise ValueError('Kernel of even size')
    gmap = np.zeros((height, width))
    for (x, y) in relative_positions:
        real_x = int(width*x)
        real_y = int(height*y)
        kernel_rad = kernel//2
        gmap[real_y-kernel_rad:real_y+kernel_rad+1, real_x-kernel_rad:real_x+kernel_rad+1] += gkern(kernel, std)
    gmap /= gmap.max()
    return gmap

def change_xmls():
    xml_folder = "xmls/202002*.xml"
    files = glob.glob(xml_folder)
    for f in files:
        with open(f) as file:
            ss = file.read()
            if "1452.7192548098114" in ss:
                new = ss.replace("1452.7192548098114", "1513.0485756883343")
                new = new.replace("553.9886948083789", "614.4465859422406")

        with open(f, 'w') as file:
            file.write(new)


def scale_pics(folder_from, folder_to, height):
    pics = glob.glob(folder_from)
    for file in pics:
        x = Image.open(file)
        file_name = file.split('/')[-1]
        x_shape = x.size
        ratio = (x_shape[1]/height)
        x = x.resize((int(x_shape[0]/ratio), height), resample=PIL.Image.BICUBIC)
        x.save(folder_to+file_name)


def sample_train_test(data_folder, train_folder, test_folder, ratio):
    files = glob.glob(data_folder)
    random.shuffle(files)
    treshold = int(len(files)*ratio)
    files_train = files[:treshold]
    files_test = files[treshold:]
    for path in files_train:
        xml_file = path.split('/')[-1]
        copyfile(path, train_folder+xml_file)
    for path in files_test:
        xml_file = path.split('/')[-1]
        copyfile(path, test_folder+xml_file)


def got_bad_images():
    xmls = "/home/filip/Documents/Univerzita/DP/src/data/xmls/*"
    out_conf = OutputConfig("/home/filip/Documents/Univerzita/DP/src/utils/files/output_config.json")
    imgs = "/home/filip/Documents/Univerzita/DP/src/data/scaled_pics/"
    dataset = utils.dataloaders2.VNET1Dataset(xmls_folder=xmls, height=50, img_folder=imgs, config=out_conf)
    directions = dataset.directions.tolist()
    dates = dataset.dates.tolist()
    labels = dataset.raw_Y
    list_of_bad = []
    N = len(dates)
    i = 0
    for direction, date, lab in zip(directions, dates, labels):
        i +=1
        if i > 6235:
            plot_results(date=date, direction=direction, output=lab, label=lab, config=out_conf,
                     mode=5, save="", show=False)
            plt.pause(0.1)
            is_corrupted = input()
            if is_corrupted == 'y':
                list_of_bad.append([date, direction])
                with open('badly_annotated', 'w') as file:
                    file.write('\n'.join([str(dat)+"_"+str(di) for dat, di in list_of_bad]))
            print("prejdenych {:} | {:}/{:}".format(date,i,N))
            plt.close('all')
    return list_of_bad


def high_pass(folder_to = "../../data/a/", height=250):
    folder_from = "../../data/pics/*"
    pics = glob.glob(folder_from)
    mask = None
    for file in pics:
        print(file)
        file_name = file.split('/')[-1]
        img = cv2.imread(file, 0)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        # mask = gen_gaussian_2d_filter(img.shape, 11112111)
        if mask is None:
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[crow - 100:crow + 100, ccol - 100:ccol + 100] = 0
        # mask = np.stack((mask, mask), axis=2)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        norm_image = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        x = Image.fromarray(norm_image)
        x_shape = x.size
        ratio = (x_shape[1]/height)
        x = x.resize((int(x_shape[0]/ratio), height), resample=PIL.Image.BICUBIC)
        x.save(folder_to+file_name)

