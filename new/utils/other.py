import cv2 as cv
import glob
from PIL import Image
import PIL
from torchvision import transforms
from shutil import copyfile
import random

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


scale_pics("/home/filip/Documents/Univerzita/DP/src/data/pics/*", "/home/filip/Documents/Univerzita/DP/src/data/scaled_pics/",200)
#sample_train_test("/home/filip/Documents/Univerzita/DP/src/data/xmls/*", "/home/filip/Documents/Univerzita/DP/src/data/train_set/", "/home/filip/Documents/Univerzita/DP/src/data/test_set/", 0.8)