import OpenEXR
import Imath
import array
import numpy as np
import cv2


def exr_scaler_grayer(file, x, y):
    img = OpenEXR.InputFile(file)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = img.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (R, G, B) = [array.array('f', img.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    np_img = np.array([(r + g + b) / 3 for r, g, b in zip(R, G, B)])
    np_img.resize(sz[1], sz[0])
    np_img = cv2.resize(np_img, (x, y), interpolation=cv2.INTER_CUBIC)
    return np_img

def png_scaler_grayer(file, x, y):
    img = cv2.imread(file)
    img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def read_camera(folder='folder', filename='Camera.txt', samples_num=250):
    res = np.empty((0, 6), np.float32)
    with open(folder + '/' + filename) as f:
        for i, line in zip(range(samples_num), f.readlines()):
            vals = [float(t) for t in line.split(', ')]
            vals.pop(0)
            res = np.append(res, [vals], axis=0)
    return res


def create_database(folder='folder', input_name='Image', ouput_name='Depth', samples_num=250, size=(160, 120)):
    frmt = "{path}/{file}{val:0>4}.{ext}"
    x = np.empty((samples_num, *reversed(size)), dtype=np.float32)
    y = np.empty((samples_num, *reversed(size)), dtype=np.float32)
    for iter in range(samples_num):
        print(iter)
        x[iter] = png_scaler_grayer(frmt.format(path=folder, file=input_name, val=(iter + 1), ext='png'), *size)
        y[iter] = exr_scaler_grayer(frmt.format(path=folder, file=ouput_name, val=(iter + 1), ext='exr'), *size)
    return x, y
