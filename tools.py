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


def create_database(folder='folder', input_name='Image', ouput_name='Depth', samples_num=3, size=(160, 120)):
    arrays_num = 3
    frmt = "{path}/{file}{val:0>4}.{ext}"
    x = [np.empty((samples_num - arrays_num + 1, *reversed(size)), dtype=np.float32) for arr in range(arrays_num)]
    y = np.empty((samples_num - arrays_num + 1, *reversed(size)), dtype=np.float32)
    for iter in range(samples_num - arrays_num + 1):
        for arr in range(arrays_num):
            if iter > 0 and arr + 1 < arrays_num:
                x[arr][iter] = x[arr + 1][iter - 1]
            else:
                x[arr][iter] = png_scaler_grayer(frmt.format(path=folder, file=input_name, val=(iter + arr + 1), ext='png'), *size)

        y[iter] = exr_scaler_grayer(frmt.format(path=folder, file=ouput_name, val=iter + arrays_num, ext='exr'), *size)
    return x, y
