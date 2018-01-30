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
    (R,G,B) = [array.array('f', img.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    np_img = np.array([(r +g + b) / 3 for r, g, b in zip(R, G, B)])
    np_img.resize(sz[1], sz[0])
    np_img = cv2.resize(np_img, (x, y), interpolation=cv2.INTER_CUBIC)
    return np_img

def png_scaler_grayer(file, x, y):
    img = cv2.imread(file)
    img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, dstCn=cv2.COLOR_BGR2GRAY)
    return img