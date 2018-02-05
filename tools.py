import OpenEXR
import Imath
import array

import h5py
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
    # cv2.normalize(np_img, np_img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
    return np_img

def png_scaler_grayer(file, x, y):
    img = cv2.imread(file)
    img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
    return img


def create_database(folder='folder', input_name='Image', ouput_name='Depth', samples_num=250, size=(160, 120)):
    arrays_num = 3
    frmt = "{path}/{file}{val:0>4}.{ext}"
    x = [np.empty((samples_num - arrays_num + 1, *reversed(size)), dtype=np.float32) for arr in range(arrays_num)]
    y = np.empty((samples_num - arrays_num + 1, *reversed(size)), dtype=np.float32)
    for iter in range(samples_num - arrays_num + 1):
        print(iter)
        for arr in range(arrays_num):
            if iter > 0 and arr + 1 < arrays_num:
                x[arr][iter] = x[arr + 1][iter - 1]
            else:
                x[arr][iter] = png_scaler_grayer(frmt.format(path=folder, file=input_name, val=(iter + arr + 1), ext='png'), *size)

        y[iter] = exr_scaler_grayer(frmt.format(path=folder, file=ouput_name, val=iter + arrays_num, ext='exr'), *size)
    return x, y

def save_dataset_file(file_name = 'dataset.h5', training = None, validation = None):
    dataset_file = h5py.File(file_name, 'w')
    def save_data(file, data, data_name):
        train_grp = file.create_group(data_name)
        for n, d in enumerate(data):
            train_grp.create_dataset(str(n), data=d)

    if training is not None:
        save_data(dataset_file, training, 'training')
    if validation is not None:
        save_data(dataset_file, validation, 'validation')

    dataset_file.close()

def read_dataset_file(file_name = 'dataset.h5', training_group = 'training', validation_group = 'validation'):
    dataset_file = h5py.File(file_name, 'r')
    def read(file, group_name):
        np_in = np.array(file[group_name].get('0'))
        return [arr for arr in np_in], np.array(file[group_name].get('1'))
        # return [[file[group_name].get(arr)] for arr in file[group_name].keys()]
    if training_group is not None:
        x = read(dataset_file, training_group)
    else:
        x = None
    if validation_group is not None:
        y = read(dataset_file, validation_group)
    else:
        y = None
    dataset_file.close()
    return  x, y

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def generate_batch(data, batch_length = 2):
    while True:
        for i in range(0, len(data[0][0]), batch_length):
            yield ([data[0][0][i:i + batch_length], data[0][1][i:i + batch_length], data[0][2][i:i + batch_length]],
                    data[1][i:i + batch_length])