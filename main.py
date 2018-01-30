from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, concatenate, Conv2D, Conv1D, GRU
from keras.preprocessing.image import ImageDataGenerator, load_img
import OpenEXR
import Imath
import array

image_size = (160, 120)
image_size_line = image_size[0] * image_size[1]

inputs = [Input(shape=(1, *image_size), name="image_{0}".format(i)) for i in range(3)]
paths = [Reshape(image_size)(layer) for layer in inputs]
paths = [LSTM(image_size_line//16)(layer) for layer in paths]
for i in range(3):
    paths = [Dense(image_size_line//8, activation='relu')(layer) for layer in paths]
merge = concatenate(paths)
for i in range(3):
    merge = Dense(image_size_line//16, activation='relu')(merge)
# merge = Conv1D(32, activation='relu', input_shape=(32,32), kernel_size=(4))(merge)
# merge = LSTM(768)(merge)
merge = Reshape((image_size[0]//4, image_size[1]//4))(merge)
merge = LSTM(image_size_line//16)(merge)
out = Dense(image_size_line, activation='softmax')(merge)
out_reshape = Reshape(image_size)(out)
model = Model(inputs=inputs, outputs=out_reshape)

print(model.input)
