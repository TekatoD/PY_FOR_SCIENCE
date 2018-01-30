from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, concatenate, Conv2D, Conv1D, GRU
from keras.preprocessing.image import ImageDataGenerator, load_img
import OpenEXR
import Imath
import array

inputs = [Input(shape=(1, 32, 24), name="image_{0}".format(i)) for i in range(3)]
paths = [Reshape((32, 24))(layer) for layer in inputs]
paths = [LSTM(768)(layer) for layer in paths]
paths = [Dense(768, activation='relu')(layer) for layer in paths]
for i in range(2):
    paths = [Dense(768, activation='relu')(layer) for layer in paths]
merge = concatenate(paths)
for i in range(3):
    merge = Dense(768, activation='relu')(merge)
# merge = Conv1D(32, activation='relu', input_shape=(32,32), kernel_size=(4))(merge)
# merge = LSTM(768)(merge)
merge = Reshape((32, 24))(merge)
merge = GRU(768)(merge)
out = Dense(768, activation='relu')(merge)
out_reshape = Reshape((32, 24))(out)
model = Model(inputs=inputs, outputs=out_reshape)

print(model.input)
