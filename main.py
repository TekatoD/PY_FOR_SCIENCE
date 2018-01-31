from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, concatenate
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from tools import create_database

image_size = (120, 160)
image_size_line = image_size[0] * image_size[1]

training = create_database(folder='/home/tekatod/Pictures/render')
validation =  create_database(folder='/home/tekatod/Pictures/test')

print(training[0][0].shape)

print('Databases created')

inputs = [Input(shape=(image_size), name="image_{0}".format(i)) for i in range(3)]
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
plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)

print('Model created')

model.compile(optimizer='nadam', loss='binary_crossentropy')

print('Model compiled, start training...')
model.fit(training[0], training[1], epochs=50, batch_size=5, validation_data=validation, callbacks=[
    ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True),
    ProgbarLogger(count_mode='samples')
])
