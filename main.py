from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Reshape, LSTM, concatenate, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, ProgbarLogger, RemoteMonitor
from tools import create_database, save_dataset_file, read_dataset_file, get_model_memory_usage, generate_batch
import numpy as np

image_size = (120, 160)
image_size_line = image_size[0] * image_size[1]
data_file = 'dataset_1.h5'
model_file = 'second_model.json'

read_data = True

read_model = False


if not read_data:
    training = create_database(folder='/home/tekatod/Pictures/render')
    validation =  create_database(folder='/home/tekatod/Pictures/test')
    print('Databases created')
    save_dataset_file(data_file, training, validation)
    print('Databases saved')
else:
    training, validation = read_dataset_file('dataset_1.h5')
    print('Databases read')


if not read_model:
    inputs = [Input(shape=(image_size), name="image_{0}".format(i)) for i in range(3)]
    paths = [Reshape((*image_size, 1))(layer) for layer in inputs]
    paths = [Conv2D(32, input_shape=image_size, kernel_size=(8, 8))(layer) for layer in paths]
    paths = [Conv2D(32, kernel_size=(4, 4))(layer) for layer in paths]
    paths = [MaxPooling2D()(layer) for layer in paths]
    paths = [Flatten()(layer) for layer in paths]
    paths = [Dense(image_size_line//16, activation='relu')(layer) for layer in paths]
    paths = [Reshape((image_size[0]//4, image_size[1]//4))(layer) for layer in paths]
    paths = [LSTM(image_size_line//16)(layer) for layer in paths]
    merge = concatenate(paths)
    merge = Dense(image_size_line//16, activation='relu')(merge)
    merge = Reshape((image_size[0]//4, image_size[1]//4, 1))(merge)
    merge = Conv2D(32, kernel_size=(4, 4))(merge)
    merge = Conv2D(64, kernel_size=(4, 4))(merge)
    merge = Conv2D(128, kernel_size=(4, 4))(merge)
    merge = MaxPooling2D()(merge)
    merge = Flatten()(merge)
    merge = Dense(image_size_line//16, activation='relu')(merge)
    merge = Reshape((image_size[0]//4, image_size[1]//4))(merge)
    merge = LSTM(image_size_line//16)(merge)
    out = Dense(image_size_line, activation='relu')(merge)
    out_reshape = Reshape(image_size)(out)
    model = Model(inputs=inputs, outputs=out_reshape)
    plot_model(model, to_file='second_model.png', show_layer_names=True, show_shapes=True)
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    print('Model saved')
else:
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

model.compile(optimizer='adagrad', loss='mean_squared_error')
print('Model created')

# checkpointer = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
# logger = ProgbarLogger(count_mode='samples')
# remote = RemoteMonitor(root='192.168.0.157:9000')


print('Model compiled, start training...')
# model.fit(training[0], training[1], epochs=250, batch_size=1, validation_data=validation, callbacks=[
#     ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True),
#     ProgbarLogger(count_mode='samples'),
#     RemoteMonitor(root='192.168.0.157:9000')
# ])

model.fit_generator(generate_batch(training), steps_per_epoch=248, epochs=500 , validation_data=generate_batch(validation), validation_steps=248, callbacks=[
    ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True),
    RemoteMonitor(root='192.168.0.157:9000')
])


# input_0 = Input(shape=(2,), name='input')
# path=Dense(4, activation='relu')(input_0)
# out=Dense(1, activation='linear')(path)
# model=Model(inputs=input_0, outputs=out)
# model.compile(optimizer='nadam', loss='mean_squared_error')
#
# in_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# out_data = np.array([0, 1, 1, 0])
# logger.set_model(model)
# remote.set_model(model)
#
# model.fit(in_data, out_data, epochs=25, batch_size=2)
