from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, concatenate, Conv1D, Conv2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from tools import create_database, read_camera

image_size = (120, 160)
image_size_line = image_size[0] * image_size[1]

camera = read_camera('/home/arssivka/3d/stair/render/', samples_num=2)
# training = create_database(folder='/home/arssivka/3d/stair/render', samples_num=2)
# validation = create_database(folder='/home/arssivka/3d/stair/test', samples_num=2)

print(camera)

# print(training[0][0].shape)

print('Databases created')

inputs= [Input(shape=(image_size), name="in_image"),
         Input(shape=(6,), name="in_camera")]

cam_branch = inputs[1]

img_branch = inputs[0]
img_branch = Reshape((*image_size, 1))(img_branch)
for i in range(4):
    img_branch = Conv2D(32, activation='relu', input_shape=(120, 160), kernel_size=(4,4))(img_branch)
img_branch = Dense(image_size_line//8, activation='relu')(img_branch)

img_branch = LSTM(image_size_line//16)(img_branch)
img_branch = Dense(image_size_line//8, activation='relu')(img_branch)
# merge = LSTM(768)(merge)
merge = concatenate([img_branch, cam_branch])

merge = Reshape((image_size[0]//4, image_size[1]//4))(merge)
merge = LSTM(image_size_line//16)(merge)
out = Dense(image_size_line, activation='softmax')(merge)
out_reshape = Reshape(image_size)(out)
model = Model(inputs=inputs, outputs=out_reshape)
plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)

print('Model created')
#
# model.compile(optimizer='nadam', loss='binary_crossentropy')
#
# print('Model compiled, start training...')
# model.fit(training[0], training[1], epochs=50, batch_size=5, validation_data=validation, callbacks=[
#     ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True),
#     ProgbarLogger(count_mode='samples')
# ])
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
