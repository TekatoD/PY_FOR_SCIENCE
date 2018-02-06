from tools import read_dataset_file
from keras.models import model_from_json, load_model
import cv2
import numpy as np

training, validation = read_dataset_file('dataset_1.h5')

print('images read')

# json_file = open('second_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# print('model read')

model = load_model('weights.13-16.03.hdf5')

print('wights and model read')


print('predicting')

ind = 0
for f, s, t, a in zip(training[0][0], training[0][1], training[0][2], training[1]):
    answer = model.predict([np.array([f]), np.array([s]), np.array([t])])
    answer = cv2.normalize(answer[0], answer[0], 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)

    x1 = cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    x2 = cv2.normalize(s, s, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    x3 = cv2.normalize(t, t, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    x4 = cv2.normalize(a, a, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    cv2.imwrite('answers/image' + str(ind) + '_x1.jpg', x1)
    cv2.imwrite('answers/image' + str(ind) + '_x2.jpg', x2)
    cv2.imwrite('answers/image' + str(ind) + '_x3.jpg', x3)
    cv2.imwrite('answers/image' + str(ind) + '_x4.jpg', x4)

    cv2.imwrite('answers/image' + str(ind) + '.jpg', answer)
    ind += 1
