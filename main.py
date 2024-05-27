import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('digit-recognition2.keras')

n = 0
while os.path.isfile(f'digits/{n}.png'):
    img = cv2.imread(f'digits/{n}.png')[:,:,0]
    img = np.array([img])
    pred = model.predict(img)
    print(f"Prediction: {np.argmax(pred)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    n += 1