import keras
import tensorflow 
import numpy as np
import os
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt

filename = r"killqueen\predict"
model = keras.models.load_model(r"killqueen\killqueen.h5")

img = []
simg = []
for i in os.listdir(filename):
    img.append(cv2.imread(os.path.join(filename, i)))
    simg.append(cv2.resize(img[-1], (32,32), interpolation=cv2.INTER_AREA))
simg = np.array(simg).astype("float32")/255.0

pre = model.predict(simg)
pre = np.argmax(pre, axis=1)

for i in range(len(pre)):
    cv2.imshow(f"{pre[i]}", img[i])
    cv2.waitKey(0)