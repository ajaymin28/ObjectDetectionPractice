
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import os
import glob
import numpy as np
import cv2

model=load_model('detect_apples.h5')

TestDataPath = os.path.join("Dataset", "test")
TrainDataPath = os.path.join("Dataset", "train")

TestImages = glob.glob(TestDataPath+ os.sep + "*.jpg")

imagepath=TestImages[0]

image = load_img(imagepath,
                 target_size=(224,224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image,axis=0)

preds=model.predict(image)[0]
(startX,startY,endX,endY)=preds

image=cv2.imread(imagepath)
(h,w)=image.shape[:2]

startX=int(startX * w)
startY=int(startY * h)

endX=int(endX * w)
endY=int(endY * h)

cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),3)

cv2.imwrite("preds.jpg", image)