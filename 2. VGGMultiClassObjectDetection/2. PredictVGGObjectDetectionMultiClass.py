from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import os
import glob
import numpy as np
import cv2

from utils.CustomLossFunctions import custom_loss_function_bb, custom_loss_function_softmax

model= load_model('detect_apples_multiclass.h5',custom_objects={"custom_loss_function_softmax": custom_loss_function_softmax,
                                                                "custom_loss_function_bb": custom_loss_function_bb})

TestDataPath = os.path.join("Dataset", "test")
TrainDataPath = os.path.join("Dataset", "train")

TestImages = glob.glob(TestDataPath+ os.sep + "*.jpg")

OutputPath = os.path.join("Output")

if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

for imgp in TestImages:

    filename = imgp.split("\\")[-1]

    image = load_img(imgp,
                    target_size=(224,224))
    image = img_to_array(image) / 255.0
    # image = preprocess_input(imagde)
    image = np.expand_dims(image,axis=0)

    boxes, preds = model.predict(image)

    (startX,startY,endX,endY)=boxes[0]

    # preds = softmax(preds)
    maxIndex = np.argmax(preds, axis=-1)
    probability = preds[0,maxIndex]

    print(filename, maxIndex, probability, boxes)

    image=cv2.imread(imgp)
    (h,w)=image.shape[:2]

    startX=int(startX * w)
    startY=int(startY * h)

    endX=int(endX * w)
    endY=int(endY * h)

    cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),3)
    
    FullImagePath = f"{OutputPath}/{maxIndex}_{filename}" 
    cv2.imwrite(FullImagePath, image)