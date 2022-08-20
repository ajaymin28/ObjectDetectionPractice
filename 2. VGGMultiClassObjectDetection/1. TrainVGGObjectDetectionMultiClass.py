import os
from utils.PrepareData import PrepareData
import time

TestDataPath = os.path.join("Dataset", "test")
TrainDataPath = os.path.join("Dataset", "train")

PrepareData_handle = PrepareData(TrainDataPath,TestDataPath,isXMLFiles=True)

TrainData,TrainTargets,TestData,TestTargets = PrepareData_handle.PrepareDataTargets()

NumberOfClasses = len(PrepareData_handle.class_index_map)
print("*"*10)
print("Number of Classes: ",NumberOfClasses)
print("*"*10)


from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input

vgg=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
vgg.summary()


from tensorflow.keras.layers import Input,Flatten,Dense

# we use VGG16 as per our requirement not use whole 
vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

# Lets make bboxhead
bboxhead = Dense(128,activation="relu")(flatten)
bboxhead = Dense(64,activation="relu")(bboxhead)
bboxhead = Dense(32,activation="relu")(bboxhead)
bboxhead = Dense(4,activation="relu", name='bboxhead')(bboxhead) # 4 box values
softMaxhead = Dense(NumberOfClasses,activation="softmax",name='class_pred')(flatten) # Softmax layer for classification of object

# lets import Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model = Model(inputs = vgg.input,outputs = [bboxhead, softMaxhead])
model.summary()

from utils.CustomLossFunctions import custom_loss_function_bb, custom_loss_function_softmax

opt = Adam(1e-4)
model.compile(loss={"class_pred": custom_loss_function_softmax, "bboxhead": custom_loss_function_bb},
optimizer=opt,
run_eagerly=True)

start = time.perf_counter()
history = model.fit(TrainData,TrainTargets,validation_data=(TestData,TestTargets),batch_size=16,epochs=100,verbose=1)
end = time.perf_counter()
print(f"Completed in : {end-start}")
# lets save model
model.save('detect_apples_multiclass.h5')