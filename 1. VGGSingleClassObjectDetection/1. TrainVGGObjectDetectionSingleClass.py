import os
from utils.PrepareData import PrepareData

TestDataPath = os.path.join("Dataset", "test")
TrainDataPath = os.path.join("Dataset", "train")

PrepareData_handle = PrepareData(TrainDataPath,TestDataPath,isXMLFiles=True)

TrainData,TrainTargets,TestData,TestTargets = PrepareData_handle.PrepareDataTargets()


from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input

# Imagenet is a competition every year held and VGG16 is winner of between  2013-14
# so here we just want limited layers so thats why we false included_top 
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
bboxhead = Dense(4,activation="relu")(bboxhead)

# lets import Model
from tensorflow.keras.models import Model
model = Model(inputs = vgg.input,outputs = bboxhead)

model.summary()

# Lets fit our model 
# Optimization 
from tensorflow.keras.optimizers import Adam
opt = Adam(1e-4)
model.compile(loss='mse',optimizer=opt)


history = model.fit(TrainData,TrainTargets,validation_data=(TestData,TestTargets),batch_size=16,epochs=50,verbose=1)

# lets save model 
model.save('detect_apples.h5')
