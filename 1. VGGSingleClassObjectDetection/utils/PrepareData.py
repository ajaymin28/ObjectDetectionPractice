from utils.XMLParser import XMLParser
import glob
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

class PrepareData:

    def __init__(self, trainPath, testPath, isXMLFiles=True, isJsonFiles=False) -> None:

        self.trainPath = trainPath
        self.testPath = testPath
        self.TestAnnotations = []
        self.TrainAnnotations = []

    def __getAnnotationsFiles(self, trainPath, testPath):
        TestDataAnnotation = glob.glob(testPath + os.sep + "*.xml")
        TrainDataAnnotation = glob.glob(trainPath + os.sep + "*.xml")
        return TrainDataAnnotation,TestDataAnnotation


    def __prepareAnnotations(self, TrainAnnotationsFiles, TestAnnotationsFiles):

        TrainAnnotationDataLength = len(TrainAnnotationsFiles)

        for annotFile in TrainAnnotationsFiles:
            Parser = XMLParser(annotFile)
            fileName, Annotations = Parser.getAnnotations()
            self.TrainAnnotations.append(Annotations)

        PreparedTrainAnnotationLength = len(self.TrainAnnotations)

        if TrainAnnotationDataLength==PreparedTrainAnnotationLength:
            print(f"[{__name__}][Info] Prepared Train data: {PreparedTrainAnnotationLength}")
        else:
            print(f"[{__name__}][Error] Train Annotations length doesn't match. Prepared/Total => {PreparedTrainAnnotationLength}/{TrainAnnotationDataLength}")


        TestAnnotationDataLength = len(TestAnnotationsFiles)

        for annotFile in TestAnnotationsFiles:
            Parser = XMLParser(annotFile)
            fileName,Annotations = Parser.getAnnotations()
            self.TestAnnotations.append(Annotations)

        PreparedTestAnnotationLength = len(self.TestAnnotations)

        if TestAnnotationDataLength==PreparedTestAnnotationLength:
            print(f"[{__name__}][Info] Prepared Test data: {PreparedTestAnnotationLength}")
        else:
            print(f"[{__name__}][Error] Test Annotations length doesn't match. Prepared/Total => {PreparedTestAnnotationLength}/{TestAnnotationDataLength}")


    def getAnnotations(self):

        self.TrainDataAnnotation,self.TestDataAnnotation = self.__getAnnotationsFiles(self.trainPath,self.testPath)
        self.__prepareAnnotations(self.TrainDataAnnotation,self.TestDataAnnotation)

        return self.TrainAnnotations, self.TestAnnotations



    def getfilePath(self, file_name, isTrain=True):
        filepath = None
        if isTrain:
            filepath = os.path.join(self.trainPath,file_name)
        else:
            filepath = os.path.join(self.testPath,file_name)
        
        return filepath


    def PrepareDataTargets(self):

        TrainAnnotations, TestAnnotations = self.getAnnotations()

        TestTargets = []
        TrainTargets = []

        TrainData = []
        TestData = []

        for TAnnotInstance in TestAnnotations:

            fname = list(TAnnotInstance.keys())[0]
            filePath = self.getfilePath(file_name=fname,isTrain=False)

            image=cv2.imread(filePath)
            (h,w)=image.shape[:2]

            image=load_img(filePath,target_size=(224,224))
            image=img_to_array(image)

            listOfAnnotations = TAnnotInstance[fname]["annotations"]
            # imageSize = TAnnotInstance[fname]["size"]

            for AnnotData in listOfAnnotations:

                startX, startY, endX, endY = AnnotData["box"]
                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                TestTargets.append((startX,startY,endX,endY))
                TestData.append(image)


        for TrainAnnotInstance in TrainAnnotations:

            fname = list(TrainAnnotInstance.keys())[0]
            filePath = self.getfilePath(file_name=fname,isTrain=True)

            image=cv2.imread(filePath)
            (h,w)=image.shape[:2]

            image=load_img(filePath,target_size=(224,224))
            image=img_to_array(image)

            listOfAnnotations = TrainAnnotInstance[fname]["annotations"]
            # imageSize = TrainAnnotInstance[fname]["size"]
            

            for AnnotData in listOfAnnotations:

                startX, startY, endX, endY = AnnotData["box"]
                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                TrainTargets.append((startX,startY,endX,endY))
                TrainData.append(image)

        TrainData=np.array(TrainData,dtype='float32') / 255.0
        TestData=np.array(TestData,dtype='float32') / 255.0

        TrainTargets=np.array(TrainTargets,dtype='float32')
        TestTargets=np.array(TestTargets,dtype='float32')

        return TrainData,TrainTargets,TestData,TestTargets


                
                





        