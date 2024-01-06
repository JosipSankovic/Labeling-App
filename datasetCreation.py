import cv2
import os
import shutil

class DatasetCreator:
    def __init__(self):
        self._path=None
        self._images=[]
        self._currentImage=0
    def loadDirectory(self, path):
        self._path=path
        self._images = []
        for filename in os.listdir(self._path+"/"):
            if filename.endswith(('.jpg','.png','.jpeg')):
                self._images.append(filename)
        self._currentImage = 0

    # create dataset with 70/20/10 split that puts images in train/val/test folders
    def createDataset(self,imgSize=None):
        #create folders
        datasetFolder=os.path.join(self._path, "dataset")
        trainPath = os.path.join(datasetFolder, "train")
        valPath = os.path.join(datasetFolder, "val")
        testPath = os.path.join(datasetFolder, "test")
        if not os.path.exists(datasetFolder):
            os.mkdir(datasetFolder)
        if not os.path.exists(trainPath):
            os.mkdir(trainPath)
        if not os.path.exists(valPath):
            os.mkdir(valPath)
        if not os.path.exists(testPath):
            os.mkdir(testPath)
        
        #make random split
        import random
        random.shuffle(self._images)
        trainCount = int(len(self._images)*0.7)
        valCount = int(len(self._images)*0.2)
        testCount = len(self._images)-trainCount-valCount
        #move images to folders
        for i in range(trainCount):
            self._checkAndCopyLabelAndImg(i,trainPath,imgSize=imgSize)
        for i in range(trainCount, trainCount+valCount):
            self._checkAndCopyLabelAndImg(i,valPath,imgSize=imgSize)
        for i in range(trainCount+valCount, trainCount+valCount+testCount):
            self._checkAndCopyLabelAndImg(i,testPath,imgSize=imgSize)

    def _checkAndCopyLabelAndImg(self,i,path,imgSize=None):
            #check if there is a label file
            filename=os.path.splitext(self._images[i])[0]+'.txt'
            if not os.path.exists(os.path.join(self._path, filename)):
                return
            
            if(not os.path.exists(os.path.join(path,"images"))):
                os.mkdir(os.path.join(path,"images"))
            if(not os.path.exists(os.path.join(path,"labels"))):
                os.mkdir(os.path.join(path,"labels"))

            if(imgSize!=None):
                img=cv2.imread(os.path.join(self._path, self._images[i]))
                img=cv2.resize(img, imgSize)
                cv2.imwrite(os.path.join(path,"images", self._images[i]),img)
            else:
                #copy image
                shutil.copyfile(os.path.join(self._path, self._images[i]), os.path.join(path,"images", self._images[i]))
            #copy label
            shutil.copyfile(os.path.join(self._path, filename), os.path.join(path,"labels", filename))