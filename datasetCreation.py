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
                #make mozaic on image
                img,addedLabels=self.mozaicMix(img)
                #save image
                cv2.imwrite(os.path.join(path,"images", self._images[i]),img)
                #copy label
                shutil.copyfile(os.path.join(self._path, filename), os.path.join(path,"labels", filename))
                #add addedLabels to label file
                with open(os.path.join(path,"labels", filename), 'a') as outfile:
                    for addedLabel in addedLabels:
                        x,y,width,height=addedLabel['points']
                        outfile.write(str(addedLabel["className"])+" "+f'{x:.5f}'+' '+f'{y:.5f}'+' '+f'{width:.5f}'+' '+f'{height:.5f}'+'\n')
                outfile.close()
            else:
                #copy image
                shutil.copyfile(os.path.join(self._path, self._images[i]), os.path.join(path,"images", self._images[i]))
                #copy label
                shutil.copyfile(os.path.join(self._path, filename), os.path.join(path,"labels", filename))

    def mozaicMix(self,image,numberOfPoints=2):
        #take random image with labels
        import random
        randomIndex=random.randint(0,len(self._images)-1)
        allPoints=self.loadLabels(randomIndex)
        addedLabels=[]
        while(allPoints is None):
            randomIndex=random.randint(0,len(self._images)-1)
            allPoints=self.loadLabels(randomIndex)
        img=cv2.imread(os.path.join(self._path, self._images[randomIndex]))
        #get random numbers
        randomPoints=random.sample(range(0,len(allPoints)),numberOfPoints)
        for randomPoint in randomPoints:
            if(randomPoint>=len(allPoints)):
                break
            p1=allPoints[randomPoint]
            #crop that rectangle
            x1,y1=p1['points'][0]
            x2,y2=p1['points'][1]

            x1=x1*img.shape[1]
            y1=y1*img.shape[0]
            x2=x2*img.shape[1]
            y2=y2*img.shape[0]
            croppedClass=img[int(y1):int(y2),int(x1):int(x2)]
            #paste it on image
            x=random.randint(0,image.shape[1]-croppedClass.shape[1])
            y=random.randint(0,image.shape[0]-croppedClass.shape[0])

            image[y:y+croppedClass.shape[0],x:x+croppedClass.shape[1]]=croppedClass
            x,y,width,height=self.pointsToYolov8Format((x,y),(x+croppedClass.shape[1],y+croppedClass.shape[0]),imgSize=image.shape)
            addedLabels.append({
                'className':p1['className'],
                'points':[x,y,width,height]
            })
        #return image
        return [image,addedLabels]

    def loadLabels(self,index):
        filename=os.path.splitext(self._images[index])[0]+'.txt'
        if not os.path.exists(os.path.join(self._path, filename)):
            return None
        allPoints=[]
        with open(os.path.join(self._path, filename)) as infile:
            for line in infile:
                className,x,y,width,height=line.split()
                x=float(x)
                y=float(y)
                width=float(width)
                height=float(height)
                x1=x-width/2
                y1=y-height/2
                x2=x+width/2
                y2=y+height/2
                allPoints.append({
                    'className':int(className),
                    'points':[(x1,y1),(x2,y2)]
                })
        return allPoints
    def pointsToYolov8Format(self,p1,p2,imgSize=None):
        x1,y1=p1
        x2,y2=p2
        if imgSize is not None:
            x1=x1/imgSize[1]
            y1=y1/imgSize[0]
            x2=x2/imgSize[1]
            y2=y2/imgSize[0]
        x=(x1+x2)/2
        y=(y1+y2)/2
        width=abs(x1-x2)
        height=abs(y1-y2)
        return [x,y,width,height]