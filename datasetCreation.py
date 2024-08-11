import cv2
import os
import shutil
import random
import numpy as np
class DatasetCreator:
    __path=None
    __dataset=[]
    __imgSize=None
    def __init__(self):
        self.__path=None
        self.__dataset=[]
        self.__imgSize=(-1,-1)
    def loadDataset(self,path):
        self.__path=path
        for filename in os.listdir(path=path):
            if filename.endswith(('.jpg','.png','.jpeg')) and os.path.exists(os.path.join(path,os.path.splitext(filename)[0]+".txt")):
                self.__dataset.append({"img":filename,"label":os.path.splitext(filename)[0]+".txt"})

        return len(self.__dataset)

    def createDataset(self,imgSize=(-1,-1),mozaicMix=-1,split=(0.70,0.20,0.10),noisePercent=-1,flip_horizontaly=False,brightness_percent=0.6,contrast=True):
        self.__imgSize=imgSize
        datasetFolder=os.path.join(self.__path, "dataset")
        trainPath = os.path.join(datasetFolder, "train")
        valPath = os.path.join(datasetFolder, "valid")
        testPath = os.path.join(datasetFolder, "test")
        if os.path.exists(datasetFolder):
            shutil.rmtree(datasetFolder)
        os.mkdir(datasetFolder)
        os.mkdir(trainPath)
        os.mkdir(valPath)
        os.mkdir(testPath)
        os.mkdir(os.path.join(trainPath,"images"))
        os.mkdir(os.path.join(trainPath,"labels"))
        os.mkdir(os.path.join(valPath,"images"))
        os.mkdir(os.path.join(valPath,"labels"))
        os.mkdir(os.path.join(testPath,"images"))
        os.mkdir(os.path.join(testPath,"labels"))
        random.shuffle(self.__dataset)
        dat_size=len(self.__dataset)
        train = self.__dataset[:int(dat_size*split[0])]
        val = self.__dataset[int(dat_size*split[0]):int(dat_size*(split[1]+split[0]))]
        test = self.__dataset[int(dat_size*(split[0]+split[1])):]
        for data in train:
            function_list=[]
            new_img=None
            yolo_labels=None
            img=data["img"]
            label=data["label"]
            if mozaicMix>0:
                function_list.append((self.__mozaicMix,"mix",img,label,mozaicMix))
            if noisePercent>0 and noisePercent<1.0:
                function_list.append((self.__addNoise,"noise",img,label,noisePercent))
            if flip_horizontaly:
                function_list.append((self.__flipImageHorizontaly,"flip",img,label))
            if brightness_percent>0 and brightness_percent<1.0:
                if(random.randint(0,100)>55):
                    function_list.append((self.__brightness,"brightness",img,label,100*brightness_percent*(-1),1))
                else:
                    function_list.append((self.__brightness,"brightness",img,label,100*brightness_percent,1))
            if contrast:
                if(random.randint(0,100)>50):
                    function_list.append((self.__brightness,"contrast",img,label,0,1.5))
                else:
                    function_list.append((self.__brightness,"contrast",img,label,0,0.7))
            for fun in function_list:
                new_img,yolo_labels=fun[0](*fun[2:])
                cv2.imwrite(os.path.join(trainPath,"images", os.path.splitext(img)[0]+f"_{fun[1]}.jpg"),new_img)
                self.__saveLabels(yolo_labels,os.path.join(trainPath,"labels", os.path.splitext(img)[0]+f"_{fun[1]}.txt"))
            image=self.readImage(os.path.join(self.__path,img))
            cv2.imwrite(os.path.join(trainPath,"images",img),image)
            shutil.copyfile(os.path.join(self.__path, label), os.path.join(trainPath,"labels",label))
        for data in val:
            img=data["img"]
            label=data["label"]
            image=self.readImage(os.path.join(self.__path,img))
            cv2.imwrite(os.path.join(valPath,"images",img),image)
            shutil.copyfile(os.path.join(self.__path, label), os.path.join(valPath,"labels",label))
        for data in test:
            img=data["img"]
            label=data["label"]
            image=self.readImage(os.path.join(self.__path,img))
            cv2.imwrite(os.path.join(testPath,"images",img),image)
            shutil.copyfile(os.path.join(self.__path, label), os.path.join(testPath,"labels",label))
        self.createYAMLFile()
    def __saveLabels(self,labels,path):
        with open(path, 'a') as outfile:
                for addedLabel in labels:
                    x,y,width,height=addedLabel['points']
                    outfile.write(str(addedLabel["className"])+" "+f'{x:.5f}'+' '+f'{y:.5f}'+' '+f'{width:.5f}'+' '+f'{height:.5f}'+'\n')
        outfile.close() 
    def __mozaicMix(self,image_path,label_path,numberOfPoints=2):
        randomIndex=random.randint(0,len(self.__dataset)-1)
        allPoints=self.loadLabels(os.path.join(self.__path,self.__dataset[randomIndex]["label"]))
        addedLabels=[]
        while(allPoints is None):
            randomIndex=random.randint(0,len(self.__dataset)-1)
            allPoints=self.loadLabels(os.path.join(self.__path,self.__dataset[randomIndex]["label"]))
        
        img=self.readImage(os.path.join(self.__path, self.__dataset[randomIndex]["img"]))
        image=self.readImage(os.path.join(self.__path, image_path))
        labels=self.loadLabels(os.path.join(self.__path, label_path))
        for label in labels:
            x1,y1=label["points"][0]
            x2,y2=label["points"][1]
            x1=x1*image.shape[1]
            y1=y1*image.shape[0]
            x2=x2*image.shape[1]
            y2=y2*image.shape[0]
            x,y,width,height=self.pointsToYolov8Format((x1,y1),(x2,y2),imgSize=image.shape)
            addedLabels.append({
                'className':label['className'],
                'points':[x,y,width,height]
            })
        randomPoints=random.sample(range(0,len(allPoints)),min(numberOfPoints,len(allPoints)))
        for randomPoint in randomPoints:
            if(randomPoint>=len(allPoints)):
                break
            p=allPoints[randomPoint]
            x1,y1=p['points'][0]
            x2,y2=p['points'][1]

            x1=x1*img.shape[1]
            y1=y1*img.shape[0]
            x2=x2*img.shape[1]
            y2=y2*img.shape[0]
            croppedClass=img[int(y1):int(y2),int(x1):int(x2)]
            if croppedClass.shape[1]<=2:
                continue
            x=0
            y=0
            maxIou=-1
            iteration_count=0
            while maxIou!=0 and iteration_count<3:
                x=random.randint(0,image.shape[1]-croppedClass.shape[1])
                y=random.randint(0,image.shape[0]-croppedClass.shape[0])
                for label in labels:
                    x11,y11=label["points"][0]
                    x21,y21=label["points"][1]
                    x11=x11*image.shape[1]
                    y11=y11*image.shape[0]
                    x21=x21*image.shape[1]
                    y21=y21*image.shape[0]
                    boxA=(x11,y11,x21,y21)
                    boxB=(x,y,x+croppedClass.shape[1],y+croppedClass.shape[0])
                    iou=self.calculateIoU(boxa=boxA,boxb=boxB)
                    if iou>maxIou:
                        maxIou=iou
                iteration_count+=1
            if maxIou>=0.001:
                continue
            labels.append({
                "className":p["className"],
                "points":[(x/img.shape[1],y/img.shape[0]),((x+croppedClass.shape[1])/img.shape[1],(y+croppedClass.shape[0])/img.shape[0])]
            })
            #zalijepi na originalnu sliku
            image[y:y+croppedClass.shape[0],x:x+croppedClass.shape[1]]=croppedClass
            x,y,width,height=self.pointsToYolov8Format((x,y),(x+croppedClass.shape[1],y+croppedClass.shape[0]),imgSize=image.shape)
            addedLabels.append({
                'className':p['className'],
                'points':[x,y,width,height]
            })
        return [image,addedLabels]
    def __addNoise(self,image_path,label_path=None,percentOfNoise=0.02):
        noisy_image=self.readImage(os.path.join(self.__path,image_path))
        yoloLabels=None
        if label_path is not None:
            labels=self.loadLabels(os.path.join(self.__path,label_path))
            yoloLabels=[]
            for label in labels:
                x1,y1=label['points'][0]
                x2,y2=label['points'][1]
                x,y,width,height=self.pointsToYolov8Format((x1,y1),(x2,y2))
                yoloLabels.append({
                    'className':label['className'],
                    'points':[x,y,width,height]
                })
        num_dots=int(noisy_image.shape[0]*noisy_image.shape[1]*percentOfNoise)
        x_coords = np.random.randint(0, noisy_image.shape[1], num_dots//2)
        y_coords = np.random.randint(0, noisy_image.shape[0], num_dots//2)
        noisy_image[y_coords, x_coords] = [0, 0, 0]
        x_coords = np.random.randint(0, noisy_image.shape[1], num_dots//2)
        y_coords = np.random.randint(0, noisy_image.shape[0], num_dots//2)
        noisy_image[y_coords, x_coords] = [255, 255, 255]
        return [noisy_image,yoloLabels]
    def __flipImageHorizontaly(self,image_path,label_path):
        image=self.readImage(os.path.join(self.__path,image_path))
        image=cv2.flip(image,1)
        labels=self.loadLabels(os.path.join(self.__path,label_path))
        new_labels=[]
        for label in labels:
            x1,y1=label['points'][0]
            x2,y2=label['points'][1]
            x1=1-x1
            x2=1-x2
            x,y,width,height=self.pointsToYolov8Format((x1,y1),(x2,y2))
            new_labels.append({
                'className':label['className'],
                'points':[x,y,width,height]
            })

        return [image,new_labels]
    def __brightness(self,image_path,label_path=None,brightness=60,contrast=1.5):
        image=self.readImage(os.path.join(self.__path,image_path))
        labels=self.loadLabels(os.path.join(self.__path,label_path))
        yoloLabels=None
        if label_path is not None:
            yoloLabels=[]
            for label in labels:
                x1,y1=label['points'][0]
                x2,y2=label['points'][1]
                x,y,width,height=self.pointsToYolov8Format((x1,y1),(x2,y2))
                yoloLabels.append({
                    'className':label['className'],
                    'points':[x,y,width,height]
                })
        image=cv2.addWeighted(image,contrast,np.zeros(image.shape,image.dtype),0,brightness)
        return [image,yoloLabels]
    def loadLabels(self,label):
        filename=label
        if not os.path.exists(os.path.join(self.__path, filename)):
            return None
        allPoints=[]
        with open(os.path.join(self.__path, filename)) as infile:
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
    def readImage(self,path):
        with open(path, 'rb') as f:
                image_bytes = f.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image= cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if self.__imgSize !=(-1,-1):
                image=cv2.resize(image,self.__imgSize)
        return image
    def createYAMLFile(self):
        classes=[]
        with open(os.path.join(self.__path,"labels.txt"),"r") as file:
            for line in file:
                classes.append(line.split("\n")[0])
        file.close()

        with open(os.path.join(self.__path,"dataset","data.yaml"),"w") as yaml_file:
            yaml_file.write("names:\n")
            for cl in classes:
                yaml_file.write(f"- \'{cl}\'\n")
            yaml_file.write(f"nc: {str(len(classes))}\n")
            yaml_file.write("train: /content/dataset/train/images\n")
            yaml_file.write("val: /content/dataset/valid/images\n")
            yaml_file.write("test: /content/dataset/test/images\n")
        yaml_file.close()


    def calculateIoU(self,boxa,boxb):
        
        x1_min, y1_min, x1_max, y1_max = boxa
        x2_min, y2_min, x2_max, y2_max = boxb

        
        # Calculate the coordinates of the intersection rectangle
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        # Calculate the area of the intersection
        inter_width = max(0, x_inter_max - x_inter_min + 1)
        inter_height = max(0, y_inter_max - y_inter_min + 1)
        intersection_area = inter_width * inter_height
        
        # Calculate the areas of the bounding boxes
        boxa_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        boxb_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
        
        # Calculate the area of union
        union_area = boxa_area + boxb_area - intersection_area
        
        # Calculate and return IOU
        iou = intersection_area / union_area
        return iou