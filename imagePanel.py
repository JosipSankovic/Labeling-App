import wx
import cv2
import os
from yolov8Detection import YOLOv8
import numpy as np
class ImagePanel(wx.Panel):
    __imagePath=None
    __originalImage=None
    __image=None
    __imageBitmap=None
    __points=None
    def __init__(self, parent,frameSize=wx.DefaultSize,position=wx.DefaultPosition):
        super(ImagePanel, self).__init__(parent,size=frameSize,pos=position)
        self.__imagePath = None
        self.__originalImage=None
        self.__image=None
        self.__imageBitmap = None
        self.__points=None
        self._classNumber=0
        # self._className="0"
        self.SetDoubleBuffered(True)
        # self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour("#2E2E2E")
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_KEY_DOWN,self.on_key_down)
    def LoadImage(self, path):
        if os.path.exists(path=path)==False:
            return
        self.__imagePath = path
        # Read the image file as a binary stream
        with open(self.__imagePath, 'rb') as f:
            image_bytes = f.read()
            
            # Convert the binary stream to a numpy array
            image_array = np.frombuffer(image_bytes, np.uint8)
            
            # Decode the image array
            self.__originalImage = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        self.__originalImage=cv2.cvtColor(self.__originalImage,cv2.COLOR_BGR2RGB)
        self.loadPointsFromLabel()
        if self.__originalImage is None:
            return False
        self.showFrame()   
    def on_paint(self, event):
        if self.__imageBitmap is not None:
            dc = wx.BufferedPaintDC(self)
            s_width, s_height = self.GetSize()
            b_width, b_height = self.__imageBitmap.GetSize()
            
            
            # Centriraj sliku
            x = (s_width - b_width) // 2
            y = (s_height - b_height) // 2
            dc.Clear()
            dc.DrawBitmap(self.__imageBitmap, x, y,True)
    def showFrame(self):
        if self.__originalImage is not None:
            #Metoda vraća novu širinu i visinu 
            #slike tako što se sačuva originalni omjer
            new_width,new_height=self.get_size_keep_aspect_ratio()    
            self.__image = cv2.resize(self.__originalImage, (new_width, new_height))
            # Crtanje objekata
            self.__points.drawObjects(self.__image)
            height, width, _ = self.__image.shape
            self.__imageBitmap = wx.Bitmap.FromBuffer(width, height, self.__image)
            self.Refresh()
    def get_size_keep_aspect_ratio(self):
        if self.__originalImage is not None:
            s_width, s_height = self.GetSize()
            i_height, i_width, _ = self.__originalImage.shape
        
            aspect_ratio = i_width / i_height
            
            if s_width / s_height > aspect_ratio:
                new_height = s_height
                new_width = int(s_height * aspect_ratio)
            else:
                new_width = s_width
                new_height = int(s_width / aspect_ratio)

            return (new_width,new_height)
        return None
    def convert_pos(self, pt):
        # veličina slike
        h,w,_=self.__image.shape
        # veličina panela
        W,H = self.GetSize()
        x,y=pt
        if w * H < W * h:
            s = w
            S = w * H // h
            o = W - S
            o //= 2
            x -= o
        else:
            s = h
            S = h * W // w
            o = H - S
            o //= 2
            y-= o

        x = x * s // S
        y = y * s // S
        return (x,y)
    def on_size(self, event):
        self.showFrame()
    def on_left_down(self, event):
        if self.__image is not None:
            x = event.GetX()
            y = event.GetY()
            height,width,_=self.__image.shape
            x,y=self.convert_pos((x,y))
            if x >=0 and x<=width and y >= 0 and y<=height:
                self.__points.addPoint((height,width),(x,y),self._classNumber)
                self.showFrame()

        event.Skip()
    def on_right_down(self, event):
        if self.__image is not None:
            x = event.GetX()
            y = event.GetY()
            height,width,_=self.__image.shape
            x,y=self.convert_pos((x,y))
            if x >=0 and x<=width and y >= 0 and y<=height:
                if self.__points.deleteObject((x,y),(height,width)):
                    self.showFrame()

        event.Skip()
    def drawImage(self):
        if self.__image is None:
            return False
    def getLabels(self):
        allPoints=self.__points.getAllPoints()
        pointsInJSON=[]
        for point in allPoints:
            pointsInJSON.append({
                'className':point.getClassName(),
                'points':point.getPoints()
            })
        return pointsInJSON
    def loadPointsFromLabel(self):
        self.__points=AllPointsHandler()
        filename=os.path.splitext(self.__imagePath)[0]+'.txt'
        allPoints=[]
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as infile:
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
                allPoints.append([className,x1,y1,x2,y2])
            infile.close()
        self.__points.LabelsToPoints(allPoints)
    def loadModel(self,path):
        self._model=YOLOv8(path)
    def detectPointsWithModel(self):
        if self._model is None:
            return False
        boxes, scores, class_ids=self._model(self.__originalImage)
        self.__points=AllPointsHandler()
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score<0.5:
                continue
            x1,y1,x2,y2=box
            self.__points.addPoint(self.__originalImage.shape,(x1,y1))
            self.__points.addPoint(self.__originalImage.shape,(x2,y2),class_id)
        self.showFrame()
    def setClassname(self,classId):
        self._classNumber=classId
    def on_key_down(self,event):
        keyPressed=event.GetKeyCode()
        if(keyPressed==wx.WXK_NUMPAD0):
            self._classNumber=0
        elif (keyPressed==wx.WXK_NUMPAD1):
            self._classNumber=1
        elif (keyPressed==wx.WXK_NUMPAD2):
            self._classNumber=2
        elif (keyPressed==wx.WXK_NUMPAD3):
            self._classNumber=3
        elif (keyPressed==wx.WXK_NUMPAD4):
            self._classNumber=4
        elif (keyPressed==wx.WXK_NUMPAD5):
            self._classNumber=5
        elif (keyPressed==wx.WXK_NUMPAD6):
            self._classNumber=6
        elif (keyPressed==wx.WXK_NUMPAD7):
            self._classNumber=7
        elif (keyPressed==wx.WXK_NUMPAD8):
            self._classNumber=8
        elif (keyPressed==wx.WXK_NUMPAD9):
            self._classNumber=9
        
        event.Skip()
class AllPointsHandler():
    __allPoints=[]
    __point=None
    def __init__(self):
        self.__allPoints=[]
        self.__point=Label()
    #dodaje point u listu svih pointova
    def addPoint(self,imgSize,point,className=0):
        last_point=self.__point.addPoint(point,imgSize,classId=className)
        if last_point==True:
            self.__allPoints.append(self.__point)
            self.__point=Label()
    #nacrtaj sve pointove kao pravokutnike
    def drawObjects(self,image):
        for point in self.__allPoints:
            point.drawRectangle(image)
        self.__point.drawRectangle(image=image)
    def deleteObject(self,point,imgSize):
        X,Y=point
        X=X/imgSize[1]
        Y=Y/imgSize[0]
        for points in self.__allPoints:
            x1,y1=points.getPoints()[0]
            x2,y2=points.getPoints()[1]
            if x1<=X<=x2 and y1<=Y<=y2:
                self.__allPoints.remove(points)
                return True
        return False

    def getAllPoints(self):
        return self.__allPoints
    def LabelsToPoints(self,allPoints):
        for point in allPoints:
            self.__point=Label()
            self.__point.addPoint((point[1],point[2]),classId=point[0])
            self.__point.addPoint((point[3],point[4]),classId=point[0])
            self.__allPoints.append(self.__point)
        self.__point=Label()
#prihvaca 2 tocke za pravokutnik koje su gornja lijeva i donja desna tocka
#sprema ih kao postotak širine i visine slike
class Label():
    __colorsForClasses = [
    (31, 119, 180),(255, 127, 14),(44, 160, 44), 
    (214, 39, 40),(148, 103, 189),(140, 86, 75),  
    (227, 119, 194),(127, 127, 127),(188, 189, 34), 
    (23, 190, 207),(255, 187, 120),(199, 199, 199),
    (158, 218, 229),(197, 176, 213),(140, 140, 140),
    (196, 156, 148),(227, 119, 180),(174, 199, 232),
    (152, 223, 138),(246, 207, 113)
]
    __className=None
    __lastPoint=None
    __points=None
    def __init__(self,className=0):
        self.__className=className
        self.__lastPoint=False
        self.__points={0:(-1,-1),1:(-1,-1)}

    
    def addPoint(self,point,imgSize=None,classId=None):
        if imgSize is not None:
            point=self.__pointsToPercent(imgSize,point)

        if classId is not None:
            self.__className=int(classId)
        #prva tocka
        if self.__lastPoint==False:
            self.__points[0]=point
            self.__lastPoint=True
        else:
            self.__points[1]=point
            self.__rearrange_points()
            self.__lastPoint=False
        return not self.__lastPoint
    
    def getPoints(self):
        return self.__points
    def getClassName(self):
        return self.__className
    def setClassName(self,className):
        self.__className=className
    def __pointsToPercent(self,imageSize,point):
        return (point[0]/imageSize[1],point[1]/imageSize[0])
    def __percentToPoints(self,imageSize,point):
        return (int(point[0]*imageSize[1]),int(point[1]*imageSize[0]))
    def getPointsForCV2(self):
        return [self.__points[0],self.__points[1]]
    def drawRectangle(self,image):
        #ako su oba pointa postavljena
        if self.__points[0] !=(-1,-1) and self.__points[1] !=(-1,-1):
            p1=self.__percentToPoints(image.shape,(self.__points[0][0],self.__points[0][1]))
            p2=self.__percentToPoints(image.shape,(self.__points[1][0],self.__points[1][1]))
            cv2.rectangle(image,p1,p2,self.getColorForClasses(self.getClassName()),2)
        #ako je postavljen samo prvi point
        elif(self.__points[0] != (-1,-1)):
            p1=self.__percentToPoints(image.shape,(self.__points[0][0],self.__points[0][1]))
            cv2.circle(image,p1,3,self.getColorForClasses(self.getClassName()),-1)
    # ako je drugi point manji od prvog zamijeni ih da se moze nacrtati pravokutnik
    def __rearrange_points(self):
        x1, y1 = self.__points[0]
        x2, y2 = self.__points[1]

        top_left_x = min(x1, x2)
        top_left_y = min(y1, y2)
        bottom_right_x = max(x1, x2)
        bottom_right_y = max(y1, y2)

        new_p1 = [top_left_x, top_left_y]
        new_p2 = [bottom_right_x, bottom_right_y]

        self.__points[0] = new_p1
        self.__points[1] = new_p2
    @classmethod
    def getColorForClasses(cls,index):
        return cls.__colorsForClasses[index%len(cls.__colorsForClasses)]