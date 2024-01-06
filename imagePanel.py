import wx
import cv2
import os
from yolov8Detection import YOLOv8
class ImagePanel(wx.Panel):
   
    def __init__(self, parent,frameSize,position=wx.DefaultPosition):
        super(ImagePanel, self).__init__(parent,size=frameSize,pos=position)
        self._imagePath = None
        self._originalImage=None
        self._image=None
        self._imageLabel = None
        self._imageBitmap = None
        self._imageLabelSize = None
        self._points=None
        self._classNumber=0
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_KEY_DOWN,self.on_key_down)

    def on_paint(self, event):
        if self._imageBitmap is not None:
            dc = wx.BufferedPaintDC(self)
            dc.DrawBitmap(self._imageBitmap, 0, 0)
    def on_size(self, event):
        if self._originalImage is not None:
            self._image=cv2.resize(self._originalImage, self.GetSize(),cv2.INTER_AREA)
            height, width, _ = self._image.shape
            self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
            self.Refresh()
    def LoadImage(self, path):
        self._imagePath = path
        self.loadPointsFromJSON()
        self._originalImage=cv2.imread(path)
        self._originalImage=cv2.cvtColor(self._originalImage,cv2.COLOR_BGR2RGB)
        if self._originalImage is None:
            return False
        self._image=cv2.resize(self._originalImage, self.GetSize(),cv2.INTER_AREA)
        self._points.drawRectangles(self._image)
        height, width, _ = self._image.shape
        self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
        self.Refresh()
    def on_left_down(self, event):
        if self._image is not None:
            x = event.GetX()
            y = event.GetY()
            width,height=self.GetSize()
            if x < self._image.shape[1] and y < self._image.shape[0]:
                self._points.addPoint((height,width),(x,y),self._classNumber)
                self._image=cv2.resize(self._originalImage, self.GetSize(),cv2.INTER_AREA)
                self._points.drawRectangles(self._image)
                height, width, _ = self._image.shape
                self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
                self.Refresh()
        event.Skip()
    def on_right_down(self, event):
        if self._image is not None:
            x = event.GetX()
            y = event.GetY()
            width,height=self.GetSize()
            if x < self._image.shape[1] and y < self._image.shape[0]:
                if self._points.deleteRectangle((x,y),(height,width)):
                    self._image=cv2.resize(self._originalImage, self.GetSize(),cv2.INTER_AREA)
                    self._points.drawRectangles(self._image)
                    height, width, _ = self._image.shape
                    self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
                    self.Refresh()
        event.Skip()
    def drawImage(self):
        if self._image is None:
            return False
    def getAllClassesAsJSON(self):
        allPoints=self._points.getAllPoints()
        pointsInJSON=[]
        for point in allPoints:
            pointsInJSON.append({
                'className':point.getClassName(),
                'points':point.getPoints()
            })
        return pointsInJSON
    def loadPointsFromJSON(self):
        self._points=PointsHandling()
        filename=os.path.splitext(self._imagePath)[0]+'.txt'
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
                allPoints.append({
                    'className':int(className),
                    'points':[(x1,y1),(x2,y2)]
                })
            infile.close()
        self._points.JSONToPoints(allPoints)
    def loadModel(self,path):
        self._model=YOLOv8(path,0.4)
    def detectPointsWithModel(self):
        if self._model is None:
            return False
        boxes, scores, class_ids=self._model.detect_objects(self._originalImage)
        self._points=PointsHandling()
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1,y1,x2,y2=box
            self._points.addPoint(self._originalImage.shape,(x1,y1))
            self._points.addPoint(self._originalImage.shape,(x2,y2),class_id)
        self._image=cv2.resize(self._originalImage, self.GetSize(),cv2.INTER_AREA)
        self._points.drawRectangles(self._image)
        height, width, _ = self._image.shape
        self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
        self.Refresh()
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

#prihvaca 2 tocke za pravokutnik koje su gornja lijeva i donja desna tocka
#sprema ih kao postotak širine i visine slike
class pointsForClass():
    _colorsForClasses=[
            (73,200,12),
            (200,35,100),
            (0,218,100),
            (12,78,200),
            (23,100,250),
            (200,43,12),
            (100,10,140),
            (0,100,0),
            (230,40,122),
            (100,12,100),

        ]

    def __init__(self,className=0):
        self._className=className
        self._firstPoint=True
        self._lastPoint=False
        self._points={0:None,1:None}

    
    def addPoint(self,point,imgSize=None,classId=None):
        if self._lastPoint:
            return
        if imgSize is not None:
            point=self.pointsToPercent(imgSize,point)
        if classId is not None:
            self._className=int(classId)
        if self._firstPoint:
            self._points[0]=point
            self._firstPoint=False
        else:
            self._points[1]=point
            self._firstPoint=True
            self._rearrange_points()
            self._lastPoint=True
    
    def getPoints(self):
        return self._points
    def getClassName(self):
        return self._className
    def setClassName(self,className):
        self._className=className
    def pointsToPercent(self,imageSize,point):
        return (point[0]/imageSize[1],point[1]/imageSize[0])
    def percentToPoints(self,imageSize,point):
        return (int(point[0]*imageSize[1]),int(point[1]*imageSize[0]))
    def getPointsForCV2(self):
        return [self._points[0],self._points[1]]
    def drawRectangle(self,image):
        #ako su oba pointa postavljena
        if self._points[0] is not None and self._points[1] is not None:
            p1=self.percentToPoints(image.shape,(self._points[0][0],self._points[0][1]))
            p2=self.percentToPoints(image.shape,(self._points[1][0],self._points[1][1]))
            cv2.rectangle(image,p1,p2,self._colorsForClasses[self._className],2)
        #ako je postavljen samo prvi point
        elif(self._points[0] is not None):
            p1=self.percentToPoints(image.shape,(self._points[0][0],self._points[0][1]))
            cv2.circle(image,p1,3,self._colorsForClasses[self._className],-1)
    # ako je drugi point manji od prvog zamijeni ih da se moze nacrtati pravokutnik
    def _rearrange_points(self,):
        x1, y1 = self._points[0]
        x2, y2 = self._points[1]

        top_left_x = min(x1, x2)
        top_left_y = min(y1, y2)
        bottom_right_x = max(x1, x2)
        bottom_right_y = max(y1, y2)

        new_p1 = [top_left_x, top_left_y]
        new_p2 = [bottom_right_x, bottom_right_y]

        self._points[0] = new_p1
        self._points[1] = new_p2

class PointsHandling():
    def __init__(self):
        self.allPoints=[]
        self._point=pointsForClass()
        self._indexOfPoint=0
    #dodaje point u listu svih pointova
    def addPoint(self,imgSize,point,className=0):
        self._point.addPoint(point,imgSize,classId=className)
        if self._point._lastPoint==False:
            self.allPoints.append(self._point)
            self._indexOfPoint=len(self.allPoints)-1
        else:
            self.allPoints[self._indexOfPoint]=self._point
            self._point=pointsForClass()
    #nacrtaj sve pointove kao pravokutnike
    def drawRectangles(self,image):
        for point in self.allPoints:
            point.drawRectangle(image)
    def deleteRectangle(self,point,imgSize):
        X,Y=point
        X=X/imgSize[1]
        Y=Y/imgSize[0]

        for points in self.allPoints:
            x1,y1=points.getPoints()[0]
            x2,y2=points.getPoints()[1]
            if x1<=X<=x2 and y1<=Y<=y2:
                self.allPoints.remove(points)
                return True
        return False

    def getAllPoints(self):
        return self.allPoints
    def JSONToPoints(self,allPoints):
        for point in allPoints:
            self._point=pointsForClass(point['className'])
            self._point.addPoint(point['points'][0],classId=point['className'])
            self._point.addPoint(point['points'][1],classId=point['className'])
            self.allPoints.append(self._point)
        self._point=pointsForClass()