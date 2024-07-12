import wx
import cv2
import os
from yolov8Detection import YOLOv8
import numpy as np
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
        self._className="0"
        self.SetDoubleBuffered(True)
        # self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour("#2E2E2E")
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_KEY_DOWN,self.on_key_down)
    def on_paint(self, event):
        if self._imageBitmap is not None:
            dc = wx.BufferedPaintDC(self)
            s_width, s_height = self.GetSize()
            b_width, b_height = self._imageBitmap.GetSize()
            
            
            # Centriraj sliku
            x = (s_width - b_width) // 2
            y = (s_height - b_height) // 2
            dc.Clear()
            dc.DrawBitmap(self._imageBitmap, x, y,True)
    def showFrame(self):
        if self._originalImage is not None:
            new_width,new_height=self.get_size_keep_aspect_ratio()    
            self._image = cv2.resize(self._originalImage, (new_width, new_height))
            self._points.drawRectangles(self._image)
            height, width, _ = self._image.shape
            self._imageBitmap = wx.Bitmap.FromBuffer(width, height, self._image)
            self.Refresh()
    def get_size_keep_aspect_ratio(self):
        if self._originalImage is not None:
            s_width, s_height = self.GetSize()
            i_height, i_width, _ = self._originalImage.shape
        
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
        h,w,_=self._image.shape
        W,H = self.GetSize()
        x,y=pt
        # Calculate offset of image within the given size
        if w * H < W * h:  # fit on height
            s = w
            S = w * H // h
            o = W - S
            o //= 2
            x -= o
        else:  # fit on width
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
    def LoadImage(self, path):
        self._imagePath = path
        self.loadPointsFromLabel()
        
        # Read the image file as a binary stream
        with open(self._imagePath, 'rb') as f:
            image_bytes = f.read()
            
            # Convert the binary stream to a numpy array
            image_array = np.frombuffer(image_bytes, np.uint8)
            
            # Decode the image array
            self._originalImage = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        self._originalImage=cv2.cvtColor(self._originalImage,cv2.COLOR_BGR2RGB)
        if self._originalImage is None:
            return False
        self.showFrame()   
    def on_left_down(self, event):
        if self._image is not None:
            x = event.GetX()
            y = event.GetY()
            height,width,_=self._image.shape
            x,y=self.convert_pos((x,y))
            if x >=0 and x<=width and y >= 0 and y<=height:
                self._points.addPoint((height,width),(x,y),self._classNumber)
                self.showFrame()

        event.Skip()
    def on_right_down(self, event):
        if self._image is not None:
            x = event.GetX()
            y = event.GetY()
            height,width,_=self._image.shape
            x,y=self.convert_pos((x,y))
            if x >=0 and x<=width and y >= 0 and y<=height:
                if self._points.deleteRectangle((x,y),(height,width)):
                    self.showFrame()

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
    def loadPointsFromLabel(self):
        self._points=AllPointsHandler()
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
                allPoints.append([className,x1,y1,x2,y2])
            infile.close()
        self._points.LabelsToPoints(allPoints)
    def loadModel(self,path):
        self._model=YOLOv8(path)
    def detectPointsWithModel(self):
        if self._model is None:
            return False
        boxes, scores, class_ids=self._model.detect_objects(self._originalImage)
        self._points=AllPointsHandler()
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1,y1,x2,y2=box
            self._points.addPoint(self._originalImage.shape,(x1,y1))
            self._points.addPoint(self._originalImage.shape,(x2,y2),class_id)
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
    def __init__(self):
        self.allPoints=[]
        self._point=Label()
        self._indexOfPoint=0
    #dodaje point u listu svih pointova
    def addPoint(self,imgSize,point,className=0):
        self._point.addPoint(point,imgSize,classId=className)
        if self._point._lastPoint==False:
            self.allPoints.append(self._point)
            self._indexOfPoint=len(self.allPoints)-1
        else:
            self.allPoints[self._indexOfPoint]=self._point
            self._point=Label()
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
    def LabelsToPoints(self,allPoints):
        for point in allPoints:
            self._point=Label()
            self._point.addPoint((point[1],point[2]),classId=point[0])
            self._point.addPoint((point[3],point[4]),classId=point[0])
            self.allPoints.append(self._point)
        self._point=Label()

#prihvaca 2 tocke za pravokutnik koje su gornja lijeva i donja desna tocka
#sprema ih kao postotak širine i visine slike
class Label():
    _colorsForClasses = [
    (31, 119, 180),   # Blue
    (255, 127, 14),   # Orange
    (44, 160, 44),    # Green
    (214, 39, 40),    # Red
    (148, 103, 189),  # Purple
    (140, 86, 75),    # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),   # Olive
    (23, 190, 207),   # Cyan
    (255, 187, 120),  # Light Orange
    (199, 199, 199),  # Light Gray
    (158, 218, 229),  # Light Cyan
    (197, 176, 213),  # Light Purple
    (140, 140, 140),  # Dark Gray
    (196, 156, 148),  # Light Brown
    (227, 119, 180),  # Light Pink
    (174, 199, 232),  # Light Blue
    (152, 223, 138),  # Light Green
    (246, 207, 113)   # Light Yellow
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
            cv2.rectangle(image,p1,p2,self._colorsForClasses[self._className%20],2)
        #ako je postavljen samo prvi point
        elif(self._points[0] is not None):
            p1=self.percentToPoints(image.shape,(self._points[0][0],self._points[0][1]))
            cv2.circle(image,p1,3,self._colorsForClasses[self._className%20],-1)
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
