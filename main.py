import wx
import os
import imagePanel as ip
import datasetCreation as dc
class MainFrame(wx.Frame):
    def __init__(self,parent,title='Labeling App'):
        super(MainFrame, self).__init__(parent, title=title)
        self.Maximize(True)
        self.config = wx.Config("LabelingApp")
        # Panels
        self._mainPanel = wx.Panel(self)
        self._controlPanel = wx.Panel(self._mainPanel)
        self._imagePanel = ip.ImagePanel(self._mainPanel)
        # Buttons
        self._loadDirectory=wx.Button(self._controlPanel, label='Load Directory')
        self._nextImage=wx.Button(self._controlPanel, label='Next Image')
        self._prevImage=wx.Button(self._controlPanel, label='Previous Image')
        self._detectObject=wx.Button(self._controlPanel, label='Detect Objects')    
        self._checkOnlyNotDetected=wx.CheckBox(self._controlPanel, label='Only not detected')
        self._createDataset=wx.Button(self._controlPanel, label='Create Dataset')
        self._listctrlImg=wx.ListCtrl(self._mainPanel,style=wx.LC_REPORT)
        self._listctrlClasses=wx.ListCtrl(self._mainPanel,style=wx.LC_REPORT)
        self._listctrlImg.InsertColumn(0,"Images")
        self._listctrlClasses.InsertColumn(0,"Classes")
       
        #style
        self.SetBackgroundColour('#2E2E2E')
        self._mainPanel.SetBackgroundColour('#2E2E2E')
        self._controlPanel.SetBackgroundColour('#2E2E2E')
        self._loadDirectory.SetBackgroundColour('#008080')
        self._loadDirectory.SetForegroundColour('#FFFFFF')
        self._nextImage.SetBackgroundColour('#00BFFF')
        self._nextImage.SetForegroundColour('#FFFFFF')
        self._prevImage.SetBackgroundColour('#FFA500')
        self._prevImage.SetForegroundColour('#FFFFFF')
        self._detectObject.SetBackgroundColour('#008080')
        self._detectObject.SetForegroundColour('#FFFFFF')
        self._checkOnlyNotDetected.SetForegroundColour('#FFFFFF')
        self._checkOnlyNotDetected.SetBackgroundColour('#2E2E2E')
        self._createDataset.SetBackgroundColour('#D3D3D3')
        self._createDataset.SetForegroundColour('#000000')

        #mainSizer
        self._mainSizer=wx.BoxSizer(wx.HORIZONTAL)
        self._mainSizer.Add(self._listctrlImg,10,wx.EXPAND)
        self._mainSizer.Add(self._listctrlClasses,5,wx.EXPAND)
        self._mainSizer.Add(self._imagePanel, 100, wx.EXPAND)
        self._mainSizer.Add(self._controlPanel,3,wx.EXPAND|wx.BOTTOM,50)
        self._mainPanel.SetSizerAndFit(self._mainSizer)
        #controlSizer
        self._controlSizer=wx.BoxSizer(wx.VERTICAL)
        self._controlSizer.Add(self._loadDirectory, 2, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._nextImage, 3, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._prevImage, 3, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._detectObject, 1, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._checkOnlyNotDetected, 1, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._createDataset, 2, wx.EXPAND|wx.TOP,10)
        self._controlPanel.SetSizerAndFit(self._controlSizer)

        # Bind events
        self._loadDirectory.Bind(wx.EVT_BUTTON, self._OnLoadDirectory)
        self._nextImage.Bind(wx.EVT_BUTTON, self._OnNextImage)
        self._prevImage.Bind(wx.EVT_BUTTON, self._OnPrevImage)
        self._detectObject.Bind(wx.EVT_BUTTON, self._DetectObjects)
        self._createDataset.Bind(wx.EVT_BUTTON, self._OnCreateDataset)
        self._listctrlClasses.Bind(wx.EVT_LIST_ITEM_SELECTED,self._On_Classname_selected)
        self._listctrlImg.Bind(wx.EVT_LIST_ITEM_SELECTED,self._On_Img_selected)
        #initialize variables
        self._selectedDirectory = None
        self._images = []
        self._currentImage = 0
        self._modelPath=None
        self.Show(True)
    
    def _OnLoadDirectory(self, event):
        #select directory
        last_path = self.config.Read("LastPathDir", "")
        dlg=wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE,defaultPath=last_path)
        
        if dlg.ShowModal() == wx.ID_OK:
            self._selectedDirectory = os.path.normpath(dlg.GetPath())
            self.config.Write("LastPathDir", self._selectedDirectory)
            self._LoadImages()
        dlg.Destroy()
        event.Skip()
    
    def _OnNextImage(self, event):
        #save file
        self._SaveFile()
        prevIndex=self._currentImage
        #go through images until one doesnt has a label file
        if(self._checkOnlyNotDetected.IsChecked()):
            while(self._currentImage<len(self._images)-1):
                self._currentImage+=1
                filename=os.path.splitext(self._images[self._currentImage])[0]+'.txt'
                if not os.path.exists(os.path.join(self._selectedDirectory, filename)):
                    self._LoadImage()
                    event.Skip()
                    return
            self._currentImage=prevIndex
            self._LoadImage()
            event.Skip()
            return
                
        if self._currentImage < len(self._images)-1:
            self._currentImage += 1
            self._LoadImage()
        event.Skip()
    def _OnPrevImage(self, event):
        self._SaveFile()
        prevIndex=self._currentImage

        #go through images until one doesnt has a label file
        if(self._checkOnlyNotDetected.IsChecked()):
                
                while(self._currentImage>0):
                    self._currentImage-=1
                    filename=os.path.splitext(self._images[self._currentImage])[0]+'.txt'
                    if not os.path.exists(os.path.join(self._selectedDirectory, filename)):
                        self._LoadImage()
                        event.Skip()
                        break
                self._currentImage=prevIndex
                self._LoadImage()
        elif self._currentImage > 0:
            self._currentImage -= 1
            self._LoadImage()
        event.Skip()
    def _LoadImage(self):
        self._imagePanel.LoadImage(os.path.join(self._selectedDirectory, self._images[self._currentImage]))
        self.SetTitle(self._images[self._currentImage]) 
    def _DetectObjects(self, event):
        if self._modelPath is None:
            last_path = self.config.Read("LastPathModel", "")
            dlg=wx.FileDialog(self, "Choose a model:", style=wx.DD_DEFAULT_STYLE,defaultDir=os.path.dirname(last_path))
            if dlg.ShowModal() == wx.ID_OK:
                self._modelPath = dlg.GetPath()
                self.config.Write("LastPathModel", self._modelPath)
                self._imagePanel.loadModel(self._modelPath)
            dlg.Destroy()
        self._imagePanel.detectPointsWithModel()
        event.Skip()
    def _OnCreateDataset(self, event):
        if(self._selectedDirectory==None):
            event.Skip()
            return
        dataset=dc.DatasetCreator()
        size_of_dataset=dataset.loadDataset(self._selectedDirectory)
        if size_of_dataset>10:
            dataset.createDataset(imgSize=(640,640),mozaicMix=20,split=(0.80,0.15,0.5),noisePercent=0.02,flip_horizontaly=True)
        event.Skip()
    def _On_Classname_selected(self,event):
        obj=event.GetEventObject()
        index=obj.GetFocusedItem()
        self._imagePanel.setClassname(index)
    def _On_Img_selected(self,event):
        self._SaveFile()
        obj=event.GetEventObject()
        index=obj.GetFocusedItem()
        self._currentImage=index
        self._LoadImage()
    # Load images from selected directory
    def _LoadImages(self):
        self._images = []
        all_files=os.listdir(self._selectedDirectory)
        self._images=[os.path.normpath(imgPath) for imgPath in all_files if imgPath.lower().endswith((".jpg",".png",".jpeg")) ]
        self._listctrlClasses.DeleteAllItems()
        self._listctrlImg.DeleteAllItems()
        self.LoadLabelsTXT()
        if len(self._images) > 0:
            #sort images by name so they are in order slika1,slika2...
            self._images = self.sort_strings(self._images)

            for i,img in enumerate(self._images):
                index=self._listctrlImg.InsertItem(i,img)
                if os.path.splitext(img)[0]+".txt" in all_files:
                    self._listctrlImg.SetItemBackgroundColour(index,wx.Colour("#547ab8"))
                    
        self._currentImage = 0
        self._LoadImage()
    def sort_strings(self,strings):
        import re
        def key_func(s):
            match = re.match(r"([a-z]+)([0-9]+)", s, re.I)
            if match:
                items = match.groups()
                return (items[0], int(items[1]))
            return (s,0)

        return sorted(strings, key=key_func)
    def _SaveFile(self):
        #className:class:int, points:[point1,point2]
        
        allLabels=self._imagePanel.getLabels()
        filename=os.path.splitext(self._images[self._currentImage])[0]+'.txt'
        filepath=os.path.join(self._selectedDirectory, filename)
        if not allLabels:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    self._listctrlImg.SetItemBackgroundColour(self._currentImage,wx.Colour(255,255,255))
        else:
            with open(filepath, 'w') as outfile:
                for className in allLabels:
                    if className['points'][0] and className['points'][1] is None:
                        continue
                    x1,y1=className['points'][0]
                    x2,y2=className['points'][1]
                    x=(x1+x2)/2
                    y=(y1+y2)/2
                    width=abs(x1-x2)
                    height=abs(y1-y2)
                    outfile.write(str(className["className"])+" "+f'{x:.5f}'+' '+f'{y:.5f}'+' '+f'{width:.5f}'+' '+f'{height:.5f}'+'\n')
                    self._listctrlImg.SetItemBackgroundColour(self._currentImage,wx.Colour("#547ab8"))
            outfile.close()
    def LoadLabelsTXT(self):
        if(os.path.exists(os.path.join(self._selectedDirectory,"labels.txt"))):
            lines=[]
            with open(os.path.join(self._selectedDirectory,"labels.txt")) as file:
                for line in file:
                    lines.append(line.split("\n")[0])
            for i,line in enumerate(lines):
                index=self._listctrlClasses.InsertItem(i,line)
                self._listctrlClasses.SetItemBackgroundColour(index,wx.Colour(ip.Label.getColorForClasses(index=index)))
                
if __name__ == '__main__':
    app=wx.App(False)
    GUI=MainFrame(None, title='Labeling App')
    app.MainLoop()
