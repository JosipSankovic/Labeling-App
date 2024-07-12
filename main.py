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
        self._imagePanel = ip.ImagePanel(self._mainPanel,(self.GetSize().GetWidth(),self.GetSize().GetHeight()))
        # Buttons
        self._loadDirectory=wx.Button(self._controlPanel, label='Load Directory')
        self._nextImage=wx.Button(self._controlPanel, label='Next Image')
        self._prevImage=wx.Button(self._controlPanel, label='Previous Image')
        self._detectObject=wx.Button(self._controlPanel, label='Detect Objects')    
        self._checkOnlyNotDetected=wx.CheckBox(self._controlPanel, label='Only not detected')
        self._createDataset=wx.Button(self._controlPanel, label='Create Dataset')
       
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
        self._mainSizer.Add(self._imagePanel, 8, wx.EXPAND)
        self._mainSizer.Add(self._controlPanel)
        self._mainPanel.SetSizerAndFit(self._mainSizer)
        #controlSizer
        self._controlSizer=wx.BoxSizer(wx.VERTICAL)
        self._controlSizer.Add(self._loadDirectory, 2, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._nextImage, 3, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._prevImage, 3, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._detectObject, 1, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._checkOnlyNotDetected, 1, wx.EXPAND|wx.TOP, 10)
        self._controlSizer.Add(self._createDataset, 1, wx.EXPAND|wx.BOTTOM, 300)
        self._controlPanel.SetSizerAndFit(self._controlSizer)

        # Bind events
        self._loadDirectory.Bind(wx.EVT_BUTTON, self._OnLoadDirectory)
        self._nextImage.Bind(wx.EVT_BUTTON, self._OnNextImage)
        self._prevImage.Bind(wx.EVT_BUTTON, self._OnPrevImage)
        self._detectObject.Bind(wx.EVT_BUTTON, self.detectObjects)
        self._createDataset.Bind(wx.EVT_BUTTON, self._OnCreateDataset)
        #initialize variables
        self._selectedDirectory = None
        self._images = []
        self._currentImage = 0
        self._modelPath=""
        self.Show(True)
    
    def _OnLoadDirectory(self, event):
        #select directory
        last_path = self.config.Read("LastPathDir", "")
        dlg=wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE,defaultPath=last_path)
        
        if dlg.ShowModal() == wx.ID_OK:
            self._selectedDirectory = dlg.GetPath()
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
                        return
                self._currentImage=prevIndex
                self._LoadImage()
                event.Skip()
                return
                
                
        if self._currentImage > 0:
            self._currentImage -= 1
            self._LoadImage()
        event.Skip()
    def _LoadImage(self):
        self._imagePanel.LoadImage(os.path.join(self._selectedDirectory, self._images[self._currentImage]))
        self.SetTitle(self._images[self._currentImage]) 
    def detectObjects(self, event):
        if self._modelPath!="":
            self._imagePanel.detectPointsWithModel()
            event.Skip()
            return
        last_path = self.config.Read("LastPathModel", "")
        dlg=wx.FileDialog(self, "Choose a model:", style=wx.DD_DEFAULT_STYLE,defaultDir=os.path.dirname(last_path))
        if dlg.ShowModal() == wx.ID_OK:
            self._modelPath = dlg.GetPath()
            self.config.Write("LastPathModel", self._modelPath)
        dlg.Destroy()
        self._imagePanel.loadModel(self._modelPath)
        self._imagePanel.detectPointsWithModel()
        event.Skip()
    def _OnCreateDataset(self, event):
        if(self._selectedDirectory==None):
            event.Skip()
            return
        dataset=dc.DatasetCreator()
        dataset.loadDirectory(self._selectedDirectory)
        dataset.createDataset((640,640),0)

        event.Skip()
    # Load images from selected directory
    def _LoadImages(self):
        self._images = []
        all_files=os.listdir(self._selectedDirectory)
        self._images=[imgPath for imgPath in all_files if imgPath.lower().endswith((".jpg",".png",".jpeg")) ]
        if len(self._images) > 0:
            #sort images by name so they are in order slika1,slika2...
            self._images = self.sort_strings(self._images)
        self._currentImage = 0
        self._LoadImage()
    def sort_strings(self,strings):
        import re
        def key_func(s):
            match = re.match(r"([a-z]+)([0-9]+)", s, re.I)
            if match:
                items = match.groups()
                return (items[0], int(items[1]))
            return s

        return sorted(strings, key=key_func)
    def _SaveFile(self):
        #className:class:int, points:[point1,point2]
        allClasses=self._imagePanel.getAllClassesAsJSON()
        
        filename=os.path.splitext(self._images[self._currentImage])[0]+'.txt'
        with open(os.path.join(self._selectedDirectory, filename), 'w') as outfile:
            if(len(allClasses)==0):
                outfile.close()
                os.remove(os.path.join(self._selectedDirectory, filename))
                return False
            for className in allClasses:
                x1,y1=className['points'][0]
                x2,y2=className['points'][1]
                x=(x1+x2)/2
                y=(y1+y2)/2
                width=abs(x1-x2)
                height=abs(y1-y2)
                outfile.write(str(className["className"])+" "+f'{x:.5f}'+' '+f'{y:.5f}'+' '+f'{width:.5f}'+' '+f'{height:.5f}'+'\n')
            outfile.close()
            return True
if __name__ == '__main__':
    app=wx.App(False)
    GUI=MainFrame(None, title='Labeling App')
    app.MainLoop()
