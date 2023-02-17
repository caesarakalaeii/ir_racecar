import cv2
import numpy as np
import asyncio
from tkinter import *
from tkinter import ttk
from enum import Enum





class ShowCam:
    CAM = 0
    TRANS = 1
    DIFF = 2
    DIFFTRANS = 3
    CANNY = 4
    CANNYTRANS = 5
    

    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.oldFrame = np.array([[None, None],[None, None]])
        self.oldDst = np.array([[None, None],[None, None]])
        self.oldtransDiff = np.array([[None, None],[None, None]])
        self.camOn = False
        self.transOn = False
        self.diffOn = False
        self.diffTransOn = False
        self.cannyOn = False
        self.cannyTransOn = False




    def enable(self, type):
        if type == ShowCam.CAM:
            self.camOn = not self.camOn
        elif type == ShowCam.TRANS:
            self.transOn = not self.transOn
        elif type == ShowCam.DIFF:
            self.diffOn = not self.diffOn
        elif type == ShowCam.DIFFTRANS:
            self.diffTransOn = not self.diffTransOn
        elif type == ShowCam.CANNY:
            self.cannyOn = not self.cannyOn
        elif type == ShowCam.CANNYTRANS:
            self.cannyTransOn = not self.cannyTransOn




    async def showCam(self,cam):
        frame = self.makeShow(cam)
        if self.camOn:
            cv2.imshow("test", frame)
        return frame

    def makeShow(self,cam):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Equalize the histogram to improve contrast
            frame = cv2.equalizeHist(frame)
        return frame
            
    def makeDiff(self,frame, oldFrame):
        return cv2.absdiff(frame, oldFrame)

    async def showDiff(self,frame, oldFrame):
    
        if (oldFrame[0,0] == None):
            return frame
            #diff = cv2.absdiff(frame, oldFrame)
            #mask = diff
            #imask =  mask>150
            #canvas = np.zeros_like(oldFrame, np.uint8)
            #canvas[imask] = oldFrame[imask]
            #canvas = np.multiply(canvas,10)
        if self.diffOn:
            cv2.imshow("diff", self.makeDiff(self,frame, oldFrame))
        return frame
        

    async def showTransDiff(self,frame, oldFrame):
        if (oldFrame[0,0] == None):
            return frame
        #diff = cv2.absdiff(frame, oldFrame)
        #mask = diff
        #imask =  mask>150
        #canvas = np.zeros_like(oldFrame, np.uint8)
        #canvas[imask] = oldFrame[imask]
        if self.diffTransOn:
            cv2.imshow("TransDiff", self.makeDiff(self,frame, oldFrame))
        return frame
        

    def makeCanny(self,frame):
        return cv2.Canny(frame, 50, 150)

    async def showCanny(self,frame):
        canny = self.makeCanny(self,frame)
        if self.showCanny:
            cv2.imshow("Canny",canny )
        return canny

    def makeTrans(self,frame):
        bordertop_bottom = 200
        borderleft_right = 240
        #extend image size
        border = cv2.copyMakeBorder(
        frame,
        top=bordertop_bottom,
        bottom=bordertop_bottom,
        left=borderleft_right,
        right=borderleft_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
        )

        pts1 = np.float32([[479,485],[648,479],[305,555],[846,527]]) # matrix to specify lens parameters
        pts2 = np.float32([[305,100],[846,100],[305,555],[846,555]])

        M = cv2.getPerspectiveTransform(pts1,pts2) # get transformation matrix

        

        dst = cv2.warpPerspective(border,M,(1080,700)) # warp image to adjust for lens distortion
        return dst

    async def showTrans(self,frame):
        
        row, col = frame.shape[:2]
        #frame = cv2.fastNlMeansDenoising(frame, None, 7, 5, 9) # denoise gaussian noise
        #kernel = np.array([
        #    [-1, -1, -1],
        #    [-1, 9, -1],
        #    [-1, -1, -1]
        #    ]) #kernel to filter for edges
        #frame = cv2.filter2D(frame, -1, kernel) # filter for edges
        #bottom = frame[row-2:row, 0:col]

        dst = self.makeTrans(self,frame)
        thrs= 180
        dst[dst < thrs] = 0 # make image monochrome
        dst[dst >= thrs] = 255
        #dst = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,a,b)
        if self.transOn:
            cv2.imshow("transformed",dst)
        return dst


    async def showCannyTrans(self,frame):
        
        dst = self.makeTrans(self,frame)
        if self.cannyTransOn:
            cv2.imshow("CannyTrans", dst)
        return dst

async def selectionGUI(show):
        

        # Erstelle das Tkinter-Fenster
        root = Tk()


        

        # Pack die GUI-Elemente in das Fenster
        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Erstelle die GUI-Elemente
        Button(root, text="Show Feed", command=show.enable(show.CAM)).grid(column=1, row=1, sticky=W)
        Button(root, text="Show Transformed", command=show.enable(show.TRANS)).grid(column=2, row=1, sticky=W)
        Button(root, text="Show Difference", command=show.enable(show.DIFF)).grid(column=3, row=1, sticky=W)
        Button(root, text="Show Difference Transformed", command=show.enable(show.DIFFTRANS)).grid(column=1, row=2, sticky=W)
        Button(root,  text="Show Transformed Canny", command=show.enable(show.CANNYTRANS)).grid(column=2, row=2, sticky=W)
        Button(root,  text="Show Transformed Canny", command=show.enable(show.CANNY)).grid(column=3, row=2, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)
        root.bind("<Return>", main(show))
        root.mainloop()


async def main(show):

    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        frame = await show.showCam(show.cam)
        try:
            oldDst = await show.showTrans(frame)
            oldFrame = await show.showDiff(frame, oldFrame)
            oldtransDiff = await show.showTransDiff(oldDst, oldtransDiff)
            canny = await show.showCanny(frame)
            cannytrans= await show.showCannyTrans(canny)
            
        except:
            continue


if __name__ == "__main__":
    show = ShowCam()
    asyncio.run(selectionGUI(show))

