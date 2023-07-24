import cv2
import numpy as np





def showCam(cam):
    

    cv2.namedWindow("test")
    
    
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize the histogram to improve contrast
        #frame = cv2.equalizeHist(frame)
        cv2.imshow("test", frame)
        return frame
        
def makeDiff(frame, oldFrame):
    return cv2.absdiff(frame, oldFrame)

def showDiff(frame, oldFrame):
    if (oldFrame[0,0] == None):
        return frame
    #diff = cv2.absdiff(frame, oldFrame)
    #mask = diff
    #imask =  mask>150
    #canvas = np.zeros_like(oldFrame, np.uint8)
    #canvas[imask] = oldFrame[imask]
    #canvas = np.multiply(canvas,10)
    cv2.imshow("diff", makeDiff(frame, oldFrame))
    return frame
    

def showTransDiff(frame, oldFrame):
    if (oldFrame[0,0] == None):
        return frame
    #diff = cv2.absdiff(frame, oldFrame)
    #mask = diff
    #imask =  mask>150
    #canvas = np.zeros_like(oldFrame, np.uint8)
    #canvas[imask] = oldFrame[imask]
    cv2.imshow("TransDiff", makeDiff(frame, oldFrame))
    return frame
    

def makeCanny(frame):
    return cv2.Canny(frame, 50, 150)

def showCanny(frame):
    canny = makeCanny(frame)
    cv2.imshow("Canny",canny )
    return canny

def makeTrans(frame):
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

def makeThresh(dst, thrs):
        dst[dst < thrs] = 0 # make image monochrome
        dst[dst >= thrs] = 255
       
        return dst

def showOtsuThreshDiff(frame, oldFrame, name):
    if (oldFrame[0,0] == None):
            return
    otsu_threshold, image_result = cv2.threshold(
        makeDiff(frame, oldFrame), 0, 255, cv2.THRESH_OTSU,
    )
    print(f"Otsu threshhold is {otsu_threshold}")
    cv2.imshow(name, image_result)

def showThresh(frame, name, thrs):
        cv2.imshow(name,makeThresh(frame, thrs))
def showThreshDiff(frame,oldFrame, name, thrs):
        if (oldFrame[0,0] == None):
            return
        diff = cv2.absdiff(frame, oldFrame)
        cv2.imshow(name,makeThresh(diff, thrs))

def showTrans(frame):
        row, col = frame.shape[:2]
        #frame = cv2.fastNlMeansDenoising(frame, None, 7, 5, 9) # denoise gaussian noise
        #kernel = np.array([
        #    [-1, -1, -1],
        #    [-1, 9, -1],
        #    [-1, -1, -1]
        #    ]) #kernel to filter for edges
        #frame = cv2.filter2D(frame, -1, kernel) # filter for edges
        #bottom = frame[row-2:row, 0:col]

        dst = makeTrans(frame)
        thrs= 180
        dst[dst < thrs] = 0 # make image monochrome
        dst[dst >= thrs] = 255
        #dst = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,a,b)
        
        cv2.imshow("transformed",dst)
        return dst

def showCannyTrans(frame):
    dst = makeTrans(frame)
    cv2.imshow("CannyTrans", dst)
    return dst


def main():


    
    
    cam = cv2.VideoCapture(1)
    oldFrame = np.array([[None, None],[None, None]])
    oldDst = np.array([[None, None],[None, None]])
    oldtransDiff = np.array([[None, None],[None, None]])
    threshhold = 100
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k%256 == 119:  
            threshhold += 1
            print(f"Threshhold is {threshhold}")
        if k%256 == 115:    
            threshhold -= 1
            print(f"Threshhold is {threshhold}")
            
        frame = showCam(cam)
        try:
            #oldDst = showTrans(frame)
            showThreshDiff(frame, oldFrame, "threshholdDiff", threshhold)
            oldFrame = showDiff(frame, oldFrame)
            showOtsuThreshDiff(frame, oldFrame,"otsudiff")
            #oldtransDiff = showTransDiff(oldDst, oldtransDiff)
            #canny = showCanny(frame)
            #cannytrans= showCannyTrans(canny)
            
        except:
            continue


if __name__ == "__main__":
    
    main()

