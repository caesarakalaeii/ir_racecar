import cv2
import numpy as np
import asyncio
import time





async def showCam(cam, new_frame_time,prev_frame_time):
    
    

    cv2.namedWindow("test")
    
    
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize the histogram to improve contrast
        frame = cv2.equalizeHist(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
        
        # Calculating the fps
        
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)
        
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        
        # putting the FPS count on the frame
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        
        cv2.imshow("test", frame)
        
        

        
        return frame
        

async def showDiff(frame, oldFrame):
    try:
        #diff = cv2.absdiff(frame, oldFrame)
        #mask = diff
        #imask =  mask>150
        #canvas = np.zeros_like(oldFrame, np.uint8)
        #canvas[imask] = oldFrame[imask]
        canvas = cv2.absdiff(frame, oldFrame)
        #canvas = np.multiply(canvas,10)
        cv2.imshow("diff", canvas)
        return frame
    except:
        return frame

async def showTransDiff(frame, oldFrame):
    try:
        #diff = cv2.absdiff(frame, oldFrame)
        #mask = diff
        #imask =  mask>150
        #canvas = np.zeros_like(oldFrame, np.uint8)
        #canvas[imask] = oldFrame[imask]
        canvas = cv2.absdiff(frame, oldFrame)
        cv2.imshow("diff", canvas)
        return frame
    except:
        return frame


async def showTrans(frame,oldDst):
        row, col = frame.shape[:2]
        #frame = cv2.fastNlMeansDenoising(frame, None, 7, 5, 9) # denoise gaussian noise
        #kernel = np.array([
        #    [-1, -1, -1],
        #    [-1, 9, -1],
        #    [-1, -1, -1]
        #    ]) #kernel to filter for edges
        #frame = cv2.filter2D(frame, -1, kernel) # filter for edges
        #bottom = frame[row-2:row, 0:col]

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
        thrs= 180
        dst[dst < thrs] = 0 # make image monochrome
        dst[dst >= thrs] = 255
        #dst = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,a,b)
        
        cv2.imshow("transformed",dst)
        showTransDiff(dst, oldDst)
        return dst

async def main():
    cam = cv2.VideoCapture(0)
    oldFrame = None
    oldDst= None
    prev_frame_time=0
    new_frame_time=0
    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        frame = await showCam(cam, new_frame_time, prev_frame_time)
        
        try:
            oldDst = await showTrans(frame,oldDst)
            oldFrame = await showDiff(frame, oldFrame)
        except:
            continue


if __name__ == "__main__":
    asyncio.run(main())

