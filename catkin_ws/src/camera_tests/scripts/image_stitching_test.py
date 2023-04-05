import camera_hconcat as pyfile
import cv2







if __name__ == "__main__":
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    while True:
        ret1, img1 = cam1.read()
        ret2, img2 = cam2.read()
        if not ret1 or not ret2:
            print("Something's wrong")
            break
        final = pyfile.Image_Stitching().blending(img1,img2)
        cv2.imshow('result',final)
        cv2.imshow("cam1", img2)
        k  = cv2.waitKey(1)
        # if the escape key is been pressed, the app will stop
        if k%256 == 27:
            print('escape hit, closing the app')
            break
    cv2.destroyAllWindows()