import cv2
import image_join as join
import traceback



if __name__ == "__main__":
    
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    joiner = join.ImageJoinFactory.create_instance(joinType=1)
    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
        try:
            ret, frame1 = cam1.read()
            ret, frame2 = cam2.read()
            cv2.imshow("Joined", joiner.blending(frame1, frame2))
        except:
            
            traceback.print_exc()
            break
