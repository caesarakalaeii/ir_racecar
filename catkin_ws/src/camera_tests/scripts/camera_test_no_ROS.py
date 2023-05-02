import cv2
import image_join as join
import traceback



if __name__ == "__main__":
    
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    joiner = join.ImageJoinFactory.create_instance(joinType=2, min_match=20, ratio= 0.85, smoothing_window_size=200, static_matrix=True)
    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
        try:
            ret, frame1 = cam1.read()
            #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            ret, frame2 = cam2.read()
            #frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Cam1", frame1)
            cv2.imshow("Cam2", frame2)
            cv2.imshow("Joined", joiner.blending(frame1, frame2))
        except:
            
            traceback.print_exc()
            break
