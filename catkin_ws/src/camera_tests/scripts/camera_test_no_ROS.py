'''
Script to test the other ones without having a ros installation
Change values of runtime_list to change desired behaviour
'''
import time
import cv2
from image_join_factory import ImageJoinFactory
import traceback
from parameters import default_list
from logger import Logger


if __name__ == "__main__":
    l= Logger(False, True, file_logging=True)
    runtime_list = dict()
    runtime_list.update({"logger": l})
    runtime_list.update({"join_type": 2})
    runtime_list.update({"verbose":True})
    runtime_list.update({"direct_import": False})
    runtime_list.update({"static_matrix": False})
    runtime_list.update({"timing":True})
    runtime_list.update({"console_log":True})
    runtime_list.update({"finder": cv2.ORB_create(nfeatures=50)})
    runtime_list.update({"matcher": cv2.BFMatcher(cv2.NORM_HAMMING)})
    show_windows = True
    for k,v in default_list.items():
        if not (k in runtime_list):
            runtime_list.update({k:v["default"]})
    l.info("Initializing Cam1")
    cam1 = cv2.VideoCapture(0)
    l.info("Initializing Cam2")
    cam2 = cv2.VideoCapture(2)
    l.info("Cameras initialized, Creating Joiner")
    joiner = ImageJoinFactory.create_instance(runtime_list)
    tries = 0
    new_frame_time = 0
    prev_frame_time = 0
    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            l.passing("Escape hit, closing...")
            break
        
        try:
            
            a, frame1 = cam1.read()
            b, frame2 = cam2.read()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #uncomment for b/w images
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if a == False or b == False:
                print(f"Couldnt open Cameras, a:{a} b:{b}")
                continue
            
            if show_windows:
                cv2.imshow("Cam1", frame1)
                cv2.imshow("Cam2", frame2)
            if tries >100:
                try:
                    joined = joiner.blending(frame1, frame2)
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
                    cv2.putText(joined, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                    if show_windows:
                        cv2.imshow("Joined", joined)
                    continue
                except Exception as e:
                    
                    l.error("Joining failed: %s"%e)
                    continue
            
            tries +=1
        except:
            
            traceback.print_exc()
            break
