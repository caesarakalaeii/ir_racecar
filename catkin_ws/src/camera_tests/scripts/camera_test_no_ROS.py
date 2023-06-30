'''
Script to test the other ones without having a ros installation
Change values of runtime_list to change desired behaviour
'''
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
    runtime_list.update({"timing":False})
    runtime_list.update({"console_log":True})
    runtime_list.update({"finder": cv2.ORB_create()})
    show_windows = True
    for k,v in default_list.items():
        if not (k in runtime_list):
            runtime_list.update({k:v["default"]})
    l.info("Initializing Cam1")
    cam1 = cv2.VideoCapture(1)
    l.info("Initializing Cam2")
    cam2 = cv2.VideoCapture(2)
    l.info("Cameras initialized, Creating Joiner")
    joiner = ImageJoinFactory.create_instance(runtime_list)
    tries = 0
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            l.passing("Escape hit, closing...")
            break
        
        try:
            
            a, frame1 = cam1.read()
            b, frame2 = cam2.read()
            #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #uncomment for b/w images
            #frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if a == False or b == False:
                raise Exception("Couldnt open Cameras")
            
            if show_windows:
                cv2.imshow("Cam1", frame1)
                cv2.imshow("Cam2", frame2)
            if tries >100:
                try:
                    joined = joiner.blending(frame1, frame2)
                    if show_windows:
                        cv2.imshow("Joined", joined)
                except Exception as e:
                    
                    l.error("Joining failed: %s"%e)
                    continue
            elif tries == 0:
                print("Waiting", end= "") #TODO replace to actual waiting animation
            elif tries == 100:
                print("")
            else: 
                print(".", end = "")
            tries +=1
        except:
            
            traceback.print_exc()
            break
