import cv2
import image_join as join
import traceback
from parameters import default_list



if __name__ == "__main__":
    runtime_list = dict()
    runtime_list.update({"join_type": 4})
    runtime_list.update({"verbose":True})
    runtime_list.update({"direct_import": False})
    runtime_list.update({"static_matrix": False})
    runtime_list.update({"timing":False})
    runtime_list.update({"console_log":True})
    for k,v in default_list.items():
        if not (k in runtime_list):
            runtime_list.update({k:v["default"]})
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)
    joiner = join.ImageJoinFactory.create_instance(runtime_list)
    
    while True:
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
        try:
            
            a, frame1 = cam1.read()
            #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #uncomment for b/w images
            #frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            b, frame2 = cam2.read()
            if a == False or b == False:
                raise Exception("Couldnt open Cameras")
            
            cv2.imshow("Cam1", frame1)
            cv2.imshow("Cam2", frame2)
            try:
                cv2.imshow("Joined", joiner.blending(frame1, frame2))
            except Exception as e:
                
                print("Joining failed: ", e)
                continue
        except:
            
            traceback.print_exc()
            break
