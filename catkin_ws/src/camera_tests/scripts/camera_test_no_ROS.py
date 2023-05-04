import cv2
import image_join as join
import traceback
import parameters as p



if __name__ == "__main__":
    runtime_list = dict()
    runtime_list.update({"join_type": 2})
    runtime_list.update({"verbose":True})
    runtime_list.update({"direct_import": False})
    runtime_list.update({"static_matrix": True})
    runtime_list.update({"timing":False})
    runtime_list.update({"console_log":True})
    for k,v in p.default_list.items():
        if not k in runtime_list.keys():
            runtime_list.update({k:v})
    cam1 = cv2.VideoCapture(0)
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
            #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            b, frame2 = cam2.read()
            if a == False or b == False:
                raise Exception("Couldnt open Cameras")
            #frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Cam1", frame1)
            cv2.imshow("Cam2", frame2)
            cv2.imshow("Joined", joiner.blending(frame1, frame2))
        except:
            
            traceback.print_exc()
            break
