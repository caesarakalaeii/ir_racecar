#!/usr/bin/env python3
"""
Source availabe on GitHub
This program is sensitive to changes in camera orientation, run it without the static matrix to find a good setup for your cameras.
Camera 1 needs to be the right camera!
"""

try:
    import image_join as ij
    import rospy
    from sensor_msgs.msg import Image
    import numpy as np
    import cv2
    import threading
    import traceback
    import time as t
    from parameters import default_list
except:
    raise ImportError("Imports failed")
class Logger():
    def __init__(self, ros_log = False, console_log = False):
        self.ros_log = ros_log
        self.console_log = console_log

    def warning(self, skk):
        
        if self.console_log:
            print("\033[93m {}\033[00m" .format("WARNING:"),"\033[93m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logwarn(skk)
       
    def error(self, skk):
        if self.console_log:   
            print("\033[91m {}\033[00m" .format("ERROR:"),"\033[91m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logerr(skk)
        
    def fail(self, skk):
        if self.console_log: 
            print("\033[91m {}\033[00m" .format("FATAL:"),"\033[91m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logfatal(skk)
    def passing(self, skk): 
        if self.console_log: 
            print("\033[92m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)
    def passingblue(self, skk): 
        if self.console_log: 
            print("\033[96m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)
    def info(self, skk): 
        if self.console_log: 
            print("\033[94m {}\033[00m" .format("Info:"),"\033[94m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)

class CameraJoin(object):
    def __init__(self,dict):
        try:
            self.l = Logger(ros_log=dict["ros_log"], console_log=dict["console_log"])
        except:
            Logger(True, True).self.l.fail("Failed to initialize Logger, exiting")
            exit(1)
        for k,v in dict.items():
            if k in default_list:
                continue
            else: self.l.warning("Not recognized key ", k, " and value ", v)
        for k,v in default_list.items():
            if k in dict:
                continue
            else: dict.update({k:v["default"]})
        self.times = [0,0]
        self.image1 = None
        self.image2 = None
        self.bridge = CvBridge()
        self.image_joined = None
        self.VERBOSE = dict["verbose"]
        self.ENCODING = dict["encoding"]
        self.timing = dict["timing"]
        self.stitcher = ij.ImageJoinFactory.create_instance(dict)
        self.pub = rospy.Publisher(dict["publish"], Image, queue_size= dict["queue_size"])
        if not dict["direct_import"]:
            rospy.Subscriber(dict["camera1"], Image, self.image1_callback)
            rospy.Subscriber(dict["camera2"], Image, self.image2_callback)
            self.thread = threading.Thread(target= self.ros_loop)
            self.thread.start()
        else:
            self.l.passing("initializing direct import")
            try:
                self.cam1 = cv2.VideoCapture(dict["direct_import_sources"][0])
                if not self.cam1.isOpened():
                    raise Exception("Unable to open Cam1")
                self.cam2 = cv2.VideoCapture(dict["direct_import_sources"][1])
                if not self.cam2.isOpened():
                    raise Exception("Unable to open Cam2")
                
            except Exception as e:
                self.l.fail("Couldn't initialize cameras, exiting! Are they used by another Process?")
                rospy.signal_shutdown("Couldn't initialize cameras, exiting! Are they used by another Process?")
                raise e
            self.thread = threading.Thread(target = self.direct_import_loop)
            self.thread.start()
            
        
        print("Node sucessfully initialized")

    
    def direct_import_loop(self):
        self.l.passing("Direct import started")
        while not rospy.is_shutdown():
            if self.timing:
                self.times[0] = t.time()
            ret1, self.image1 = self.cam1.read()
            if not ret1 == True: 
                self.l.fail("couldn't fetch frame from cam1")
                continue
            ret2, self.image2 = self.cam2.read()
            if not ret2 == True: 
                self.l.fail("couldn't fetch frame from cam2")
                continue
            if self.timing:
                print("Time between calls:", (self.times[0]-self.times[1])*1000, "ms")
                self.times[1] = self.times[0]
            self.image_join()

    def image1_callback(self,msg):
        if self.VERBOSE:
            str = "Image1 received with encoding: " + msg.encoding
            print(str)
        if self.timing:
            self.times[0] = t.time()
            print("Time between calls:", (self.times[0]-self.times[1])*1000, "ms")
            self.times[1] = self.times[0]
        self.set_image(msg, 1)


    def image2_callback(self,msg):
        if self.VERBOSE:
            str = "Image2 received with encoding: " + msg.encoding
            print(str)
        self.set_image(msg, 2)

    def loop(self):
        self.l.passing("Starting Loop...")

        rospy.spin()
    
    def ros_loop(self):
        while not rospy.is_shutdown():
            self.image_join()
    
    
    def set_image(self, image, number):
        try:
            if not isinstance(image, Image):
                self.l.fail("Input image is not of type sensor_msgs/Image")
            if self.timing:
                start_set = t.time()
        
            if number == 1:
                self.image1 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            elif number == 2:
                self.image2 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            if self.timing:
                end_set  =t.time()
                print("Time to set image:", (end_set-start_set)*1000, "ms")
            if not self.thread.is_alive():
                self.thread.start()
        except Exception as e:
            print(e)





    def image_join(self):
    # Join images
        if self.image1 is not None and self.image2 is not None:
            if self.VERBOSE:
                print("Joining images")
            try:
                if not isinstance(self.image1, np.ndarray):
                    if self.timing:
                        start_img2 = t.time()
                    self.image1 = np.asarray(self.image1)
                    if self.timing:
                        end_img2 = t.time()
                        print("Time to convert img1: ", (end_img2-start_img2)*1000, "ms")
                    if self.VERBOSE:
                        print("Converting Image 1 to ndarray")
                if not isinstance(self.image2, np.ndarray):
                    if self.timing:
                        start_img2 = t.time()
                    self.image2 = np.asarray(self.image2)
                    if self.timing:
                        end_img2 = t.time()
                        print("Time to convert img2: ", (end_img2-start_img2)*1000, "ms")
                    if self.VERBOSE:
                        print("Converting Image 2 to ndarray")
                if self.timing:
                    start_blending = t.time()
                old_image = self.image_joined
                try:
                    self.image_joined = self.stitcher.blending(self.image1, self.image2)
                except Exception as e:
                    if self.VERBOSE:
                        self.l.warning("Stale image will be used: {}".format(e))
                    self.image_joined = old_image
                if self.timing:
                    end_blending = t.time()
                    print("Time to blend:", (end_blending-start_blending)*1000, "ms")
                try:
                    self.publish_image(self.image_joined)
                except:
                    pass
                
            except Exception as e :
                print(e)
        elif self.VERBOSE and self.image1 is None and self.image2 is None:
            print("Both Images are None")
        elif self.VERBOSE and self.image1 is None:
            
            print("Image 1 is None")
        elif self.VERBOSE and self.image2 is None:
            
            print("Image 2 is None")
        elif self.VERBOSE:
            l.info("How did we get here?")

    
    def publish_image(self,image_joined):
        if self.timing:
            publish_start= t.time()
        image_joined_msg = self.bridge.cv2_to_imgmsg(image_joined, "bgr8")
        self.pub.publish(image_joined_msg)
        if self.timing:
            publish_end = t.time()
            print("Time to publish: ", (publish_end-publish_start)*1000, "ms")


        




if __name__ == '__main__':
    runtime_list = dict()
    runtime_list.update({"join_type": 2})
    runtime_list.update({"verbose":True})
    runtime_list.update({"direct_import": False})
    runtime_list.update({"static_matrix": True})
    runtime_list.update({"timing":False})
    runtime_list.update({"console_log":True})
    
    l = Logger(False, runtime_list["console_log"])
    value_list = dict() #used to store values from ros parameters
    for k,v in default_list:
        value_list.update({k:v["default"]})
    # ROS Image message -> OpenCV2 image converter
    from cv_bridge import CvBridge, CvBridgeError
    name = 'camera_join' #TODO find a way to make this variable
    rospy.init_node(name, anonymous=True, log_level=rospy.DEBUG)
    l.passing("Starting Node, Fetching params")
    simulate_params = False
    if simulate_params:
        l.warning("Simulating set Parameters, if not launched from a .launch file")
    print("Available Params:")
    param_names = rospy.get_param_names()
    for i in param_names:
        if "camera_join" in i:
            print(i)
            if not any(True for k in default_list if k in i):
                l.warning("Param {} was not recognized as a valid parameter and will be ignored!".format(i))
    
    if not any(True for i in param_names if i in default_list.keys()) and simulate_params:
            for k,v in default_list.items():
                try:
                    rospy.set_param(v["ros_param"], runtime_list[k])
                except KeyError:
                    rospy.set_param(v["ros_param"], v["default"])
    try:
        for k,v in default_list.items():
            if rospy.has_param(v["ros_param"]): #not using the built in default values of rospy for better verbosity
                value_list.update({k: rospy.get_param(v["ros_param"])})
                l.passing("Parameter {} has been found and added".format(v["ros_param"].replace('~', '')) )
                
            elif k in runtime_list:
                value_list.update({k: runtime_list[k]})
                l.passingblue("Parameter {} has been found in Runtime list and will be used".format(v["ros_param"].replace('~', '')))
                
            
            else: 
                value_list.update({k: default_list[k]})
                l.warning("Parameter {} has not been found, using default".format(v["ros_param"].replace('~', '')) )
               

        my_subs = CameraJoin(value_list)
        l.passing("Node started with given Params")
        my_subs.loop()
        
    except KeyError:
        l.warning("Fetching Params failed, using default params")
        my_subs = CameraJoin(value_list)
        my_subs.loop()
    except:
        l.fail("Couldn't start Node, is Cam1 the right one?")
        traceback.print_exc()