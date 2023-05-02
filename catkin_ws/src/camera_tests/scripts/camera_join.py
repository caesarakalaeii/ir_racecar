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
    import warnings
    import threading
except:
    raise ImportError("Imports failed")



class CameraJoin(object):
    param_list = {
        "camera1": "/camera_join/camera1",
        "camera2": "/camera_join/camera2",
        "publish": "/camera_join/publish",
        "queue_size": "/camera_join/queue_size",
        "verbose": "/camera_join/verbose",
        "encoding": "/camera_join/camera1",
        "joinType": "/camera_join/type",
        "left_y_offset": "/camera_join/left_y_offset",
        "right_y_offset": "/camera_join/right_y_offset",
        "left_x_offset": "/camera_join/left_x_offset",
        "right_x_offset": "/camera_join/right_x_offset",
        "ratio": "/camera_join/ratio",
        "min_match": "/camera_join/min_match",
        "smoothing_window_size": "/camera_join/smoothing_window_size",
        "matching_write": "/camera_join/matching_write",
        "static_matrix": "/camera_join/static_matrix",
        "static_mask": "/camera_join/static_mask",
        "stitchter_type": "/camera_join/stitchter_type"
    }

    default_list = {
        "camera1": "joined_cams/usb_cam1/image_raw",
        "camera2": "joined_cams/usb_cam2/image_raw",
        "publish": "joined_image/image_raw",
        "queue_size": 10,
        "encoding": 'bgr8',
        "verbose": False,
        "joinType": 1,
        "left_y_offset": 20,
        "right_y_offset": 0,
        "left_x_offset": 0,
        "right_x_offset": 0,
        "ratio": 0.85,
        "min_match": 10,
        "smoothing_window_size": 50,
        "matching_write": False,
        "static_matrix": False,
        "static_mask": False,
        "stitchter_type": cv2.Stitcher_PANORAMA,
        "direct_import": False,
        "direct_import_sources": (0,2)
    }
    
    
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            if k in CameraJoin.default_list:
                continue
            else: print("Not recognized key ", k, " and value ", v)
        for k,v in CameraJoin.default_list.items():
            if k in kwargs:
                continue
            else: kwargs.update(k=v)
        
        self.image1 = None
        self.image2 = None
        self.bridge = CvBridge()
        self.VERBOSE = kwargs["verbose"]
        self.ENCODING = kwargs["encoding"]
        self.stitcher = ij.ImageJoinFactory.create_instance(kwargs)
        self.pub = rospy.Publisher(kwargs["publish"], Image, queue_size= kwargs["queue_size"])
        if not kwargs["direct_import"]:
            rospy.Subscriber(kwargs["camera1"], Image, self.image1_callback)
            rospy.Subscriber(kwargs["camera2"], Image, self.image2_callback)
        else:
            print("initializing direct import")
            self.cam1 = cv2.VideoCapture(kwargs["direct_import_sources"][0])
            self.cam2 = cv2.VideoCapture(kwargs["direct_import_sources"][1])
            
            self.thread = threading.Thread(target = self.direct_import_loop)
            self.thread.start()
            
        
        print("Node sucessfully initialized")

    
    def direct_import_loop(self):
        print("Direct import started")
        while not rospy.is_shutdown():
            ret1, self.image1 = self.cam1.read()
            if not ret1 == True: 
                warnings.warn("couldn't fetch frame from cam1")
                continue
            ret2, self.image2 = self.cam2.read()
            if not ret2 == True: 
                warnings.warn("couldn't fetch frame from cam2")
                continue
            self.image_join()

    def image1_callback(self,msg):
        if self.VERBOSE:
            str = "Image1 received with encoding: " + msg.encoding
            print(str)
        self.set_image(msg, 1)


    def image2_callback(self,msg):
        if self.VERBOSE:
            str = "Image2 received with encoding: " + msg.encoding
            print(str)
        self.set_image(msg, 2)

    def loop(self):
        print("Starting Loop...")
        rospy.spin()
    
    
    def set_image(self, image, number):
        try:
            if not isinstance(image, Image):
                print("Input image is not of type sensor_msgs/Image")
               
            if number == 1:
                self.image1 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            elif number == 2:
                self.image2 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            
            self.image_join()
        except Exception as e:
            print(e)





    def image_join(self):
    # Join images
        if self.image1 is not None and self.image2 is not None:
            if self.VERBOSE:
                print("Joining images")
            try:
                if not isinstance(self.image1, np.ndarray):
                    self.image1 = np.asarray(self.image1)
                if not isinstance(self.image2, np.ndarray):
                    self.image2 = np.asarray(self.image2)
                image_joined = self.stitcher.blending(self.image1, self.image2)
                self.publish_image(image_joined)
                
            except Exception as e :
                print(e)
        else:
            print("At least one Image is None")

    
    def publish_image(self,image_joined):
        image_joined_msg = self.bridge.cv2_to_imgmsg(image_joined, "bgr8")
        self.pub.publish(image_joined_msg)
        

        




if __name__ == '__main__':

   
    value_list = dict()
    for k, v in CameraJoin.param_list.items():
        value_list.update(k= None)
    # ROS Image message -> OpenCV2 image converter
    from cv_bridge import CvBridge, CvBridgeError
    rospy.init_node('camera_join', anonymous=True, log_level=rospy.DEBUG)
    print("Starting Node, Fetching params")
    try:
        for k,v in CameraJoin.param_list.items():
            if rospy.has_param(v):
                value_list.update(k= rospy.get_param(v))
                print("Parameter ", v, "has been found and added" )
            else: 
                print("Parameter ", v, "has not been found and added" )
                continue
        else: 
            my_subs = CameraJoin(value_list)
            
            my_subs.loop()
            print("Node started with given Params")
    except KeyError:
        print("Fetching Params failed, using default params")
        my_subs = CameraJoin(camera1="joined_cams/usb_cam1/image_raw", camera2="joined_cams/usb_cam2/image_raw", joinType=2, static_matrix=False, direct_import=True)
        my_subs.loop()
    except:
        warnings.warn("Couldn't start Node, is Cam1 the right one?")