#!/usr/bin/env python
import image_join as ij
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError



class CameraJoin(object):
    
    
    def __init__(self,camera1 = "joined_cams/usb_cam1/image_raw", camera2 = "joined_cams/usb_cam2/image_raw", publish = "joined_image/image_raw", queue_size = 10, encoding = 'bgr8',joinType = 1,  left_y_offset = 20, right_y_offset = 0, left_x_offset = 0, right_x_offset = 0, ratio = 0.85, min_match = 10,smoothing_window_size = 50, matching_write = False, static_matrix = False, static_mask = False , stitchter_type = cv2.Stitcher_PANORAMA):
        self.image1 = None
        self.image2 = None
        self.bridge = CvBridge()
        self.ENCODING = encoding
        self.stitcher = ij.ImageJoinFactory.create_instance(joinType ,left_y_offset, right_y_offset, left_x_offset, right_x_offset, ratio, min_match, smoothing_window_size, matching_write, static_matrix, static_mask, stitchter_type)
        rospy.Subscriber(camera1, Image, self.image1_callback)
        rospy.Subscriber(camera2, Image, self.image2_callback)
        self.pub = rospy.Publisher(publish, Image, queue_size= queue_size)
        rospy.loginfo("Node sucessfully initialized")


    def image1_callback(self,msg):
        str = "Image1 received with encoding: " + msg.encoding
        rospy.logdebug(str)
        self.set_image(msg, 1)


    def image2_callback(self,msg):
        str = "Image2 received with encoding: " + msg.encoding
        rospy.logdebug(str)
        self.set_image(msg, 2)

    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()
    
    
    def set_image(self, image, number):
        try:
            if not isinstance(image, Image):
                raise ValueError("Input image is not of type sensor_msgs/Image")
               
            if number == 1:
                self.image1 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            elif number == 2:
                self.image2 = self.bridge.imgmsg_to_cv2(image, self.ENCODING)
            
            self.image_join()
        except CvBridgeError as e:
            rospy.logerr(e)
        except ValueError as e:
            rospy.logerr(e)
        except Exception as e:
            rospy.logerr(e)





    def image_join(self):
    # Join images
        if self.image1 is not None and self.image2 is not None:
            rospy.logdebug("Joining images")
            try:
                if not isinstance(self.image1, np.ndarray):
                    self.image1 = np.asarray(self.image1)
                if not isinstance(self.image2, np.ndarray):
                    self.image2 = np.asarray(self.image2)
                image_joined = self.stitcher.blending(self.image1, self.image2)
                self.publish_image(image_joined)
                
            except Exception as e :
                rospy.logerr(e)
        else:
            rospy.logdebug("At least one Image is None")

    
    def publish_image(self,image_joined):
        image_joined_msg = self.bridge.cv2_to_imgmsg(image_joined, "bgr8")
        self.pub.publish(image_joined_msg)

        




if __name__ == '__main__':
    rospy.init_node('camera_join', anonymous=True, log_level=rospy.DEBUG)
    rospy.loginfo("Starting Node, Fetching params")
    try:
        camera1 = rospy.get_param("/camera_join/camera1")
        camera2 = rospy.get_param("/camera_join/camera2")
        publish = rospy.get_param("/camera_join/publish")
        queue_size = rospy.get_param("/camera_join/queue_size")
        verbose = rospy.get_param("/camera_join/verbose")
        encoding = rospy.get_param("/camera_join/camera1")
        joinType = rospy.get_param("/camera_join/type")
        if joinType == 1:
            left_y_offset = rospy.get_param("/camera_join/left_y_offset")
            right_y_offset = rospy.get_param("/camera_join/right_y_offset")
            left_x_offset = rospy.get_param("/camera_join/left_x_offset")
            right_x_offset = rospy.get_param("/camera_join/right_x_offset")
        if joinType == 2:
            ratio = rospy.get_param("/camera_join/ratio")
            min_match = rospy.get_param("/camera_join/min_match")
            smoothing_window_size = rospy.get_param("/camera_join/smoothing_window_size")
            matching_write = rospy.get_param("/camera_join/matching_write")
            static_matrix = rospy.get_param("/camera_join/static_matrix")
            static_mask = rospy.get_param("/camera_join/static_mask")
        else: 
            stitchter_type = rospy.get_param("/camera_join/stitchter_type")
            #my_subs = CameraJoin(joinType ,left_y_offset, right_y_offset, left_x_offset, right_x_offset, ratio, min_match, smoothing_window_size, matching_write, static_matrix, static_mask, stitchter_type)
            my_subs = CameraJoin(camera1="joined_cams/usb_cam1/image_rect", camera2="joined_cams/usb_cam2/image_rect")
            my_subs.loop()
            rospy.loginfo("Node started with given Params")
    except KeyError:
        rospy.logerr("Fetching Params failed, using default params")
        my_subs = CameraJoin(camera1="joined_cams/usb_cam1/image_rect", camera2="joined_cams/usb_cam2/image_rect", joinType=2, static_matrix=True)
        my_subs.loop()
    except:
        rospy.logfatal("Couldn't start Node")