
import image_join as ij
import rospy
from sensor_msgs.msg import Image
import numpy as np

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError



class CameraJoin(object):
    
    
    def __init__(self,camera1 = "joined_cams/usb_cam1/image_raw", camera2 = "joined_cams/usb_cam2/image_raw", publish = "joined_image/image_raw", queue_size = 10, verbose = False, encoding = 'bgr8',joinType = ij.JoinType.CONCAT,  left_y_offset = None, right_y_offset = None, left_x_offset = None, right_x_offset = None, ratio = None, min_match = None,smoothing_window_size = None, matching_write = None, static_matrix = None , stitchter_type = None):
        self.image1 = None
        self.image2 = None
        self.bridge = CvBridge()
        self.VERBOSE = verbose
        self.ENCODING = encoding
        self.stitcher = ij.ImageJoinFactory(joinType ,left_y_offset, right_y_offset, left_x_offset, right_x_offset, ratio, min_match, smoothing_window_size, matching_write, static_matrix, stitchter_type)
        rospy.Subscriber(camera1, Image, self.image1_callback)
        rospy.Subscriber(camera2, Image, self.image2_callback)
        self.pub = rospy.Publisher(publish, Image, queue_size= queue_size)


    def image1_callback(self,msg):
        if(self.VERBOSE):
            print("Image1 received with encoding:", msg.encoding)
        self.set_image(msg, 1)


    def image2_callback(self,msg):
        if(self.VERBOSE):
            print("Image2 received with encoding:", msg.encoding)
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
            print(e)
        except ValueError as e:
            print(e)
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
                if self.VERBOSE:
                    print(e)
                pass
        else:
            if self.VERBOSE:
                print("At least one Image is None")

    
    def publish_image(self,image_joined):
        image_joined_msg = self.bridge.cv2_to_imgmsg(image_joined, "bgr8")
        self.pub.publish(image_joined_msg)

        




if __name__ == '__main__':
    rospy.init_node('camera_join_subscriber', anonymous=True, log_level=rospy.WARN)
    my_subs = CameraJoin()
    my_subs.loop()