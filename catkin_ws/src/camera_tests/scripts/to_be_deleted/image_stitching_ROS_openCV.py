
import image_stitching_opencv as stitcher
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

VERBOSE = True
ENCODING = 'rgb8'

class CameraJoin(object):
    
    
    def __init__(self):
        self.image1 = None
        self.image2 = None
        self.bridge = CvBridge()
        self.stitcher = stitcher.Image_Stitching()
        camera1 = "joined_cams/usb_cam1/image_raw"
        camera2 = "joined_cams/usb_cam2/image_raw"
        rospy.Subscriber(camera1, Image, self.image1_callback)
        rospy.Subscriber(camera2, Image, self.image2_callback)
        self.pub = rospy.Publisher('joined_image/image_raw', Image, queue_size=10)


    def image1_callback(self,msg):
        if(VERBOSE):
            print("Image1 received with encoding:", msg.encoding)
        self.set_image(msg, 1)


    def image2_callback(self,msg):
        if(VERBOSE):
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
                self.image1 = self.bridge.imgmsg_to_cv2(image, ENCODING)
                
            elif number == 2:
                self.image2 = self.bridge.imgmsg_to_cv2(image, ENCODING)
            
            self.image_join()
        except CvBridgeError as e:
            print(e)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)





    def image_join(self):
    # Join images
        if (self.image1 is not None) and (self.image2 is not None):         
            if VERBOSE:
                print("Joining images")
            try:
                if not isinstance(self.image1, np.ndarray):
                    self.image1 = np.asarray(self.image1)
                if not isinstance(self.image2, np.ndarray):
                    self.image2 = np.asarray(self.image2)
                image_joined = self.stitcher.blending(self.image1, self.image2)
                self.publish_image(image_joined)
                rospy.sleep(3)
            except Exception as e :
                if VERBOSE:
                    print(e)
                pass
        else:
            if VERBOSE:
                print("At least one Image is None")

        
        



    def publish_image(self,image_joined):
        image_joined_msg = self.bridge.cv2_to_imgmsg(image_joined, "bgr8")
        self.pub.publish(image_joined_msg)

        


if __name__ == '__main__':
    rospy.init_node('camera_join_subscriber', anonymous=True, log_level=rospy.WARN)
    my_subs = CameraJoin()
    my_subs.loop()


