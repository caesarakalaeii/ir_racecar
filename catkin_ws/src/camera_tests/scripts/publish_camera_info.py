#!/usr/bin/env python
"""
source: https://gist.github.com/rossbar/ebb282c3b73c41c1404123de6cea4771#file-yaml_to_camera_info_publisher-py
pointgrey_camera_driver (at least the version installed with apt-get) doesn't
properly handle camera info in indigo.
This node is a work-around that will read in a camera calibration .yaml
file (as created by the cameracalibrator.py in the camera_calibration pkg),
convert it to a valid sensor_msgs/CameraInfo message, and publish it on a
topic.
The yaml parsing is courtesy ROS-user Stephan:
    http://answers.ros.org/question/33929/camera-calibration-parser-in-python/
This file just extends that parser into a rosnode.
"""
import rospy
import yaml
from sensor_msgs.msg import CameraInfo

def yaml_to_CameraInfo(yaml_fname):
    """
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle, Loader=yaml.FullLoader)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

if __name__ == "__main__":
    # Get fname from command line (cmd line input required)
    #import argparse
    #arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument("filename", help="Path to yaml file containing " +\
    #                                         "camera calibration data")
    #args = arg_parser.parse_args()
    #filename = args.filename
    filename1 = "/home/rtlabor/.ros/camera_info/usb_cam1.yaml"
    filename2 = "/home/rtlabor/.ros/camera_info/usb_cam2.yaml"
    # Parse yaml file
    camera_info_msg1 = yaml_to_CameraInfo(filename1)
    camera_info_msg2 = yaml_to_CameraInfo(filename2)

    # Initialize publisher node
    rospy.init_node("camera_info_publisher", anonymous=True)
    publisher1 = rospy.Publisher("joined_cams/usb_cam1/camera_info", CameraInfo, queue_size=10)
    publisher2 = rospy.Publisher("joined_cams/usb_cam2/camera_info", CameraInfo, queue_size=10)
    rate = rospy.Rate(10)

    # Run publisher
    while not rospy.is_shutdown():
        publisher1.publish(camera_info_msg1)
        publisher2.publish(camera_info_msg2)
        rate.sleep()