import rospy
from sensor_msgs.msg import Image
import time

class FpsCalculator:
    def __init__(self):
        self.frame_count = 0
        self.start_time = 0

    def image_callback(self, image):
        if self.frame_count == 0:
            self.start_time = time.time()

        self.frame_count += 1

        if self.frame_count == 10:  # Berechnung alle 10 Bilder
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            rospy.loginfo("FPS: %.2f" % fps)

            self.frame_count = 0

def main():
    rospy.init_node('fps_calculator')
    calculator = FpsCalculator()
    rospy.Subscriber('/kamera_topic', Image, calculator.image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
