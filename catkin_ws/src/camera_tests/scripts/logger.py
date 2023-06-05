has_ros = True
try:
    import rospy #checks if rospy is present
except ImportError:
            has_ros = False 


class Logger():
    def __init__(self, ros_log = False, console_log = False):
        if has_ros:
            self.ros_log = ros_log
        else:
            self.ros_log = False #set Ros loggin to false if rospy has not been detected
        
        self.console_log = console_log

    def warning(self, skk): #yellow
        
        if self.console_log:
            print("\033[93m {}\033[00m" .format("WARNING:"),"\033[93m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logwarn(skk)
       
    def error(self, skk): #red
        if self.console_log:   
            print("\033[91m {}\033[00m" .format("ERROR:"),"\033[91m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logerr(skk)
        
    def fail(self, skk): #red
        if self.console_log: 
            print("\033[91m {}\033[00m" .format("FATAL:"),"\033[91m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.logfatal(skk)
    def passing(self, skk): #green
        if self.console_log: 
            print("\033[92m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)
    def passingblue(self, skk): #blue
        if self.console_log: 
            print("\033[96m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)
    def info(self, skk): #blue
        if self.console_log: 
            print("\033[94m {}\033[00m" .format("Info:"),"\033[94m {}\033[00m" .format(skk))
        if self.ros_log:
            rospy.loginfo(skk)