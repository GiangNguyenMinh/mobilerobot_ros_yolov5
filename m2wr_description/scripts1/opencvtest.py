#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError 
import cv2 as cv
import sys
import rospy
from std_msgs.msg import String 
from sensor_msgs.msg import Image

class imgae_converter: 

    def __init__(self): 
        self.image_pub = rospy.Publisher("imge_topic_cv", Image)
        self.image_sub = rospy.Subscriber("m2wr/camera/image_raw", Image, self.callback)
        self.bridge = CvBridge()

    def callback(self, data): 
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        except CvBridgeError as e: 
            print(e)

        rospy.loginfo(str(cv_image.shape))

        cv.circle(cv_image, (20, 20), 5, 255)
        cv.putText(cv_image, 'Giang', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv.imshow("camera", cv_image)
        cv.waitKey(3)

        try: 
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e: 
            print(e)

        
def main(args): 
    ic = imgae_converter()
    rospy.init_node("image_converter", anonymous=True)
    try: 
        rospy.spin()
    except KeyboardInterrupt: 
        print('Shutdow')
    cv.destroyAllWindows()


if __name__ == '__main__': 
    main(sys.argv)

        
