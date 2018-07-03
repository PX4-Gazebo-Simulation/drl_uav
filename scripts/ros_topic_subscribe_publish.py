#!/usr/bin/env python
# Func: publish UAV pos information

import rospy

from geometry_msgs.msg import PoseStamped

current_pos = PoseStamped()

def callback(data):
    # Subscribing:
    rospy.loginfo('Receive Game Status: %f %f %f', data.pose.position.x, data.pose.position.x, data.pose.position.x)
    
    # data.pose.position.x = 0;
    # data.pose.position.y = 0;
    # data.pose.position.z = 1;
    # data.pose.orientation.x = 0;			/* orientation expressed using quaternion. -libn */
    # data.pose.orientation.y = 0;			/* w = cos(theta/2), x = nx * sin(theta/2),  y = ny * sin(theta/2), z = nz * sin(theta/2) -libn */
    # data.pose.orientation.z = 0.707;
    # data.pose.orientation.w = 0.707;


def get_UAV_pos():

    # initialize the node:
    rospy.init_node('get_UAV_pos', anonymous=True)

    # Subscriber:
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        get_UAV_pos()
    except rospy.ROSInterruptException:
        pass
