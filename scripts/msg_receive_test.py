#!/usr/bin/env python
import rospy
from beginner_tutorials.msg import AttControlRunning 
from beginner_tutorials.msg import Restart_Finished 
from beginner_tutorials.msg import AttitudeTarget

att_running = AttControlRunning()
def att_running_callback(data):
    global att_running
    # Subscribing:
    # rospy.loginfo('UAV att running!: %d', data.running)
    att_running = data

restart_finished = Restart_Finished()
def restart_finished_callback(data):
    # Subscribing:
    # rospy.loginfo('Restarted: %d', data.finished)
    restart_finished = data    

def local_attitude_callback(data):
    # Subscribing:
    rospy.loginfo('thrust: %f', data.thrust)
    
# def simple_game():    
# global att_running, restart_finished

# initialize the node:
rospy.init_node('msg_receive_test', anonymous=True)
# Subscriber:
rospy.Subscriber('att_running_msg', AttControlRunning, att_running_callback)

# Subscriber:
rospy.Subscriber('restart_finished_msg', Restart_Finished, restart_finished_callback)

# Subscriber:
rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget, local_attitude_callback)


while True:
    pass
    # rospy.loginfo('Main loop: UAV att running!: %d', att_running.running)
    # rospy.loginfo('Main loop: Restart finished: %d', restart_finished.finished)

# spin() simply keeps python from exiting until this node is stopped
rospy.spin()

# if __name__ == '__main__':
#     try:
#         simple_game()
#     except rospy.ROSInterruptException:
#         pass
