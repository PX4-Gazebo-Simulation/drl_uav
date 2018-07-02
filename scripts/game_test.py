#!/usr/bin/env python
# A test platform of game(Environment)
#
# Subscribe: action
# Publish: game(Environment) status: continuously send the status
#
# author: bingbing li 07.01.2018

import rospy
from beginner_tutorials.msg import Input_Game
from beginner_tutorials.msg import Output_Game
# from beginner_tutorials.msg import Num

# Publisher:
pub = rospy.Publisher('game_status', Output_Game, queue_size=10)
msg = Output_Game()
msg.vel1 = 0.5  # intial game status

def callback(data):
    # Subscribing:
    # rospy.loginfo(rospy.get_caller_id() + 'Action: %f', data.action)
    rospy.loginfo('Receiving Action: %f', data.action)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %d', data.num)

def simple_game():

    # initialize the node:
    rospy.init_node('simple_game', anonymous=True)

    # Subscriber:
    rospy.Subscriber('game_input', Input_Game, callback)
    # rospy.Subscriber('custom_chatter', Num, callback)


    # First Publishing:
    rospy.loginfo('Publishing first game_status: %f', msg.vel1)
    pub.publish(msg)

    r = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        # Publishing:
        rospy.loginfo('Publishing game_status: %f', msg.vel1)
        pub.publish(msg)
        r.sleep()


if __name__ == '__main__':
    try:
        simple_game()
    except rospy.ROSInterruptException:
        pass
