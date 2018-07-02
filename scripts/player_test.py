#!/usr/bin/env python
# A test platform of game player(for future DRL)
#
# Subscribe: game(Environment) status
# Publish: action: only sent when game status is received
#
# author: bingbing li 07.01.2018

import rospy
from beginner_tutorials.msg import Input_Game
from beginner_tutorials.msg import Output_Game

# Publisher:
pub = rospy.Publisher('game_input', Input_Game, queue_size=10)
msg = Input_Game()
msg.action = 0.3    # initial action

def callback(data):
    # Subscribing:
    # rospy.loginfo(rospy.get_caller_id() + 'Receive Game Status: %f %f %f %f',
    # data.vel1, data.vel2, data.pos1, data.pos2)
    rospy.loginfo('Receive Game Status: %f %f %f %f', data.vel1, data.vel2, data.pos1, data.pos2)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %d', data.num)
    
    # Publishing:
    rospy.loginfo('Publishing action: %f', msg.action)
    pub.publish(msg)

def simple_game():

    # initialize the node:
    rospy.init_node('player_test', anonymous=True)

    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        simple_game()
    except rospy.ROSInterruptException:
        pass
