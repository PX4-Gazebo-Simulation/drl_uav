#!/usr/bin/env python
# A test platform of game player(for future DRL)
#
# Subscribe: game(Environment) status
# Publish: action: only sent when game status is received
#
# author: bingbing li 07.01.2018

import rospy
from drl_uav.msg import Input_Game
from drl_uav.msg import Output_Game
from drl_uav.msg import PoseStamped	# UAV pos status
from drl_uav.msg import TwistStamped	# UAV vel status
from drl_uav.msg import Restart_Finished # UAV restart finished
import random

# Publisher:
pub = rospy.Publisher('game_input', Input_Game, queue_size=10)
msg = Input_Game()
msg.action = 0.3    # initial action

def output_game_callback(data):
    # Subscribing:
    # rospy.loginfo(rospy.get_caller_id() + 'Receive Game Status: %f %f %f %f',
    # data.vel1, data.vel2, data.pos1, data.pos2)
    rospy.loginfo('Receive Game Status: %f %f %f %f', data.vel1, data.vel2, data.pos1, data.pos2)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %d', data.num)
    
    # Publishing:
    rospy.loginfo('Publishing action: %f', msg.action)
    pub.publish(msg)

UAV_Vel = TwistStamped()
def UAV_pos_callback(data):
    # Subscribing:
    rospy.loginfo('Receive UAV Pos Status: %f %f %f', data.pose.position.x, data.pose.position.y, data.pose.position.z)
    rospy.loginfo('Receive UAV Vel Status: %f %f %f', UAV_Vel.twist.linear.x, UAV_Vel.twist.linear.y, UAV_Vel.twist.linear.z)

    # Publishing:
    random_action =  random.randint(0, 1)
    msg.action = random_action

    # Restart the env:
    if((data.pose.position.z > 30.0) or (data.pose.position.z < 8.0)):  # allowed trial height:[8m,30m], release_height=restart_height=15m
        msg.action = 200.0

    rospy.loginfo('Publishing action: %f', msg.action)
    pub.publish(msg)

def UAV_vel_callback(data):
    global UAV_Vel
    # Subscribing:
    # rospy.loginfo('Receive UAV Vel Status: %f %f %f', data.twist.linear.x, data.twist.linear.y, data.twist.linear.z)  
    UAV_Vel = data


def restart_finished_callback(data):
    # Subscribing:
    rospy.loginfo('UAV restart finished: %d', data.finished)

    
def simple_game():

    # initialize the node:
    rospy.init_node('player_test', anonymous=True)

    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, output_game_callback)

    # Subscriber:
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, UAV_pos_callback)
    # Subscriber:
    rospy.Subscriber('mavros/local_position/velocity', TwistStamped, UAV_vel_callback)

    # Subscriber:
    rospy.Subscriber('restart_finished_msg', Restart_Finished, restart_finished_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        simple_game()
    except rospy.ROSInterruptException:
        pass
