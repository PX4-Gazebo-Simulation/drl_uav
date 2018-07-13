#!/usr/bin/env python
# An implementation of UAV DRL

# Func:
# 1) kepp the pos1 fixed in 6+-0.3  D!
#
# Implementation:
# 1) Work with player_test.py   D!
#
# Subscribe: game(Environment) status
# Publish: action: only sent when game status is received
#
# author: bingbing li 07.02.2018

import rospy
from drl_uav.msg import Num
from drl_uav.msg import Input_Game
from drl_uav.msg import Output_Game

# get UAV status:
from geometry_msgs.msg import PoseStamped	# UAV pos status
from geometry_msgs.msg import TwistStamped	# UAV vel status
from drl_uav.msg import Restart_Finished # UAV restart finished
from drl_uav.msg import AttControlRunning    # UAV att_control running: ready for Memory::observe().
from drl_uav.msg import AttitudeTarget       # UAV att setpoint(thrust is used)

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random, numpy, math, gym

import sys

class Brain:
    def __init__(self, num_state, num_action, RL_GAMMA = 0.99):
        self.num_state = num_state
        self.num_action = num_action
        self.model = self._createModel()
        # self.model.load_weights("cartpole_libn.h5")
        # self.model.load_weights("DRL_libn_13272.h5")
        # self.model.load_weights("DRL_UAV_2326.h5")
        # self.model.load_weights("DRL_UAV_revised_303.h5")
        # self.model.load_weights("DRL_UAV_latest_4_full_connection.h5")
        # self.model.load_weights("DRL_UAV_latest.h5")
     
        # parameters for RL algorithm:
        self.GAMMA = RL_GAMMA

    def _createModel(self): # model: state -> v(state value)
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.num_state))
        # model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_action, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        
        return model
    
    def train(self, x, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, batch_states, verbose=0):   # batch prediction!
        # input type: state=[state1, state2,...]  -> type: list!
        # e.g.:
        # [array([-0.02851337,  0.04295018, -0.0197721 , -0.00788878]), array([-0.02851337,  0.04295018, -0.0197721 , -0.00788878])]
        # print("Batch len_state: ",len(batch_states))
        # print("Batch state",batch_states)
        return self.model.predict(batch_states, verbose=verbose)
    
    def predictOne(self, state_test):           # solo prediction! 
        # (You have to reshape the input!!!)
        # input type: state_test                -> type: array!
        # e.g.:
        # [-0.02851337  0.04295018 -0.0197721  -0.00788878]
        # reshape: state_test.reshape(1, self.num_state) =>
        # array([[-0.02851337,  0.04295018, -0.0197721 , -0.00788878]])
        # print("One len_state: ",len(state_test))
        # print("One state",state_test)
        return self.predict(state_test.reshape(1, self.num_state)).flatten()


class Memory:
    def __init__(self, memory_capacity):      
        self.memory_capacity = memory_capacity
        self.samples = []
    def add(self, experience):  # experience: [state, action, reward, state_next]
        self.samples.append(experience)
        if len(self.samples) > self.memory_capacity:
            self.samples.pop(0)     # if full, FIFO
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    def num_experience(self):   # return the number of experience!
        return len(self.samples)


class Agent:
    steps = 0
    def __init__(self, num_state, num_action):
        # parameters of External Environment:
        self.num_state = num_state
        self.num_action = num_action

        # parameters of Internal DRL algorithm:
        ## Memory:
        self.MEMORY_CAPACITY = 100000
        ## RL algorithm:
        self.GAMMA = 0.99
        ## Deep network: 
        self.MEMORY_BATCH_SIZE = 64 # number of data for one training! ?(Maybe we can set MEMORY_BATCH_SIZE = MEMORY_CAPACITY)
        ## Random selection proportion:
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.01
        self.LAMBDA = 0.0015  # speed of decay

        self.epsilon = self.MAX_EPSILON
        self.brain = Brain(num_state, num_action, RL_GAMMA=self.GAMMA)
        self.memory = Memory(self.MEMORY_CAPACITY)

    def act(self, state):   # action:[0,1,2,...,num_action-1]

        # Limit: 3) forced input in Emergency: Vz is out of [-3,3].
        global UAV_Vel
        if(UAV_Vel.twist.linear.z > 3.0):
            return 0
        if(UAV_Vel.twist.linear.z < -3.0):
            return 1

        if random.random() < self.epsilon:
            return random.randint(0, self.num_action-1)
        else:
            return numpy.argmax(self.brain.predictOne(state_test=state))  # get the index of the largest number, that is the action we should take. -libn

    def observe(self, experience):
        self.memory.add(experience)

        # decrease Epsilon to reduce random action and trust more in greedy algorithm
        self.steps += 1
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.steps)

    def replay(self):   # get knowledge from experience!
        batch = self.memory.sample(self.MEMORY_BATCH_SIZE)
        # batch = self.memory.sample(self.memory.num_experience())  # the training data size is too big!
        len_batch = len(batch)

        no_state = numpy.zeros(self.num_state)

        batch_states = numpy.array([o[0] for o in batch])
        batch_states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch])

        # print('Batch states:')
        # print(batch_states)
        # print('Batch states_:')
        # print(batch_states_)
        
        v = self.brain.predict(batch_states)
        v_ = self.brain.predict(batch_states_)

        # inputs and outputs of the Deep Network:
        x = numpy.zeros((len_batch, self.num_state))
        y = numpy.zeros((len_batch, self.num_action))

        for i in range(len_batch):
            o = batch[i]
            s = o[0]; a = int(o[1]); r = o[2]; s_ = o[3]

            v_t = v[i]
            if s_ is None:
                v_t[a] = r
            else:
                v_t[a] = r + self.GAMMA * numpy.amax(v_[i]) # We will get max reward if we select the best option.

            x[i] = s
            y[i] = v_t

        self.brain.train(x, y, batch_size=len_batch)

# get UAV status:
UAV_Vel = TwistStamped()
UAV_Pos = PoseStamped()
att_running = AttControlRunning()
UAV_Att_Setpoint = AttitudeTarget()

def UAV_pos_callback(data):
    global UAV_Pos  
    # Subscribing:
    # rospy.loginfo('Receive UAV Pos Status: %f %f %f', data.pose.position.x, data.pose.position.y, data.pose.position.z)
    UAV_Pos = data

def UAV_vel_callback(data):
    global UAV_Vel
    # Subscribing:
    # rospy.loginfo('Receive UAV Vel Status: %f %f %f', data.twist.linear.x, data.twist.linear.y, data.twist.linear.z)  
    UAV_Vel = data

def restart_finished_callback(data):
    # Subscribing:
    # rospy.loginfo('UAV restart finished: %d', data.finished)
    pass


def att_running_callback(data):
    global att_running
    # Subscribing:
    # rospy.loginfo('UAV att running!: %d', data.running)
    att_running = data
    # rospy.loginfo('att_running!:  ~~~~~~~~~~~~~~~~~~~~~~ %d ~~~~~~~~~~~~~~~~~~~~', att_running.running)

def local_attitude_setpoint__callback(data):
    global UAV_Att_Setpoint
    # Subscribing:
    # rospy.loginfo('thrust: %f', data.thrust)
    UAV_Att_Setpoint = data

# Publisher:
pub = rospy.Publisher('game_input', Input_Game, queue_size=10)
env_input = Input_Game()
env_input.action = 1    # initial action

current_status = Output_Game()

def status_update(data):
    # Subscribing:
    # rospy.loginfo(rospy.get_caller_id() + 'Receive Game Status: %f %f %f %f',
    # data.vel1, data.vel2, data.pos1, data.pos2)
    # rospy.loginfo('Receive Game Status: %f %f %f %f', data.vel1, data.vel2, data.pos1, data.pos2)
    global current_status
    current_status = data
    # rospy.loginfo('Receive Game Status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %d', data.num)
    

   
def interact():
    # # current_status.pos1:[2, 10] => failed = False
    # # current_status.pos:[5.7, 6.3] => done = True
    # global current_status
    # # 1) get pre_status = current_status
    # pre_status = current_status
    # # rospy.loginfo('pre_status.pos1 = %f', pre_status.pos1)
    # # 2) publish action
    # # rospy.loginfo('Publishing action: %f', env_input.action)
    # pub.publish(env_input)
    # # 3) judge from current_status: calculate: r, done, failed
    # # 4) return current_status, reward, done, failed(NOT Used!)
    # state_ = numpy.array(current_status.pos1)
    # if (current_status.pos1 > 10.0 or current_status.pos1 < 2.0):
    #     done = True
    #     return state_, -0.5, done, True
    # # reward = 10.0 / (numpy.square(current_status.pos1 - 6.0) + 1.0)
    # done = False
    # reward = 0.0
    # if (math.fabs(current_status.pos1 - 6.0) < 0.3):
    #     reward = 1.0
    # return state_, reward, done, False

    # publish env_input(action):
    global pub, env_input
    # get UAV status:
    global UAV_Vel, UAV_Pos

    # 1) publish action
    # rospy.loginfo('Publishing action: %f', env_input.action)
    pub.publish(env_input)


    # 2) judge from current_status: calculate: r, done, failed
    # 3) return current_status, reward, done, failed(NOT Used!)
    normalized_pos_z = (UAV_Pos.pose.position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
    normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
    normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
    state_ = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
    
    done = False
    reward = 0.0

    if((UAV_Pos.pose.position.z > 30.0) or (UAV_Pos.pose.position.z < 10.0)): # allowed trial height:[8m,30m], release_height=restart_height=15m
        done = True                                               # Restart the env:
        rospy.loginfo("Let's restart!")
   
    if (math.fabs(UAV_Pos.pose.position.z - 20.0) < 0.3):
        reward = 1.0

    return state_, reward, done, True






def env_restore():
    # 1) publish pos destination: [0,0,3]
    # 2) judge if pos arrived?
    # 3) hover for 1 second -> break!
    # sleep for 1 seconds
    rospy.sleep(1.)

def main_loop():

    global current_status, pub, env_input

    # get UAV status:
    global UAV_Vel, UAV_Pos, att_running, UAV_Att_Setpoint

    rospy.init_node('custom_talker', anonymous=True)

    # 1) get current status:
    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, status_update) 
    # rospy.loginfo('current_status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)

    # Subscriber:
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, UAV_pos_callback)
    # Subscriber:
    rospy.Subscriber('mavros/local_position/velocity', TwistStamped, UAV_vel_callback)

    # Subscriber:
    rospy.Subscriber('restart_finished_msg', Restart_Finished, restart_finished_callback)

    # Subscriber:
    rospy.Subscriber('att_running_msg', AttControlRunning, att_running_callback)

    # Subscriber:
    rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget, local_attitude_setpoint__callback)

    # 2) take action
    num_state = 3   # state=[UAV_height, UAV_vertical_vel, , UAV_Att_Setpoint.thrust]
    num_action = 2  # action=[0,1]
    agent = Agent(num_state, num_action)
    R = 0
    n = 0
    model_saved = 0
    num_trial = 0   # the number of current trial
    new_trial = False

    r = rospy.Rate(20)  # 20Hz

    # get states:
    normalized_pos_z = (UAV_Pos.pose.position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
    normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
    normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
    state = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
    # take action:
    env_input.action = agent.act(state)

    output_file_name = 'result_output.txt'  # record the training result


    while not rospy.is_shutdown():
        # rospy.loginfo('UAV Vel Status: %f %f %f', UAV_Vel.twist.linear.x, UAV_Vel.twist.linear.y, UAV_Vel.twist.linear.z)  
        # rospy.loginfo('UAV Pos Status: %f %f %f', UAV_Pos.pose.position.x, UAV_Pos.pose.position.y, UAV_Pos.pose.position.z)
        # rospy.loginfo('main loop: att_running.running: %d', att_running.running)

        # print("att_running.running: %d", att_running.running)
        if(att_running.running):
            
            n += 1

            state_, reward, done, failed = interact()
            if done:
                state_ = None
                # record the memory:
                rospy.loginfo('Memory: state(Pos, Vel, thrust): %f, %f, %f  action: %f  reward: %f state_: %f, %f, %f', state[0], state[1], state[2], env_input.action, reward, 0.0, 0.0, 0.0)                                        
            else:
                # record the memory:
                rospy.loginfo('Memory: state(Pos, Vel, thrust): %f, %f, %f  action: %f  reward: %f state_: %f, %f, %f', state[0], state[1], state[2], env_input.action, reward, state_[0], state_[1], state_[2])

                # ignore final experience(state_ = None) for action = -1 then, and will lead to no value of v_[action] in RL.
                try:
                    agent.observe((state, env_input.action, reward, state_))
                    agent.replay()                                
                except KeyboardInterrupt:
                    print('Interrupted')
                    # break
                    sys.exit(0)



            # agent.observe((state, env_input.action, reward, state_))
            # agent.replay()

            R += reward
            # rospy.loginfo('current action: %f', env_input.action)
            # rospy.loginfo('current reward: %f', reward)

            # # display the result in star figure:
            # state_scale = int(state*5.0)
            # # rospy.loginfo('state = %f', state)
            # for i in range(50):
            #     if(i == state_scale):
            #         print 'x',
            #     else:
            #         print '_',
            # print('Total reward:', R)

            # prepare for the next loop:
            # get states:
            normalized_pos_z = (UAV_Pos.pose.position.z - 20.0) / 10.0      # UAV_Pos.pose.position.z: [10, 30]     -> normalized_pos_z: [][-1, 1]
            normalized_vel_z = UAV_Vel.twist.linear.z / 3.0                 # UAV_Vel.twist.linear.z: [-3, 3]       -> normalized_vel_z: [-1, 1]
            normalized_thrust = (UAV_Att_Setpoint.thrust - 0.59) / 0.19     # UAV_Att_Setpoint.thrust: [0.4, 0.78]  -> normalized_thrust: [-1, 1]
            state = numpy.array((normalized_pos_z, normalized_vel_z, normalized_thrust))
            # take action:
            env_input.action = agent.act(state)

            if done:    # restart(these code may run several times because of the time delay)!
                env_input.action = -1.0    # Restart the game!
                pub.publish(env_input)
                rospy.loginfo('Restarting!')

            
            if((new_trial == True) and done):       # to make sure this loop runs only once!
                num_trial += 1
                new_trial = False

                # record the trial result:  # stored in $HOME folder!
                with open(output_file_name, 'a') as f:
                    f.write(str(num_trial) + 'th trial: ' + ' Total reward: ' + str(R) + '\n')

                rospy.sleep(0.1)

                # save model:
                if (R > 300.0):
                    model_saved += 1
                    agent.brain.model.save("DRL_UAV_revised_"+str(int(R))+".h5")
                    # agent.brain.model.save("DRL_libn.h5")
                n = 0
                R = 0.0
            
            rospy.loginfo('%d th trial: n: %d current state(Pos, Vel, thrust): %f, %f, %f  current action: %f  current reward: %f Total reward: %f', num_trial, n, 
                UAV_Pos.pose.position.z, UAV_Vel.twist.linear.z, UAV_Att_Setpoint.thrust, env_input.action, reward, R)
            
        else:   # restarting!
            new_trial = True
            # publish random action(0/1) to stop Env-restart(-1) commander!
            env_input.action =  random.randint(0, 1)    # Restart the game!
            # rospy.loginfo('Random action: %f', env_input.action)
            pub.publish(env_input)

        r.sleep()

        # sleep for 10 seconds
        # rospy.sleep(10.)

  
    agent.brain.model.save("DRL_UAV_latest.h5")
    print("Running: Total reward:", R)
            

if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print 'Interrupted'
        sys.exit(0)  

# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         print 'Interrupted'
#         sys.exit(0)        
