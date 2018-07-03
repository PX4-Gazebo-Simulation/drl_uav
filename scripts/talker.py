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
from beginner_tutorials.msg import Num
from beginner_tutorials.msg import Input_Game
from beginner_tutorials.msg import Output_Game

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
        self.model.load_weights("DRL_libn_13272.h5")

        # parameters for RL algorithm:
        self.GAMMA = RL_GAMMA

    def _createModel(self): # model: state -> v(state value)
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.num_state))
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
        self.LAMBDA = 0.001  # speed of decay

        self.epsilon = self.MAX_EPSILON
        self.brain = Brain(num_state, num_action, RL_GAMMA=self.GAMMA)
        self.memory = Memory(self.MEMORY_CAPACITY)

    def act(self, state):   # action:[0,1,2,...,num_action-1]
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
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            v_t = v[i]
            if s_ is None:
                v_t[a] = r
            else:
                v_t[a] = r + self.GAMMA * numpy.amax(v_[i]) # We will get max reward if we select the best option.

            x[i] = s
            y[i] = v_t

        self.brain.train(x, y, batch_size=len_batch)


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
    # current_status.pos1:[2, 10] => failed = False
    # current_status.pos:[5.7, 6.3] => done = True
    global current_status
    # 1) get pre_status = current_status
    pre_status = current_status
    # rospy.loginfo('pre_status.pos1 = %f', pre_status.pos1)
    # 2) publish action
    # rospy.loginfo('Publishing action: %f', env_input.action)
    pub.publish(env_input)
    # 3) judge from current_status: calculate: r, done, failed
    # 4) return current_status, reward, done, failed(NOT Used!)
    state_ = numpy.array(current_status.pos1)
    if (current_status.pos1 > 10.0 or current_status.pos1 < 2.0):
        done = True
        return state_, -0.5, done, True
    # reward = 10.0 / (numpy.square(current_status.pos1 - 6.0) + 1.0)
    done = False
    reward = 0.0
    if (abs(current_status.pos1 - 6.0) < 0.3):
        reward = 1.0
    return state_, reward, done, False

def env_restore():
    # 1) publish pos destination: [0,0,3]
    # 2) judge if pos arrived?
    # 3) hover for 1 second -> break!
    # sleep for 1 seconds
    rospy.sleep(1.)

def main_loop():

    global current_status, pub, env_input

    rospy.init_node('custom_talker', anonymous=True)

    # 1) get current status:
    # Subscriber:
    rospy.Subscriber('game_status', Output_Game, status_update) 
    # rospy.loginfo('current_status: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)

    # 2) take action
    num_state = 1
    num_action = 2  # action=[0,1]
    agent = Agent(num_state, num_action)
    R = 0
    n = 0
    model_saved = 0
    while True:
        state = current_status.pos1
        state = numpy.array(state)
        # rospy.loginfo('current_status1111: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)

        # # sleep for 10 seconds
        # rospy.sleep(0.1)
        # rospy.loginfo('current_status2222: %f %f %f %f', current_status.vel1, current_status.vel2, current_status.pos1, current_status.pos2)
        n += 1

        env_input.action = agent.act(state)

        # sleep for 10 seconds
        # rospy.sleep(10.)


        state_, reward, done, failed = interact()
        if done:
            state_ = None

        try:
            agent.observe((state, env_input.action, reward, state_))
            agent.replay()
        except KeyboardInterrupt:
            print('Interrupted')
            break
            sys.exit(0)

        # agent.observe((state, env_input.action, reward, state_))
        # agent.replay()

        R += reward
        rospy.loginfo('n: %d current state: %f  current action: %f  current reward: %f Total reward: %f', n, state, env_input.action, reward, R)
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

        if done:    # restart!
            env_input.action = 200.0    # Restart the game!
            pub.publish(env_input)
            if (R > 300.0):
                model_saved += 1
                agent.brain.model.save("DRL_libn_"+str(int(R))+".h5")
                # agent.brain.model.save("DRL_libn.h5")
            R = 0.0

  

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
