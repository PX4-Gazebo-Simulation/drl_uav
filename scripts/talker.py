#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from beginner_tutorials.msg import Num

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random, numpy, math, gym

class Brain:
    def __init__(self, num_state, num_action, RL_GAMMA = 0.99):
        self.num_state = num_state
        self.num_action = num_action
        self.model = self._createModel()
        # self.model.load_weights("cartpole_libn.h5")

        # parameters for RL algorithm:
        self.GAMMA = RL_GAMMA

    def _createModel(self): # model: state -> v(state value)
        model = Sequential()

        # 2) #2 Adaption!
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

        # 3) #3 Adaption!
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

        # 1) #1 Adaption!
        no_state = numpy.zeros(self.num_state)

        batch_states = numpy.array([o[0] for o in batch])
        batch_states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch])
        
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

# 3) #3 Adaption!
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        state = self.env.reset()
        R = 0
        while True:
            self.env.render()
            action = agent.act(state)
            state_, r, done, info = self.env.step(action)

            # 1) #1 Adaption!
            if done:
                state_ = None

            agent.observe((state, action, r, state_))
            agent.replay()

            state = state_
            R += r

            if done:
                break
        
        print("Running: Total reward:", R)

def talker():

    ## + custom message:
    pub_custom = rospy.Publisher('custom_chatter', Num)
    rospy.init_node('custom_talker', anonymous=True)
    r = rospy.Rate(100)  # 100Hz
    msg = Num()
    msg.num = 5

    # DRL:
    PROBLEM = 'CartPole-v0'
    env = Environment(PROBLEM)
    num_state = env.env.observation_space.shape[0]
    num_action = env.env.action_space.n
    agent = Agent(num_state, num_action)

    while not rospy.is_shutdown():
        
        ## + custom message:
        rospy.loginfo(msg)
        pub_custom.publish(msg)

        # DRL:
        env.run(agent)

        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
