import numpy.random as npr
import numpy as np
import sys
import csv

from SwingyMonkey import SwingyMonkey

gamma = 0.9
alpha = 0.1
deltaY = 20
deltaX = 50
epsilon = 0.2

class Learner:
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = {} # map State, Action pairs to real numbers.
        for x in xrange(-600/deltaX, 600/deltaX):
            for y in xrange(-400/deltaY,400/deltaY):
                self.Q[(x,y), 0] = 0.4
                self.Q[(x,y), 1] = 0.1

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    #computes best action using Q function for next state
    def maxQ(self, state):
        v1 = self.Q[state,1]
        v2 = self.Q[state,0]
        return (v1,1) if (v1 > v2) else (v2,0)

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        hdist_to_trunk = state['tree']['dist']
        monkey_height = state['monkey']['bot']
        lower_trunk_height = state['tree']['bot']

        y = np.floor((monkey_height - lower_trunk_height) * 1. / deltaY)
        x = np.floor(hdist_to_trunk *1. /deltaX)

        curState = (x,y)

        if (self.last_state != None):
            previousQ = self.Q[self.last_state,self.last_action] 
            optimal_value, optimal_action = self.maxQ(curState)

            updateVal = (self.last_reward + gamma * optimal_value - previousQ)
            self.Q[self.last_state,self.last_action] = previousQ + alpha * updateVal

            self.last_action = optimal_action

        else:
            self.last_action = npr.rand() < epsilon

        self.last_state = curState
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

iters = 100
learner = Learner()

scores = []
for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    scores.append(swing.score)

    # Reset the state of the learner.
    learner.reset()

#output scores
results = open("testStandard.csv",'wb')
writer = csv.writer(results)
for score in scores:
    writer.writerow([score])
