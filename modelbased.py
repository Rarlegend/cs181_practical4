import numpy as np
import numpy.random as npr
import scipy.linalg
import sys
import math
import random

from SwingyMonkey import SwingyMonkey

class ModelBasedLearner:

    def __init__(self):

        self.gamma = 0.5
        self.epsilon = 0

        self.tBottomRange = (0, 400)
        self.tBottomBuckets = 10
        self.tTopRange = (0, 400)
        self.tTopBuckets = 10
        self.tDistRange = (0, 600)
        self.tDistBuckets = 10
        self.mVRange = (-50,50)
        self.mVBuckets = 10
        self.mBottomRange = (0, 450)
        self.mBottomBuckets = 10
        self.mTopRange = (0, 450)
        self.mTopBuckets = 10

        dims = self.basis_dimensions()
        self.N = np.ones(dims + (2,))
        self.R = np.zeros(dims + (2,))
        self.Np = np.zeros(dims + (2,) + dims)

        self.Pi = np.zeros(dims)
        self.V = np.zeros(dims)

        self.Q = np.zeros(dims + (2,))

        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        if (random.random() < self.epsilon):
            newAction = random.choice((0,1))
        else:
            newAction = self.Pi[self.basis(state)]

        newState = state

        self.last_action = newAction
        self.last_state  = self.current_state
        self.current_state = newState

        if (self.last_state != None):
            s  = self.basis(self.last_state)
            sp = self.basis(self.current_state)
            a  = (self.last_action,)
            self.Np[s + a + sp] += 1
            self.N[s + a] += 1

        self.vIteration()
            
        return newAction

    #computes best action using Q function for next state
    def maxQ(self, state):
        v1 = self.Q[state,1]
        v2 = self.Q[state,0]
        return 1 if (v1 > v2) else 0

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if ((self.last_state != None) and (self.current_state != None) and (self.last_action != None)):
            s  = self.basis(self.last_state)
            a  = (self.last_action,)

            self.R[s + a] += self.last_reward
        
        self.last_reward = reward

    def getBucket(self, value, range, buckets):
        #Kobe
        bucketSize = (range[1] - range[0]) / buckets
        return math.floor((value - range[0]) / bucketSize)

    def basis_dimensions(self):
        return (self.tTopBuckets, self.tDistBuckets, self.mTopBuckets)

    def basis(self, state):
        top = state["tree"]["top"]
        dist = state["tree"]["dist"]

        treeBucket = self.getBucket(state["tree"]["top"],self.tTopRange,self.tTopBuckets)
        treeDist = self.getBucket(state["tree"]["dist"],self.tDistRange,self.tDistBuckets)
        monkeyBucket = self.getBucket(state["monkey"]["top"],self.mTopRange,self.mTopBuckets)

        return (treeBucket, treeDist, monkeyBucket)

    def solveV(self, pi):
        reward = np.array([1])
        for s in np.ndindex(np.shape(self.Pi)):
            np.append(reward, self.R[s + (self.Pi[s],)])

        A = np.insert(np.insert(self.transition_matrix(),0,0,axis=0),0,reward,axis=1)
        B = np.insert(self.V,0,1)
        self.V = np.linalg.solve(A,B)[1:]

    def transition_matrix(self):
        transition = []
        for s in np.ndindex(np.shape(self.Pi)):
            transition.append([(self.Np[s + (self.Pi[s],) + (Ellipsis,)] / self.N[(Ellipsis,) + (self.Pi[s],)]).flatten()])
        transition = np.concatenate(tuple(transition),axis=0)
        return transition

    def vIteration(self):
        #compute new value iteration value and update
        while True:
            duplicateV = np.copy(self.V)

            for s in np.ndindex(np.shape(self.Pi)):
                eValues = np.array([ np.dot( (self.Np[ s + a + (Ellipsis,) ] / self.N[(Ellipsis,) + a]).flat, duplicateV.flat) for a in [(0,), (1,)] ])
                self.Pi[s] = np.argmax(self.R[s + (Ellipsis,)] + self.gamma * eValues)
                self.V[s] = np.max(self.R[s + (Ellipsis,)] + self.gamma * eValues)

            if (np.isclose(self.V, duplicateV, atol=0.1, rtol=0.0).all()):
                break

    def pIteration(self):
        #Compute New Policy Iteration Value and update
        while True:
            Pi_copy = np.copy(self.Pi)

            self.solveV(Pi_copy)

            for s in np.ndindex(np.shape(self.Pi)):
                for a in [(0,),(1,)]:
                    self.Q[s + a] = self.R[s + a] + self.gamma * np.dot((self.Np[s + a + (Ellipsis,)] / self.N[(Ellipsis,) + a]).flat, self.V.flat)
                self.Pi[s] = np.argmax(self.Q[s])

            if (np.array_equal(self.Pi, Pi_copy)):
                break

def evaluate(gamma=0.4, iters=100):

    learner = ModelBasedLearner()
    learner.gamma = gamma

    highscore = 0
    avgscore = 0.0

    print "epoch", "\t", "score", "\t", "high", "\t", "avg"

    for ii in xrange(iters):

        learner.epsilon = 1/(ii+1)

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,            # Don't play sounds.
                             text="Epoch %d" % (ii), # Display the epoch on screen.
                             tick_length=1,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        score = swing.get_state()['score']
        highscore = max([highscore, score])
        avgscore = (ii*avgscore+score)/(ii+1)

        print ii, "\t", score, "\t", highscore, "\t", avgscore

        # Reset the state of the learner.
        learner.reset()

    return -avgscore


def find_hyperparameters():
    #optimize hyperparameters
    maxValue = 0
    maxParams = (0,0)
    for gamma in np.arange(0.1,1,0.1):
        parameters = {"gamma": gamma}
        value = evaluate(**parameters)
        if (value < maxValue):
            maxParams = parameters
            print "Best: ",parameters, " : ", value

    return maxParams

evaluate(iters=1000,gamma=0.4)