# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import numpy

import util
from mdp import MarkovDecisionProcess as MDP

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: MDP, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        for i in range(self.iterations):
            previous_values = self.values.copy()
            self.value_iteration(previous_values)

        print(self.values)

        "*** YOUR CODE HERE ***"

    def value_iteration(self, prev_values):
        reward = 0
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_v = 0
                for action in self.mdp.getPossibleActions(state):
                    sum_v = 0
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, next_state)
                        v = prob * prev_values[next_state]
                        sum_v += v
                    max_v = sum_v if sum_v > max_v else max_v
                self.values[state] = reward + self.discount * max_v

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        return sum([prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]) for
                    next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_val = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                val = prob * self.values[next_state]
                if best_val < val:
                    best_action = action
                    best_val = val
        return best_action

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
