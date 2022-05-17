"""
In search.py, you will implement generic search algorithms
"""

import util

PATH = 1

STATE = 0
ACTION = 1
COST = 2


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state
STATE
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.
    """

    fringe = util.Stack()
    current_node = (problem.get_start_state(), [])

    visited = set()
    fringe.push(current_node)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.is_goal_state(current_node[STATE]):
            return current_node[PATH]
        elif current_node[STATE] not in visited:
            neighbors = problem.get_successors(current_node[STATE])
            for neighbor in neighbors:
                fringe.push((neighbor[STATE], current_node[PATH]+[neighbor[ACTION]]))
            visited.add(current_node[STATE])
    return []


class Node:

    def __init__(self, state, path, cost, scores=[]):
        self.state = state
        self.path = path
        self.cost = cost
        self.scores = scores


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    fringe = util.Queue()
    current_node = (problem.get_start_state(), [])

    visited = set()
    fringe.push(current_node)

    while not fringe.isEmpty():
        current_node = fringe.pop()

        if problem.is_goal_state(current_node[STATE]):
            return current_node[PATH]

        elif current_node[STATE] not in visited:
            neighbors = problem.get_successors(current_node[STATE])

            for neighbor in neighbors:
                fringe.push((neighbor[STATE], current_node[PATH] + [neighbor[ACTION]]))
            visited.add(current_node[STATE])
    return []


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    start_state = problem.get_start_state()
    current = Node(start_state, [], 0)
    fringe = util.PriorityQueue()
    visited = set()
    fringe.push(current, current.cost)
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.is_goal_state(current.state):
            return current.path
        elif current.state not in visited:
            neighbors = problem.get_successors(current.state)
            for triplet in neighbors:
                neighbor = Node(triplet[STATE], current.path + [triplet[ACTION]], current.cost + triplet[COST])
                fringe.push(neighbor, neighbor.cost)
            visited.add(current.state)
    return []


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    start_state = problem.get_start_state()
    current = Node(start_state, [], heuristic(start_state, problem))
    fringe = util.PriorityQueue()
    visited = set()
    fringe.push(current, current.cost)
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.is_goal_state(current.state):
            # for action in current.path:
            #     print(action)
            return current.path
        elif current.state not in visited:
            neighbors = problem.get_successors(current.state)
            for triplet in neighbors:
                neighbor = Node(triplet[STATE], current.path + [triplet[ACTION]], current.cost + triplet[COST])
                fringe.push(neighbor, neighbor.cost + heuristic(neighbor.state, problem))
            visited.add(current.state)
    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search