from collections import namedtuple

import numpy
import numpy as np

from board import Board, Move
from search import SearchProblem, ucs, a_star_search, dfs
import util
from typing import List


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.targets = [(0, 0), (board_w - 1, 0), (board_w - 1, board_h - 1), (0, board_h - 1)]

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state: Board):
        "*** YOUR CODE HERE ***"
        return all([state.get_position(pos[0], pos[1]) != -1 for pos in self.targets])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions: List[Move]):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        if not actions:
            return 0
        actionsp = np.array(actions)
        cost = lambda a: a.piece.get_num_tiles()
        vfunc = numpy.vectorize(cost)
        return numpy.sum(vfunc(actionsp))


def chebyshev_distance(xy1, xy2):
    "Returns the chebyshev distance between points xy1 and xy2"
    return max(abs(xy1[0] - xy2[0]), abs(xy1[1] - xy2[1]))


def blokus_corners_heuristic(state: Board, problem: BlokusCornersProblem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    new_targets = [(t[1], t[0]) for t in problem.targets]
    return blokus_heuristic_template(state, new_targets)


def blokus_heuristic_template(state, to_reach: List):
    min_reach = np.array([float('inf') for i in to_reach])
    flags = [False for i in to_reach]
    board = state.state
    for xy, element in np.ndenumerate(board):
        if element != -1:
            distances = check_distances_from_points(xy, to_reach, flags)
            min_reach = np.minimum(min_reach, distances)
    for i, flag in enumerate(flags):
        if flag and min_reach[i] != 0:  # Check for false alarm, if the goal is occupied everything is good
            return float('inf')
    return np.sum(min_reach)


def check_distances_from_points(xy, points: List, flags) -> np.ndarray:
    """
    caclulates the distance between a point xy to all points
    @param xy:
    @param points:
    @return: ndarray with the distance of the point from all the rest of the points
    """
    distances = []
    for i, point in enumerate(points):
        che_dist = chebyshev_distance(xy, point)
        man_dist = util.manhattanDistance(xy, point)
        if man_dist == 1:
            flags[i] = True
        distances.append(che_dist)
    return np.array(distances)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)], checks=[(0, 0)], index=0):
        self.targets = targets.copy()
        self.expanded = 0
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.checks = checks
        self.index = index
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return all([state.get_position(pos[1], pos[0]) != -1 for pos in self.targets])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        if not actions:
            return 0
        cost = lambda a: a.piece.get_num_tiles()
        vfunc = numpy.vectorize(cost)
        return numpy.sum(vfunc(actions))


def blokus_cover_heuristic(state: Board, problem: BlokusCoverProblem):
    "*** YOUR CODE HERE ***"
    return blokus_heuristic_template(state, problem.targets)


def simplified_heuristic(state: Board, problem: BlokusCoverProblem):
    checks = problem.checks
    index = problem.index
    flags = [False for i in checks]
    min_reach = np.array([float('inf') for i in checks])
    board = state.state
    for xy, element in np.ndenumerate(board):
        if element != -1:
            distances = check_distances_from_points_simplified(xy, checks, flags)
            min_reach = np.minimum(min_reach, distances)
    for i, flag in enumerate(flags):
        if flag and min_reach[i] != 0:  # Check for false alarm, if the goal is occupied everything is good
            return float('inf')
    return min_reach[index]


def check_distances_from_points_simplified(xy, checks: List, flags) -> np.ndarray:
    """
    caclulates the distance between a point xy to all points
    @param xy:
    @param checks:
    @return: ndarray with the distance of the point from all the rest of the points
    """
    distances = []
    for i, point in enumerate(checks):
        man_dist = util.manhattanDistance(xy, point)
        if man_dist == 1:
            flags[i] = True
        distances.append(man_dist)
    return np.array(distances)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = np.array(targets.copy())
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def closes_point(self):
        closet_point = np.argmin(np.sum(np.abs(self.targets - np.array(self.starting_point)[None, :]), axis=1))
        return self.targets[closet_point], closet_point

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        current_state = self.board.__copy__()
        backtrace = []
        reached = [False for target in self.targets]
        closest_point, index = self.closes_point()
        problem = BlokusCoverProblem(current_state.board_w, current_state.board_h, current_state.piece_list,
                                     self.starting_point, [closest_point], self.targets.tolist(), index)
        while not all(reached):
            actions = a_star_search(problem, simplified_heuristic)
            reached[index] = True
            backtrace.extend(actions)
            self.starting_point = np.copy(self.targets[index])
            self.targets[index] = np.array([-1000, -1000])
            for move in actions:
                current_state = current_state.do_move(0, move)
            closest_point, index = self.closes_point()
            problem.board = current_state
            problem.targets = [closest_point]
            problem.index = index
        self.expanded = problem.expanded
        return backtrace


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
