import math
from collections import namedtuple

import numpy as np
import abc
import util
from enum import Enum
from game import Agent, Action
from game_state import GameState
from typing import Tuple


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action: Action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        # score = successor_game_state.score
        score = 0
        if action == Action.UP:
            return 0
        # region Check for descending order in the top left row (with max_tile at the left corner)
        j = board.shape[0] - 1
        prev_tile = board[j][0]
        if board[j][0] == max_tile:
            score += max_tile
            for tile in board[j, 1:]:
                if prev_tile < tile:
                    break
                score += tile
                prev_tile = tile
        score += np.array([(i + 1) * np.sum(board[i, :]) for i in
                           range(board.shape[0])]).sum()  # Pyramid like - give more power for the upper rows

        "*** YOUR CODE HERE ***"
        return score


def score_evaluation_function(current_game_state: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state: GameState):
        return


class Agent(Enum):
    Player = 0
    Computer = 1


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        minimax = self.minimax(game_state, self.depth, Agent.Player)
        return minimax[1]

    def minimax(self, game_state: GameState, depth: int, agent: Agent) -> Tuple[int, Action]:
        # region if 𝑑𝑒𝑝𝑡ℎ = 0 or v is a terminal node then return 𝑢(𝑣)
        if depth == 0 or not game_state.get_legal_actions(0):
            return self.evaluation_function(game_state), Action.STOP
        # endregion

        costume_key = lambda x: x[0]

        # region  if isMaxNode then return max
        if agent == Agent.Player:
            legal_moves = game_state.get_legal_actions(agent.value)
            max_val = (float("-inf"), Action.STOP)
            for move in legal_moves:
                new_state = game_state.generate_successor(agent.value, move)
                response_val = self.minimax(new_state, depth - 1, Agent.Computer)[0], move
                max_val = max(max_val, response_val, key=costume_key)
            return max_val

        # endregion

        # region  if isMinNode then return min
        if agent == Agent.Computer:
            legal_moves = game_state.get_legal_actions(agent.value)
            min_val = (float("inf"), Action.STOP)
            for move in legal_moves:
                new_state = game_state.generate_successor(agent.value, move)
                response_val = self.minimax(new_state, depth, Agent.Player)[0], move
                min_val = min(min_val, response_val, key=costume_key)
            return min_val
        # endregion


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        alpha_beta = self.alpha_beta(game_state, Agent.Player, self.depth)
        return alpha_beta[1]

    def alpha_beta(self, game_state: GameState, agent: Agent, depth: int, alpha=float("-inf"), beta=float("inf")) -> \
            Tuple[int, Action]:
        # region End Condition
        if depth == 0 or not game_state.get_legal_actions(0):
            return self.evaluation_function(game_state), Action.STOP
        # endregion

        costume_key = lambda x: x[0]

        # region alpha pruning
        if agent == Agent.Player:
            legal_moves = game_state.get_legal_actions(agent.value)
            return_alpha = (alpha, Action.STOP)
            for move in legal_moves:
                new_state = game_state.generate_successor(agent.value, move)
                alpha = return_alpha[0]
                response_val = self.alpha_beta(new_state, Agent.Computer, depth - 1, alpha, beta)[0], move
                return_alpha = max(return_alpha, response_val, key=costume_key)
                if return_alpha[0] >= beta:
                    break
            return return_alpha
        # endregion

        # region beta pruning
        if agent == Agent.Computer:
            legal_moves = game_state.get_legal_actions(agent.value)
            return_beta = (beta, Action.STOP)
            for move in legal_moves:
                new_state = game_state.generate_successor(agent.value, move)
                beta = return_beta[0]
                response_val = self.alpha_beta(new_state, Agent.Player, depth, alpha, beta)[0], move
                return_beta = min(return_beta, response_val, key=costume_key)
                if alpha >= return_beta[0]:
                    break
            return return_beta
        # endregion


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        expectimax = self.expctimax(game_state, self.depth, Agent.Player)
        return expectimax[1]

    def expctimax(self, game_state: GameState, depth: int, agent: Agent):
        # region End Condition
        if depth == 0 or not game_state.get_legal_actions(0):
            return self.evaluation_function(game_state), Action.STOP
        # endregion

        costume_key = lambda x: x[0]

        # region Expected Max
        if agent == Agent.Player:
            legal_moves = game_state.get_legal_actions(agent.value)
            max_val = (float("-inf"), Action.STOP)
            for move in legal_moves:
                new_state = game_state.generate_successor(agent.value, move)
                response_val = self.expctimax(new_state, depth - 1, Agent.Computer)[0], move
                max_val = max(max_val, response_val, key=costume_key)
            return max_val

        # endregion

        # region Expected Min
        if agent == Agent.Computer:
            legal_moves = game_state.get_legal_actions(agent.value)
            succesors = []
            for move in legal_moves:
                succesors.append(game_state.generate_successor(agent.value, move))
            succesors = np.array(succesors)
            probability_s = 1 / len(succesors)
            vfunc_expectimax = np.vectorize(self.expctimax)
            responses = vfunc_expectimax(succesors, depth, agent.Player)
            expectation = np.sum(responses[0] * probability_s), Action.STOP
            return expectation

        # endregion
        return


def better_evaluation_function(current_game_state: GameState):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successor_game_state = current_game_state
    board = successor_game_state.board
    score = 0
    max_tile = successor_game_state.max_tile
    features = []
    DEFAULT_WEIGHT = 10
    Feature = namedtuple("Feature", "name weight")

    # region Snake Board
    snake = np.array([[1, 2, 4, 8], [128, 64, 32, 16], [256, 512, 1024, 2048], [32768, 16384, 8192, 4096]])
    snake = snake / 1024
    score += np.sum(board * snake)
    # endregion

    # region Edge Cases
    moves = current_game_state.get_agent_legal_actions()
    if len(moves) == 1 and moves[0] == Action.UP:
        return float("-inf")
    # endregion

    # region Penalty for max block not being at the corner
    j = board.shape[0] - 1
    if board[j][0] != max_tile:
        features.append(Feature(board[j][0], -1000))
    # endregion

    # region monotonic bottom row
    num_of_descending_bottom_row = 0
    prev_tile = board[j][0]
    for tile in board[j, 1:]:
        if prev_tile < tile:
            break
        num_of_descending_bottom_row += tile
        prev_tile = tile

    features.append(Feature(num_of_descending_bottom_row, 2))
    # endregion

    # region Empty tiles

    empty_tiles = len(current_game_state.get_empty_tiles()[0])

    features.append(Feature(empty_tiles, 2))
    features.append(Feature(perfect_num_of_tiles_score(empty_tiles), 10))

    # endregion

    # region pyramid-like rows, best sum row should be at the bottom
    row_sums = np.sum(board, axis=1)
    for i, row in enumerate(row_sums):
        features.append(Feature(row, i + 1))
    # endregion

    # region merge-ability of a certain board
    ICKY_VAL = 4496
    board = current_game_state.board
    board_dim = board.shape[0]
    adjacent_cols = np.abs(board[:, :board_dim - 1] - board[:, 1:])  # left to right
    adjacent_rows = np.abs(board[:board_dim - 1, :] - board[1:, :])  # up to down
    # the smaller the adjacents_sum - the better our situation 🙂
    # todo: find a way to properly weight this sum.. This seems to work if our
    #  sum is (1 <= sum <= 4496), it gives a number from 0 to 100, depending on
    #  how close we are to a good matrix. Not perfect, but good enough
    adjacent_rows = np.sum(adjacent_rows)
    adjacent_cols = np.sum(adjacent_cols)
    if adjacent_rows != 0:
        ratio = (np.log(adjacent_rows) / np.log(ICKY_VAL))
        features.append(Feature(ratio, 30))
    if adjacent_cols != 0:
        ratio = (np.log(adjacent_cols) / np.log(ICKY_VAL))
        features.append(Feature(ratio, 10))


    # endregion

    row = board[j, 1:]
    num_of_merges_max_tile = math.pow(2, math.log(max_tile, 2) - 1) - 1
    two = np.ones(row.shape).astype(int) * 2
    num_of_merges_row = np.sum(np.power(two, np.log2(row) - 1) - 1)
    if (num_of_merges_max_tile - num_of_merges_row) > 0:
        new = 1/math.ceil(num_of_merges_max_tile - num_of_merges_row)
        features.append(Feature(new, 10))

    for feature in features:
        score += feature.name * feature.weight
    return score


def perfect_num_of_tiles_score(real_amount, min_amount=1, best_amount=7, max_amount=16):
    ratio = 100 / (best_amount - min_amount)
    if (real_amount <= best_amount):
        return ratio * (real_amount - 1)
    else:
        rest = real_amount - best_amount
        return 100 - (ratio / 2) * rest


def test_evaluation(game_state: GameState):
    successor_game_state = game_state
    board = successor_game_state.board
    score = 0
    max_tile = successor_game_state.max_tile
    # if max_tile != board[board.shape[0] - 1, 0]:
    #     score -= 10000
    for i, row in enumerate(board):
        score += 1 * len(game_state.get_empty_tiles())
        score += 10 * merges_in_row(row)
        score -= 2 * min(num_of_monotonic(row), num_of_monotonic(row[::-1]))
        score += 5 * np.sum(row)

    board = board.T
    for i, row in enumerate(board):
        score += 1 * len(game_state.get_empty_tiles())
        score += 10 * merges_in_row(row)
        score -= 2 * min(num_of_monotonic(row), num_of_monotonic(row[::-1]))
        score += 20 * np.sum(row)
    return score


def merges_in_row(row: np.ndarray):
    return len(np.where((row[:-1] - row[1:]) == 0)[0])


def num_of_monotonic(row: np.ndarray):
    score = 0
    prev_tile = row[0]
    for tile in row[1:]:
        if prev_tile < tile:
            break
        score += 1
        prev_tile = tile
    return score


# Abbreviation
better = better_evaluation_function
# better = test_evaluation
