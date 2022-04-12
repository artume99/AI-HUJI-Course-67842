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
        print(alpha_beta)
        return alpha_beta[1]

    def alpha_beta(self, game_state: GameState, agent: Agent, depth: int, alpha=float("-inf"), beta=float("inf")) -> \
            Tuple[int, Action]:
        # region if 𝑑𝑒𝑝𝑡ℎ = 0 or v is a terminal node then return 𝑢(𝑣)
        if depth == 0 or not game_state.get_legal_actions(0):
            return self.evaluation_function(game_state), Action.STOP
        # endregion

        costume_key = lambda x: x[0]

        # region if isMaxNode then a = max(𝛼, alpha_beta( 𝑠, 𝑑𝑒𝑝𝑡ℎ − 1, 𝛼, 𝛽, 𝑓𝑎𝑙𝑠𝑒)), prune if a>=b
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

        # region if isMinNode then b = min((𝛽, alpha_beta( 𝑠, 𝑑𝑒𝑝𝑡ℎ − 1, 𝛼, 𝛽, 𝑡𝑟𝑢𝑒))), prune if a>=b
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
        util.raiseNotDefined()


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
