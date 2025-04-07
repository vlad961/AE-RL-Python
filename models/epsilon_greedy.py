from typing import List
from models.policy import Policy
import numpy as np
import tensorflow as tf
import sys

from models.q_network import QNetwork
from utils.config import GLOBAL_RNG

class EpsilonGreedy(Policy):
    """
    Implements the Epsilon-Greedy policy for reinforcement learning.

    The Epsilon-Greedy policy selects a random action (exploration) with probability epsilon and the best action (exploitation)
    (based on the Q-values predicted by the estimator) with probability 1 - epsilon. The epsilon value
    decays over time to balance exploration and exploitation.

    """
    def __init__(self, estimator: QNetwork, actions, epsilon: float, min_epsilon: float, decay_rate: float, epoch_length: int, parent_agent):
        super().__init__(len(actions), estimator)
        self.name = "Epsilon Greedy"
        self.parent_agent = parent_agent

        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush=True)
            sys.exit(0)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.actions = actions
        self.step_counter = 0
        self.num_actions = len(actions)
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate

        # if epsilon is up 0.1, it will be decayed over time
        if self.epsilon > 0.01:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False

    def get_actions(self, states) -> List[int]:
        """
        Selects actions based on the Epsilon-Greedy policy.
        In epsilon percent of the cases, a random action is selected. 
        In the remaining cases, the action with the highest Q-value is selected.
        Epsilon is decayed over time to balance exploration and exploitation.

        Args:
            states (np.ndarray): The current states.

        Returns:
            list: The selected actions.
        """
        # get next action
        dice = GLOBAL_RNG.random() # returns a random float between 0 and 1
        if dice <= self.epsilon:
            actions: List[int] = GLOBAL_RNG.choice(list(self.actions), states.shape[0]).tolist() # Choose a random value from self.actions (possible action IDs for the agent)
        else:
            # Get Q values
            ######
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            self.Q = self.estimator.predict(states_tensor, states_tensor.shape[0])  #states.shape[0]
            ##### inserted in predict method
            actions: List[int] = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row])).astype(np.int16)
                best_action_index = best_actions[GLOBAL_RNG.choice(len(best_actions))].item()
                actions.append(list(self.actions)[best_action_index])

        self.step_counter += 1
        # decay epsilon after each episode
        if self.epsilon_decay:
            decay_interval = self.epoch_length * 4 if self.parent_agent.name == "Defender" else self.epoch_length
            if self.parent_agent.name == "Defender":
                decay_step = self.step_counter // 4 # Reduce the step_counter by 4 for the defender, as it takes 4 actions in one step. Otherwise, the epsilon would decay too fast for the defender.
            else:
                decay_step = self.step_counter # Attacker takes only one action in one step, so decay_step is equal to step_counter
            if self.step_counter % decay_interval == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate ** decay_step)

        return actions