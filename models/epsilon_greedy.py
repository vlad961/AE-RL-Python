from typing import List
from models.policy import Policy
import numpy as np
import tensorflow as tf
import sys

from models.q_network import QNetwork

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
        dice = np.random.rand() # returns a random float between 0 and 1
        if dice <= self.epsilon:
            actions = np.random.choice(list(self.actions), states.shape[0]).tolist() # Choose a random value from self.actions (possible action IDs for the agent)
        else:
            # Get Q values
            ######
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            self.Q = self.estimator.predict(states_tensor, states.shape[0])
            ##### inserted in predict method
            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                best_action_index = best_actions[np.random.choice(len(best_actions))].item()
                actions.append(np.array(list(self.actions)[best_action_index]))

        self.step_counter += 1 # //TODO: rework logic, as now multiple actions will be taken by defender in one step, so step_counter will be incremented multiple times --> make decay rate multiple of 4
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.parent_agent.name == "Defender" and self.step_counter % (self.epoch_length * 4) == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate ** self.step_counter)
            elif self.parent_agent.name != "Defender" and self.step_counter % self.epoch_length == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate ** self.step_counter)

        return actions # FIXME: Return Type für beide Fälle abchecken und sicherstellen, dass es eine Liste ist von int oder np.int 