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

    Attributes:
        estimator (object): The Q-value estimator.
        num_actions (int): The number of possible actions.
        epsilon (float): The probability of selecting a random action.
        min_epsilon (float): The minimum value of epsilon after decay.
        decay_rate (float): The rate at which epsilon decays.
        epoch_length (int): The number of steps in one epoch.
        step_counter (int): The counter for the number of steps taken.
        epsilon_decay (bool): Whether epsilon should decay over time.
    """
    def __init__(self, estimator: QNetwork, num_actions, epsilon, min_epsilon, decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator)
        self.name = "Epsilon Greedy"

        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush=True)
            sys.exit(0)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.actions = list(range(num_actions))
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate

        # if epsilon is up 0.1, it will be decayed over time
        if self.epsilon > 0.01:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False

    def get_actions(self, states):
        """
        Selects actions based on the Epsilon-Greedy policy.

        Args:
            states (np.ndarray): The current states.

        Returns:
            list: The selected actions.
        """
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions, states.shape[0])
        else:
            # Get Q values
            ######
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            self.Q = self.estimator.predict(states_tensor, states.shape[0])
            ##### inserted in predict method
            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                actions.append(best_actions[np.random.choice(len(best_actions))].item())

        self.step_counter += 1
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate ** self.step_counter)

        return actions