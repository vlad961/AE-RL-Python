from models.policy import Policy
import numpy as np
import tensorflow as tf
import sys

class Policy:
    def __init__(self, num_actions, estimator):
        self.num_actions = num_actions
        self.estimator = estimator

class EpsilonGreedy(Policy):
    def __init__(self, estimator, num_actions, epsilon, min_epsilon, decay_rate, epoch_length):
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
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions, states.shape[0])
        else:
            # Convert states to Tensor before prediction##
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            # Get Q values
            ######
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