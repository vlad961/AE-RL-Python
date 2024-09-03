import numpy as np

class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
            'obs': np.zeros(self.max_size * 1 * self.observation_size,
                            dtype=np.float32).reshape(self.max_size, self.observation_size),
            'action': np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
            'reward': np.zeros(self.max_size * 1).reshape(self.max_size, 1),
            'terminal': np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
        }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices + 1, :], dtype=np.float32)

        a = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)