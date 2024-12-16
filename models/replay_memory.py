import numpy as np

class ReplayMemory(object):
    """
    Implements basic replay memory for reinforcement learning.

    This class stores experiences (state, action, reward, terminal) and allows sampling of minibatches
    for training reinforcement learning agents. It helps in breaking the correlation between consecutive
    experiences by randomly sampling from the memory.

    Attributes:
        observation_size (int): The size of the observation space.
        num_observed (int): The number of observed experiences.
        max_size (int): The maximum size of the replay memory.
        samples (dict): A dictionary to store observations, actions, rewards, and terminal flags.
    """
    
    def __init__(self, observation_size, max_size):
        """
        Initializes the replay memory.

        Args:
            observation_size (int): The size of the observation space.
            max_size (int): The maximum size of the replay memory.
        """
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size

        # Debugging-Ausgaben
        print(f"Initializing ReplayMemory with observation_size={self.observation_size} and max_size={self.max_size}")

        if self.max_size <= 0 or self.observation_size <= 0:
            raise ValueError("max_size and observation_size must be positive and greater than zero")
        
        self.samples = {
            'obs': np.zeros(self.max_size * 1 * self.observation_size,
                            dtype=np.float32).reshape(self.max_size, self.observation_size),
            'action': np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
            'reward': np.zeros(self.max_size * 1).reshape(self.max_size, 1),
            'terminal': np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
        }

    def observe(self, state, action, reward, done):
        """
        Stores a new experience in the replay memory.

        Args:
            state (np.ndarray): The observed state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode has ended.
        """
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        """
        Samples a minibatch of experiences from the replay memory.

        Args:
            minibatch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next states, and terminal flags.
        """
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices + 1, :], dtype=np.float32)

        a = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)