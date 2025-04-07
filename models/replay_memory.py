import logging
import sys
from typing import List

import pandas as pd
import numpy as np
from utils.config import GLOBAL_RNG

class ReplayMemory(object):
    """
    Implements basic replay memory for reinforcement learning.

    This class stores experiences (state, action, reward, terminal) and allows sampling of minibatches
    for training reinforcement learning agents. It helps in breaking the correlation between consecutive
    experiences by randomly sampling from the memory. Random sampling is crucial for training stability.
    It also allows for the storage of experiences in a circular buffer manner, ensuring that the oldest 
    samples are replaced when the memory is full.

    Attributes:
        observation_size (int): The size of the observation space.
        num_observed (int): The number of observed experiences.
        max_size (int): The maximum size of the replay memory.
        samples (dict): A dictionary to store observations, actions, rewards, and terminal flags.
    """
    
    def __init__(self, observation_size: int, max_size: int, parent_agent):
        """
        Initializes the replay memory.

        Args:
            observation_size (int): The size of the observation space.
            max_size (int): The maximum size of the replay memory.
        """
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.parent_agent = parent_agent
        # track the indices of the samples for a deeper analysis
        self.tracked_indices = []

        # Logging
        logging.info(f"Initializing {parent_agent.name}'s ReplayMemory with observation_size={self.observation_size} and max_size={self.max_size}")

        if self.max_size <= 0 or self.observation_size <= 0:
            raise ValueError("max_size and observation_size must be positive and greater than zero")
        
        self.samples = {
            'obs': [None] * self.max_size,
            'action':  np.zeros((self.max_size, 1), dtype=np.int16),
            'reward': np.zeros((self.max_size, 1), dtype=np.int32),
            'terminal': np.zeros((self.max_size, 1), dtype=np.bool_),
        }


    def observe(self, state: np.ndarray, action: List[int], reward: np.int32, done: bool):
        """
        Stores a new experience in the replay memory.
        """
        # Type checks
        assert isinstance(state, pd.DataFrame), "state must be a pandas DataFrame"
        assert isinstance(action, List), "action must be an integer"
        assert isinstance(reward, np.int32), "reward must be a float or np.float32"
        assert isinstance(done, np.ndarray), "done must be a boolean"

        index = self.num_observed % self.max_size
        self.samples['obs'][index] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, 0] = reward
        self.samples['terminal'][index, 0] = done

        # Track the index for analysis
        self.tracked_indices.append(state.index.tolist())

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
        sampled_indices = GLOBAL_RNG.integers(0, max_index + 1, size=minibatch_size)
        # Extract samples
        states = []
        states_indices = [] # actual indices of the nsl-kdd dataset
        next_states = []
        next_state_indices = [] # actual indices of the nsl-kdd dataset
        for i in sampled_indices:
            if self.samples['obs'][i] is None:
                logging.error(f"Sampled index {i} in 'obs' is None.")
                sys.exit(0)
            states_indices.append(self.samples['obs'][i].index.tolist())
            states.append(self.samples['obs'][i].to_numpy(dtype=np.float32)[0])

            next_index = (i + 1) % max_index
            if self.samples['obs'][next_index] is None:
                logging.error(f"Next index {next_index} in 'obs' is None.")
            next_states.append(self.samples['obs'][next_index].to_numpy(dtype=np.float32)[0])
            next_state_indices.append(self.samples['obs'][next_index].index.tolist())



        states = np.array(states)#np.array([self.samples['obs'][i].to_numpy(dtype=np.float32)[0] for i in sampled_indices]) #np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        next_states = np.array(next_states)
        actions = self.samples['action'][sampled_indices, 0]#self.samples['action'][sampled_indices].reshape(minibatch_size)
        rewards = self.samples['reward'][sampled_indices, 0] #self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done = self.samples['terminal'][sampled_indices, 0]#self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))
        # Extract indices for analysis
        indices = [states_indices, next_state_indices]

        return (states, actions, rewards, next_states, done, indices)