import numpy as np
from models.q_network import QNetwork
from models.replay_memory import ReplayMemory
from models.epsilon_greedy import EpsilonGreedy
from utils.config import GLOBAL_RNG

class Agent(object):
    """
        Reinforcement learning Agent definition
    """
    def __init__(self, actions, obs_size, parent_agent, policy="EpsilonGreedy", **kwargs):
        self.done = None
        self.rewards = None
        self.next_states = None
        self.states = None
        self.actions = actions
        self.num_actions = len(actions)
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.obs_size = obs_size
        self.parent_agent = parent_agent
        self.name = self.parent_agent.name

        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate', 0.99)
        self.experience_replay = kwargs.get('ExpRep', True)
        if self.experience_replay:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10), self)

        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time

        self.model_network: QNetwork = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2),
                                      kwargs.get('model_name', 'model'))
        self.target_model_network: QNetwork = QNetwork(self.obs_size, self.num_actions,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2),
                                             kwargs.get('target_model_name', 'target_model'))
        self.target_model_network.model = QNetwork.copy_model(self.model_network.model)

        if policy == "EpsilonGreedy":
            self.policy = EpsilonGreedy(self.model_network, actions,
                                         self.epsilon, self.min_epsilon,
                                         self.decay_rate, self.epoch_length, self)

    def learn(self, states, actions, next_states, rewards, done):
        """
        Store the experience(states) in the replay memory if experience replay is enabled.
        Otherwise, store the experience in the agent's attributes.
        """
        if self.experience_replay:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done

    def update_model(self):
        if self.experience_replay:
            (states, actions, rewards, next_states, done, indices) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states.to_numpy(dtype=np.float32)
            rewards = self.rewards.to_numpy(dtype=np.float32)
            next_states = self.next_states.to_numpy(dtype=np.float32)
            actions = self.actions.to_numpy(dtype=np.int32)
            done = self.done.to_numpy(dtype=np.bool_)

        next_actions = []
        # Compute Q targets
        Q_prime = self.target_model_network.predict(next_states, self.minibatch_size)
        # TODO: fix performance in this loop
        for row in range(Q_prime.shape[0]):
            best_next_actions = np.argwhere(Q_prime[row] == np.amax(Q_prime[row]))
            next_actions.append(best_next_actions[GLOBAL_RNG.choice(len(best_next_actions))].item())
        sx = np.arange(len(next_actions))
        # Compute Q(s,a)
        # Map actions to indices of the Q values
        mapped_actions = np.array([self.action_to_index[action] for action in actions])
        Q = self.model_network.predict(states, self.minibatch_size)
        # Q-learning update
        # target = reward + gamma * max_a'{Q(next_state,next_action)}
        targets = rewards.reshape(Q[sx, mapped_actions].shape) + \
                  self.gamma * Q[sx, next_actions] * \
                  (1 - done.reshape(Q[sx, mapped_actions].shape)) # if done (episode has ended), no update
        Q[sx, mapped_actions] = targets

        result = self.model_network.model.train_on_batch(states, Q)  # inputs,targets
        mse_before_update = result[1]  # implicit (TensorFlow)
        mae_before_update = result[2]  # implicit (TensorFlow)
        # explicit calculation of MSE and MAE after update
        #mse_after_update = np.mean((Q - self.model_network.model.predict(states))**2)
        #mae_after_update = np.mean(np.abs(Q - self.model_network.model.predict(states)))
        loss = result[0]
        
        # timer to ddqn update
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            #            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
            self.target_model_network.model.set_weights(self.model_network.model.get_weights())

        return {"result": result, "loss": loss, "mse_before": mse_before_update, "mae_before": mae_before_update , "sample_indices": indices}

    def act(self, state): # NOTE: In comparison to original code, the policy parameter was deleted since it is already defined in the constructor 
        raise NotImplementedError