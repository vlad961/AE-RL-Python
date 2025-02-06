import numpy as np
from models.q_network import QNetwork
from models.replay_memory import ReplayMemory
from models.epsilon_greedy import EpsilonGreedy


class Agent(object):
    """
        Reinforcement learning Agent definition
    """
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        self.done = None
        self.rewards = None
        self.next_states = None
        self.states = None
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size

        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate', 0.99)
        self.ExpRep = kwargs.get('ExpRep', True)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))

        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time

        self.model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2),
                                      kwargs.get('model_name', 'model'))
        self.target_model_network = QNetwork(self.obs_size, self.num_actions,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2),
                                             kwargs.get('target_model_name', 'target_model'))
        self.target_model_network.model = QNetwork.copy_model(self.model_network.model)

        if policy == "EpsilonGreedy":
            self.policy = EpsilonGreedy(self.model_network, len(actions),
                                         self.epsilon, self.min_epsilon,
                                         self.decay_rate, self.epoch_length)

    def learn(self, states, actions, next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done

    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done

        next_actions = []
        # Compute Q targets
        # Q_prime = self.model_network.predict(next_states,self.minibatch_size)
        Q_prime = self.target_model_network.predict(next_states, self.minibatch_size)
        # TODO: fix performance in this loop
        for row in range(Q_prime.shape[0]):
            best_next_actions = np.argwhere(Q_prime[row] == np.amax(Q_prime[row]))
            next_actions.append(best_next_actions[np.random.choice(len(best_next_actions))].item())
        sx = np.arange(len(next_actions))
        # Compute Q(s,a)
        Q = self.model_network.predict(states, self.minibatch_size)
        # Q-learning update
        # target = reward + gamma * max_a'{Q(next_state,next_action)}
        targets = rewards.reshape(Q[sx, actions].shape) + \
                  self.gamma * Q[sx, next_actions] * \
                  (1 - done.reshape(Q[sx, actions].shape)) # if done (episode has ended), no update
        Q[sx, actions] = targets

        result = self.model_network.model.train_on_batch(states, Q)  # inputs,targets

        # timer to ddqn update
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            #            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
            self.target_model_network.model.set_weights(self.model_network.model.get_weights())

        return result

    def act(self, state): # NOTE: In comparison to original code, the policy parameter was deleted since it is already defined in the constructor 
        raise NotImplementedError