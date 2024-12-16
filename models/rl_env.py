import numpy as np

from data.data_cls import DataCls
from models.attack_agent import AttackAgent



class RLenv(DataCls):
    def __init__(self, dataset_type, attack_agent, trainset_path, testset_path, formated_train_path, formated_test_path, **kwargs):
        self.true_labels = None
        self.attack_agent = attack_agent
        DataCls.__init__(self, trainset_path, testset_path, formated_train_path, formated_test_path, dataset_type=dataset_type)
        DataCls.load_formatted_df(self)
        self.data_shape = DataCls.get_shape(self)
        self.batch_size = kwargs.get('batch_size', 1)  # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode', 10)
        if self.batch_size == 'full':
            self.batch_size = int(self.data_shape[0] / self.iterations_episode)



    def _update_state(self):
        '''
        _update_state: function to update the current state
        Returns:
            None
        Modifies the self parameters involved in the state:
            self.state and self.labels
        Also modifies the true labels to get learning knowledge
        '''
        self.states, self.labels = DataCls.get_batch(self)

        # Update statistics
        self.true_labels += np.sum(self.labels).values

    def reset(self):
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_names), dtype=int)

        self.state_numb = 0

        DataCls.load_formatted_df(self)  # Reload and random index
        self.states, self.labels = DataCls.get_batch(self, self.batch_size)

        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values

    def act(self, defender_actions, attack_actions):
        # Clear previous rewards
        self.att_reward = np.zeros(len(attack_actions))
        self.def_reward = np.zeros(len(defender_actions))

        attack = [self.attack_types.index(self.attack_map[self.attack_names[att]]) for att in attack_actions]

        self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 1
        self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 1

        self.def_estimated_labels += np.bincount(defender_actions, minlength=len(self.attack_types))
        # TODO
        # list comprehension

        for act in attack_actions:
            self.def_true_labels[self.attack_types.index(self.attack_map[self.attack_names[act]])] += 1

        # Get new state and new true values
        # NOTE: the following two uncommented lines were in the original code, however everything was written in one file and the attacker_agent and env variables where defined in __name__ == '__main__' block after this declaration.
        #attack_actions = attacker_agent.act(self.states)
        #self.states = env.get_states(attack_actions) ORIGINAL
        attack_actions = self.attack_agent.act(self.states)
        self.states = self.get_states(attack_actions)

        # Done allways false in this continuous task
        self.done = np.zeros(len(attack_actions), dtype=bool)

        return self.states, self.def_reward, self.att_reward, attack_actions, self.done



    def get_states(self, attacker_actions):
        '''
        Provide the actual states for the selected attacker actions
        Parameters:
            self:
            attacker_actions: optimum attacks selected by the attacker
                it can be one of attack_names list and select random of this
        Returns:
            State: Actual state for the selected attacks
        '''
        first = True
        for attack in attacker_actions:
            if first:
                minibatch = (self.df[self.df[self.attack_names[attack]] == 1].sample(1))
                first = False
            else:
                minibatch = minibatch.append(self.df[self.df[self.attack_names[attack]] == 1].sample(1))

        self.labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names, axis=1, inplace=True)
        self.states = minibatch

        return self.states
