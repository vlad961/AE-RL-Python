import numpy as np
import os
import tensorflow as tf

from data.data_cls import DataCls
from models.attack_agent import AttackAgent

cwd = os.getcwd()
data_root_dir = os.path.join(cwd, "data/datasets/")
data_original_dir = os.path.join(data_root_dir, "origin-kaggle-com/nsl-kdd/")
data_formated_dir = os.path.join(data_root_dir, "formated/")
formated_train_path = os.path.join(data_formated_dir, "balanced_training_data.csv") # formated_train_adv.csv
formated_test_path = os.path.join(data_formated_dir, "balanced_test_data.csv") # formated_test_adv.csv
kdd_train = os.path.join(data_original_dir, "KDDTrain+.txt")
kdd_test = os.path.join(data_original_dir, "KDDTest+.txt")

class RLenv(DataCls):
    def __init__(self, dataset_type, attack_agent: AttackAgent, trainset_path=kdd_train, testset_path=kdd_test, formated_train_path=formated_train_path, formated_test_path=formated_test_path, **kwargs):
        self.true_labels = None
        self.attack_agent = attack_agent

        self.specific_attack_type = kwargs.get('specific_attack_type')
        data = kwargs.get('data')
        if self.specific_attack_type is None and data is None:
            DataCls.__init__(self, trainset_path, testset_path, formated_train_path, formated_test_path, dataset_type=dataset_type)
            DataCls.load_formatted_df(self)
        elif data is not None:
            self.df = data.df
            self.attack_names = kwargs.get('attack_names')
            self.attack_types = data.attack_types
            self.loaded = True
            self.index = data.index
            self.attack_map = data.attack_map
            self.all_attack_names = data.all_attack_names
        else:
            raise ValueError("If 'specific_attack' is provided, 'data' must also be provided.")


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
        """
        Reset the environment to the initial state.
        Calculates a random index to start the episode and loads the batch of data.
        Returns:
            State: The initial state.
        """
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_types), dtype=int) # We set the true labels of attack in the range of the attack types as the defender infers the attack types.

        DataCls.load_formatted_df(self)  # Reload and set a random index.
        self.states, self.labels = DataCls.get_batch(self, self.batch_size)
        self.total_reward = 0
        self.steps_in_episode = 0

        return self.states

    def act(self, defender_actions, attack_actions):
        # Clear previous rewards
        self.att_reward = np.zeros(len(attack_actions))
        self.def_reward = np.zeros(len(defender_actions))

        attack_names_mapped = [self.attack_names[att] for att in attack_actions]
        attack_types_mapped = [self.attack_map[attack_name] for attack_name in attack_names_mapped]
        # Get the indices of the attack types
        attack = [self.attack_types.index(attack_type) for attack_type in attack_types_mapped]

        self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 1
        self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 1

        self.def_estimated_labels += np.bincount(defender_actions, minlength=len(self.attack_types))
        # Update amount of att_true_labels using the indices from attack
        for idx in attack:
            self.att_true_labels[idx] += 1
        
        # Update def_true_labels using the indices from attack
        for def_action, att_action in zip(defender_actions, attack):
            if def_action == att_action:
                self.def_true_labels[def_action] += 1

        # Get new state and new true values
        attack_actions = self.attack_agent.act(self.states)
        self.states = self.get_states(attack_actions)

        # Done allways false in this continuous task
        self.done = np.zeros(len(attack_actions), dtype=bool)

        return self.states, self.def_reward, self.att_reward, attack_actions, self.done



    def get_states(self, attacker_actions):
        '''
        Provide the actual states for the selected attacker actions / chosen attacks by the attacker.

        A random instance is selected from the dataset for each selected attack class.
        Parameters:
            attacker_actions: optimum attacks selected by the attacker
                it can be one of attack_names list and select random of this
        Side effects:
            self.states: Actual states for the selected attacks
            self.labels: Actual labels for the selected attacks
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
