from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from data.data_cls import DataCls, attack_types
from models.attack_agent import AttackAgent

cwd = os.getcwd()
data_root_dir = os.path.join(cwd, "data/datasets/")
data_original_dir = os.path.join(data_root_dir, "origin-kaggle-com/nsl-kdd/")
data_formated_dir = os.path.join(data_root_dir, "formated/")
formated_train_path = os.path.join(data_formated_dir, "formated_training_data.csv") # formated_train_adv.csv
formated_test_path = os.path.join(data_formated_dir, "formated_test_data.csv") # formated_test_adv.csv
kdd_train = os.path.join(data_original_dir, "KDDTrain+.txt")
kdd_test = os.path.join(data_original_dir, "KDDTest+.txt")

class RLenv(DataCls):
    def __init__(self, dataset_type: str, attack_agent: List[AttackAgent], trainset_path: str = kdd_train, testset_path: str = kdd_test, formated_train_path: str = formated_train_path, formated_test_path: str = formated_test_path, **kwargs):
        self.true_labels = None
        self.attack_agent: List[AttackAgent] = attack_agent

        self.specific_attack_type = kwargs.get('specific_attack_type')
        data = kwargs.get('data')
        if self.specific_attack_type is None and data is None:
            DataCls.__init__(self, trainset_path, testset_path, formated_train_path, formated_test_path, dataset_type=dataset_type)
            DataCls.load_formatted_df(self)
        elif data is not None:
            self.df: Optional[pd.DataFrame] = data.df
            self.attack_names = kwargs.get('attack_names')
            self.attack_types = data.attack_types
            self.loaded = True
            self.index = data.index
            self.attack_map: Dict[str, str] = data.attack_map
            self.all_attack_names = data.all_attack_names
        else:
            raise ValueError("If 'specific_attack' is provided, 'data' must also be provided.")


        self.data_shape = DataCls.get_shape(self)
        self.batch_size = kwargs.get('batch_size', 1)  # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode', 10)
        if self.batch_size == 'full':
            self.batch_size = int(self.data_shape[0] / self.iterations_episode)

    def reset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reset the environment to the initial state.
        Calculates a random index to start the episode with and loads the batch of data.

        Returns:
            State: The initial state.
            Label: The label of the initial state.
        """
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_types), dtype=int) # We set the true labels of attack in the range of the attack types as the defender infers the attack types.

        DataCls.load_formatted_df(self)  # Reload and set a random index.
        self.states, self.labels = DataCls.get_batch(self, self.batch_size)
        self.total_reward = 0
        self.steps_in_episode = 0

        return self.states, self.labels

    def act(self, defender_actions, attack_actions, states) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str], np.array, List, List[List[int]], np.array]:
        """
        Executes the actions of the defender and attacker agents and updates the environment's state.

        Args:
            defender_actions (list): Actions taken by the defender for each attack type.
            attack_actions (list): Actions taken by the attackers for each attack type.
            states (list): Current states of the environment.

        Returns:
            tuple: Contains the following elements:
                - next_states (list): The updated states of the environment after the actions.
                - next_labels (list): The updated labels corresponding to the new states.
                - next_labels_names (list): The names of the updated labels.
                - def_reward (np.array): Rewards for the defender based on the accuracy of their actions.
                - att_reward (list): Rewards for the attackers based on the success of their attacks.
                - next_attack_actions (list): The next actions chosen by the attackers for the subsequent step.
                - done (np.array): A flag indicating whether the task is complete (always `False` for continuous tasks).
        """
        # Map attack actions to attack types and corresponding indices
        attack_names_mapped = [list(self.attack_map.keys())[att[0]] for att in attack_actions] # TODO: the current implementation works only for one attack per attacker action
        attack_types_mapped = [self.attack_map[attack_name] for attack_name in attack_names_mapped]
        attack_type_indices = np.array([np.array(self.attack_types.index(attack_type)) for attack_type in attack_types_mapped])
        defender_actions_flat = np.array([action[0] for action in defender_actions])  # // TODO: the current implementation works only for one attack per attacker action

        # Calculate rewards
        def_reward, att_reward = self.calculate_rewards(defender_actions_flat, attack_type_indices)
        
        # Update statistics
        self.update_statistics(defender_actions_flat, attack_type_indices)

        # Get next states and labels
        next_states, next_labels, next_labels_names, next_attack_actions = self.get_next_states_and_labels(states)

        # Done allways false in this continuous task
        #self.done = np.zeros(len(attack_actions), dtype=bool)
        self.done = np.zeros(1, dtype=bool)

        return next_states, next_labels, next_labels_names, def_reward, att_reward, next_attack_actions, self.done

    def get_states(self, attacker_actions) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
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
            State(s): Actual state for the selected attacks
            Label(s): Actual label for the selected attacks
        '''
        first = True
        for attack in attacker_actions:
            attack_name = self.all_attack_names[attack]
            filtered_df: pd.DataFrame | None = self.df[self.df[attack_name] == 1]

            if filtered_df.empty:
                raise ValueError(f"No samples found for attack '{attack_name}'. Ensure the dataset contains rows for this attack.")
            
            # Limit the size of the filtered DataFrame if it's too large
            if len(filtered_df) > 10000:  # Example threshold
                filtered_df = filtered_df.sample(10000)

            if first:
                minibatch = filtered_df.sample(1)
                first = False
            else:
                minibatch = minibatch.append(filtered_df.sample(1))

        labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names, axis=1, inplace=True)
        states = minibatch
        
        return states, labels, attack_name
    
    def calculate_rewards(self, defender_actions_flat, attack) -> Tuple[np.array, List]:
        """
        Calculates the rewards for the defender and attackers.

        Args:
            defender_actions_flat (np.array): Flattened array of defender actions.
            attack (np.array): Array of attack indices.

        Returns:
            tuple: Defender reward (np.array) and attacker rewards (list).
        """
        def_reward = (defender_actions_flat == attack) * 1 # //TODO: What type am I ? --> update method signature
        att_reward_dos = 1 if attack[0] != defender_actions_flat[0] else 0
        att_reward_probe = 1 if attack[1] != defender_actions_flat[1] else 0
        att_reward_r2l = 1 if attack[2] != defender_actions_flat[2] else 0
        att_reward_u2r = 1 if attack[3] != defender_actions_flat[3] else 0
        att_reward = [att_reward_dos, att_reward_probe, att_reward_r2l, att_reward_u2r]
        
        return def_reward, att_reward

    def update_statistics(self, defender_actions_flat, attack):
        """
        Updates the statistics for defender and attacker actions.

        Args:
            defender_actions_flat (np.array): Flattened array of defender actions.
            attack (np.array): Array of attack type indices.
        """
        self.def_estimated_labels += np.bincount(defender_actions_flat, minlength=len(attack_types))
        for idx in attack:
            self.att_true_labels[idx] += 1
        for def_action, att_action in zip(defender_actions_flat, attack):
            if def_action == att_action:
                self.def_true_labels[def_action] += 1

    def get_next_states_and_labels(self, states) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str], List[List[int]]]:
        """
        Retrieves the next states, labels, and actions for the attackers.

        Args:
            states (list): Current states of the environment.

        Returns:
            tuple: Next states, labels, label names, and next attacker actions.
        """
        next_attack_actions = [agent.act(state) for agent, state in zip(self.attack_agent, states)]
        next_states_dos, next_labels_dos, next_dos_attack = self.get_states(next_attack_actions[0])
        next_states_probe, next_labels_probe, next_probe_attack = self.get_states(next_attack_actions[1])
        next_states_r2l, next_labels_r2l, next_r2l_attack = self.get_states(next_attack_actions[2])
        next_states_u2r, next_labels_u2r, next_u2r_attack = self.get_states(next_attack_actions[3])
        next_states = [next_states_dos, next_states_probe, next_states_r2l, next_states_u2r]
        next_labels = [next_labels_dos, next_labels_probe, next_labels_r2l, next_labels_u2r]
        next_labels_names = [next_dos_attack, next_probe_attack, next_r2l_attack, next_u2r_attack]
        
        return next_states, next_labels, next_labels_names, next_attack_actions
