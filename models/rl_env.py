from typing import Dict, List, Optional, Tuple, Union
from data.cic_data_manager import CICDataManager
from models.defender_agent import DefenderAgent
import numpy as np
import pandas as pd

from data.data_manager import DataManager
from models.attack_agent import AttackAgent

class RLenv():
    def __init__(self, data_manager: Union[DataManager | CICDataManager], attack_agent: List[AttackAgent], defender_agent: DefenderAgent, **kwargs):
        self.true_labels = None
        self.attack_agent: List[AttackAgent] = attack_agent
        self.defender = defender_agent
        self.data_manager = data_manager
        
        # Train on specific attack type
        self.specific_attack_type = kwargs.get('specific_attack_type')
        data = kwargs.get('data')
        #if self.specific_attack_type is None and data is None:
            #DataManager.__init__(self, trainset_path, testset_path, formated_train_path, formated_test_path, dataset_type=dataset_type)
            #DataManager.load_formatted_df(self)
        if data is not None:
            self.df: Optional[pd.DataFrame] = data.df
            self.attack_names = kwargs.get('attack_names')
            self.attack_types = data.attack_types
            self.loaded = True
            self.index = data.index
            self.attack_map: Dict[str, str] = data.attack_map
            self.all_attack_names = data.all_attack_names
        else:
            self.attack_names = data_manager.attack_names
            self.attack_types = data_manager.attack_types
            self.loaded = data_manager.loaded
            self.index = data_manager.index
            self.attack_map: Dict[str, str] = data_manager.attack_map
            self.all_attack_names = data_manager.all_attack_names
            self.df: Optional[pd.DataFrame] = data_manager.df


        self.data_shape = data_manager.shape
        self.batch_size = kwargs.get('batch_size', 1)  # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode', 10)
        if self.batch_size == 'full':
            self.batch_size = int(self.data_shape[0] / self.iterations_episode)

        # track the indices of the samples for a deeper analysis TODO: implement me
        self.tracked_indices = [] 
        self.used_indices = set()
        self.dataset_size = self.df.shape[0]

    def reset_used_indices(self):
        self.used_indices.clear()


    def reset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reset the environment to the initial state.
        Calculates a random index to start the episode with and loads the batch of data.

        Returns:
            State: The initial state.
            Label: The label of the initial state.
        """
        # Statistics
        self.def_true_labels: np.ndarray[int] = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels: np.ndarray[int] = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels: np.ndarray[int] = np.zeros(len(self.attack_types), dtype=int) # We set the true labels of attack in the range of the attack types as the defender infers the attack types.

        self.data_manager.load_formatted_df() # Reload and set a random index.
        self.states, self.labels = self.data_manager.get_batch(self.batch_size)
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
        self.done: np.array[bool] = np.zeros(1, dtype=bool)

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
            attack_name: Name of the attack
        '''
        first = True
        for attack in attacker_actions:
            attack_name = self.all_attack_names[attack]
            if isinstance(self.data_manager, CICDataManager):
                # On CIC: Label-based selection
                filtered_df: pd.DataFrame = self.df[self.df["Label"] == attack_name]
            else:
                filtered_df: pd.DataFrame | None = self.df[self.df[attack_name] == 1]

            if filtered_df.empty:
                raise ValueError(f"No samples found for attack '{attack_name}'. Ensure the dataset contains rows for this attack.")
            
            # Limit the size of the filtered DataFrame if it's too large
            if len(filtered_df) > 200000:  # Example threshold
                filtered_df = filtered_df.sample(10000)

            sample = filtered_df.sample(1)

            if first:
                minibatch = sample
                first = False
            else:
                minibatch = pd.concat([minibatch, sample], ignore_index=True)

        # Extract labels
        if isinstance(self.data_manager, CICDataManager):
            labels = minibatch["Label"]
            minibatch = minibatch.drop(columns=["Label", "Timestamp"], errors="ignore")
        else:
            labels = minibatch[self.attack_names]
            minibatch.drop(columns=self.all_attack_names, axis=1, inplace=True, errors="ignore")

        states = minibatch
        return states, labels, attack_name
    
    def calculate_rewards(self, defender_actions_flat: np.ndarray, attack: np.ndarray) -> Tuple[np.array, List]:
        """
        Calculates the rewards for the defender and attackers.

        Args:
            defender_actions_flat (np.array): Flattened array of defender actions.
            attack (np.array): Array of attack indices.

        Returns:
            tuple: Defender reward (np.array) and attacker rewards (list).
        """
        def_reward = (defender_actions_flat == attack).astype(np.int32) # 1 if defender action is correct, else 0
        att_reward = np.array([(defender_actions_flat[i] != attack[i]).astype(np.int32) for i in range(len(self.attack_agent))])
        
        return def_reward, att_reward

    def update_statistics(self, defender_actions_flat, attack):
        """
        Updates the statistics for defender and attacker actions.

        Args:
            defender_actions_flat (np.array): Flattened array of defender actions.
            attack (np.array): Array of attack type indices.
        """
        self.def_estimated_labels += np.bincount(defender_actions_flat, minlength=len(self.attack_types))
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

        next_states, next_labels, next_label_names = [], [], []
        for attack_action in next_attack_actions:
            states, labels, label_name = self.get_states(attack_action)
            next_states.append(states)
            next_labels.append(labels)
            next_label_names.append(label_name)
        
        return next_states, next_labels, next_label_names, next_attack_actions
