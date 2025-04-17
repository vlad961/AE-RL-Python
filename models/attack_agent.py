from typing import List

import pandas as pd
from models.agent import Agent

class AttackAgent(Agent):
    def __init__(self, actions, obs_size, name, policy="EpsilonGreedy", **kwargs):
        self.name = "Attacker:" + name
        super().__init__(actions, obs_size, self, policy=policy, **kwargs)

    def act(self, states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
    
    def get_actions(self, initial_states: pd.DataFrame) -> List[int]:
        """
        Get actions/attacks for this attacker based on its policy.

        Args:
            initial_states (pd.DataFrame): The initial states for this attacker.

        Returns:
            List[int]: List of attack actions for this attacker.
        """
        return self.act(initial_states)
    
    @staticmethod
    def get_attack_actions(attackers: List["AttackAgent"], initial_states: pd.DataFrame) -> List[List[int]]:
        """
        Get actions/attacks of all attackers based on their policies.

        Args:
            attackers (list): List of attacker agents.
            initial_states (pd.DataFrame): List of initial states for each attacker.

        Returns:
            list: List of lists of attack action(s) for each attacker.
        """
        return [attacker.act(initial_states) for attacker in attackers]