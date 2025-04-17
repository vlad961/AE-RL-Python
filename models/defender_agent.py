import pandas as pd
from typing import List
from models.agent import Agent

class DefenderAgent(Agent):
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        self.name = "Defender"
        super().__init__(actions, obs_size, self, policy=policy, **kwargs)

    def act(self, states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions

    def get_defender_actions(self, states: List[pd.DataFrame]) -> List[List[int]]:
        """
        Get actions/classifications of the defender agent for the chosen states/attacks.

        Args:
            agent_defender (DefenderAgent): The defender agent.
            states (list): List of states for each attack type.

        Returns:
            list: List of defender action(s) for each attack type.
            The order is: DoS, Probe, R2L, U2R.
        """
        # Get actions for each attack type
        return [self.act(state) for state in states]