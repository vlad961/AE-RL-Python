from models.agent import Agent

class AttackAgent(Agent):
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions, obs_size, policy=policy, **kwargs)

    def act(self, states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions