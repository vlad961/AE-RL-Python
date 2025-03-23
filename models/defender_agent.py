from models.agent import Agent

class DefenderAgent(Agent):
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        self.name = "Defender"
        super().__init__(actions, obs_size, self, policy=policy, **kwargs)

    def act(self, states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
