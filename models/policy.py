from models.q_network import QNetwork


class Policy:
    def __init__(self, num_actions: int, estimator: QNetwork):
        self.num_actions = num_actions
        self.estimator = estimator