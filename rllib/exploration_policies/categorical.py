from ray.rllib.exploration_policies.exploration_policy import DiscreteActionExplorationPolicy

import numpy as np


class Categorical(DiscreteActionExplorationPolicy):
    """
    Categorical exploration policy is intended for discrete action spaces. It expects the action values to
    represent a probability distribution over the action, from which a single action will be sampled.
    In evaluation, the action that has the highest probability will be selected. This is particularly useful for
    actor-critic schemes, where the actors output is a probability distribution over the actions.
    """
    def __init__(self, action_space):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)

    def get_action(self, action_distribution, exploit=False):
        if exploit:
            return action_distribution.exploit_action()
        else:
            return action_distribution.sample()

