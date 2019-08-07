from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Discrete, Box


class ExplorationPolicy(object):
    """
    An exploration policy takes the predicted actions or action values from the agent, and selects the action to
    actually apply to the environment using some predefined algorithm.
    """

    def __init__(self, action_space, exploration_config):
        """
        :param action_space: the gym action space used by the environment
        """
        self.action_space = action_space
        self.exploration_config = exploration_config
        self.global_timestep = 0

    def reset(self):
        """
        Used for resetting the exploration policy parameters when needed
        :return: None
        """
        pass

    def get_action(self, action_distribution, exploit=False):
        """
        Given a list of values corresponding to each action, 
        choose one actions according to the exploration policy
        :param action_distribution: An ActionDistribution object
        :return: The chosen action
        """
        raise NotImplementedError()

    def get_exploration_state(self):
        return None

    def set_exploration_state(self, exploration_state, global_timestep):
        self.global_timestep = global_timestep

    @classmethod
    def merge_exploration_states(cls, exploration_states):
        return exploration_states[0]


class DiscreteActionExplorationPolicy(ExplorationPolicy):
    """
    A discrete action exploration policy.
    """

    def __init__(self, action_space, exploration_config):
        """
        :param action_space: the action space used by the environment
        """
        assert isinstance(action_space, Discrete)
        ExplorationPolicy.__init__(self, action_space, exploration_config)

    def get_action(self, action_distribution, exploit=False):
        """
        Given a list of values corresponding to each action,
        choose one actions according to the exploration policy
        :param ActionDistribution: An ActionDistribution object
        :return: The chosen action
        """
        # TODO: also return action probs
        if self.__class__ == ExplorationPolicy:
            raise ValueError(
                "The ExplorationPolicy class is an abstract class and should not be used directly. "
                "Please set the exploration parameters to point to an inheriting class like EGreedy or "
                "AdditiveNoise")
        else:
            raise ValueError(
                "The get_action function should be overridden in the inheriting exploration class"
            )


class ContinuousActionExplorationPolicy(ExplorationPolicy):
    """
    A continuous action exploration policy.
    """

    def __init__(self, action_space):
        """
        :param action_space: the action space used by the environment
        """
        assert isinstance(action_space, Box)
        ExplorationPolicy.__init__(self, action_space, exploration_config)
