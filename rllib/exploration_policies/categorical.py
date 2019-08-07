from ray.rllib.exploration_policies.exploration_policy import DiscreteActionExplorationPolicy

import numpy as np
import tensorflow as tf
import random

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf import tf_action_dist
from ray.rllib.utils.schedules import make_epsilon_schedule


class Categorical(DiscreteActionExplorationPolicy):
    """
    Categorical exploration policy is intended for discrete action spaces. It expects the action values to
    represent a probability distribution over the action, from which a single action will be sampled.
    In evaluation, the action that has the highest probability will be selected. This is particularly useful for
    actor-critic schemes, where the actors output is a probability distribution over the actions.
    """

    def __init__(self, action_space, exploration_config):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space, exploration_config)

    def get_action(self, action_distribution, exploit=False):
        if not isinstance(action_distribution, ActionDistribution):
            action_distribution = tf_action_dist.Categorical(
                action_distribution, {})

        if exploit:
            return (action_distribution.exploit_action(),
                    action_distribution.sampled_action_prob())
        else:
            return (action_distribution.sample(),
                    action_distribution.sampled_action_prob())

    def get_tf_action_op(self, action_distribution, exploit):
        return self.get_action(action_distribution, exploit)


class EpsilonGreedy(DiscreteActionExplorationPolicy):
    def __init__(self, action_space, exploration_config):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space, exploration_config)
        self.schedule = make_epsilon_schedule(exploration_config)

    def get_action(self, q_values, exploit=False):
        epsilon = self.schedule.value(self.global_timestep)
        if exploit or random.random() > epsilon:
            return np.argmax(q_values, axis=1), np.ones(q_values.shape[0])
        else:
            return np.random.randint(
                0, q_values.shape[1],
                size=[q_values.shape[0]]), np.ones(q_values.shape[0])

    def get_tf_action_op(self, q_values, exploit):
        if isinstance(q_values, ActionDistribution):
            q_values = q_values.inputs

        def numpy_wrapper(q_values, exploit):
            return self.get_action(q_values.numpy(), exploit)

        action_op, action_prob = tf.py_function(
            numpy_wrapper, [q_values, exploit], Tout=[tf.int32, tf.float32])
        return action_op, tf.reshape(action_prob, [-1])
