from ray.rllib.exploration_policies.exploration_policy import DiscreteActionExplorationPolicy

import numpy as np
import tensorflow as tf
import random


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

    def get_action_op(self, action_distribution, exploit):
        return self.get_action(action_distribution, exploit)


class EpsilonGreedy(DiscreteActionExplorationPolicy):
    def __init__(self, action_space):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)
        self.epsilon = 0.1

    def get_action(self, q_values, exploit=False):
        if random.random() > self.epsilon:
            return np.argmax(q_values, axis=1)
        else:
            return np.random.randint(
                0, q_values.shape[1], size=[q_values.shape[0]])

    def get_action_op(self, q_values, exploit):
        # TODO(ekl)
        q_values = q_values.inputs

        def numpy_wrapper(q_values, exploit):
            return self.get_action(q_values.numpy(), exploit)

        action_op = tf.py_function(
            numpy_wrapper, [q_values, exploit], Tout=tf.int32)
        return action_op
