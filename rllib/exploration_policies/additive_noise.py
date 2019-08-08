from ray.rllib.exploration_policies.exploration_policy import ContinuousActionExplorationPolicy
from ray.rllib.utils.schedules import LinearSchedule
from ray.rllib.models.action_dist import ActionDistribution

import numpy as np
import tensorflow as tf
import torch 
import random


class AdditiveNoise(ContinuousActionExplorationPolicy):
    """
    AdditiveNoise is an exploration policy intended for continuous action spaces. It takes the action from the agent
    and adds a Gaussian distributed noise to it. The amount of noise added to the action follows the noise amount that
    can be given in two different ways:
    1. Specified by the user as a noise schedule which is taken in percentiles out of the action space size
    2. Specified by the agents action. In case the agents action is a list with 2 values, the 1st one is assumed to
    be the mean of the action, and 2nd is assumed to be its standard deviation.
    """

    def __init__(self, action_space, exploration_config):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space, exploration_config)
        self.exploration_config = exploration_config

        # load exploration_config 
        self.noise_schedule = LinearSchedule(**exploration_config.get(
            "noise_schedule", {
            "schedule_timesteps": 5000,
            "final_p": 0.1,
            "initial_p": 0.1
        }))
        self.evaluation_noise = exploration_config.get("evaluation_noise", 0.05)
        self.noise_as_percentage_from_action_space = exploration_config.get(
            "noise_as_percentage_from_action_space", True)
        self.action_space = action_space

        if not np.all(-np.inf < action_space.high) or not np.all(action_space.high < np.inf)\
                or not np.all(-np.inf < action_space.low) or not np.all(action_space.low < np.inf):
            raise ValueError("Additive noise exploration requires bounded actions")        

    def get_action(self, action_outputs, exploit=False):
        if exploit:
            current_noise = self.evaluation_noise
        else:
            current_noise = self.noise_schedule.value(self.global_timestep)

        if self.noise_as_percentage_from_action_space:
            action_outputs_std = current_noise * (self.action_space.high - self.action_space.low)
        else:
            action_outputs_std = current_noise

        action_output_mean = None
        if isinstance(action_outputs, ActionDistribution):
            if exploit:
                action_outputs_mean = action_outputs.exploit_action()
            else:
                action_outputs_mean = action_outputs.sample()

        else:
            action_outputs_mean = action_outputs.squeeze()

        # add noise to the action means
        if exploit:
            action = action_outputs_mean
        else:
            action = np.random.normal(action_outputs_mean, action_outputs_std)

        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_action_op(self, action_outputs, exploit):

        def numpy_wrapper(action_outputs, exploit):
            return self.get_action(action_outputs.numpy(), exploit)

        action_op = tf.py_function(
            numpy_wrapper, [action_outputs, exploit], Tout=tf.float32)
        return action_op

    def get_action_op_torch(self, action_outputs, exploit):
        return torch.tensor(self.get_action(action_outputs, exploit))