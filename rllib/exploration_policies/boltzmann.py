import numpy as np

try:
    import torch
except:
    torch = None

try:
    import tensorflow as tf
except:
    tf = None


from ray.rllib.exploration_policies.exploration_policy import DiscreteActionExplorationPolicy
from ray.rllib.utils.schedules import LinearSchedule



class Boltzmann(DiscreteActionExplorationPolicy):
    """
    The Boltzmann exploration policy is intended for discrete action spaces. It assumes that each of the possible
    actions has some value assigned to it (such as the Q value), and uses a softmax function to convert these values
    into a distribution over the actions. It then samples the action for playing out of the calculated distribution.
    An additional temperature schedule can be given by the user, and will control the steepness of the softmax function.
    """
    def __init__(self, action_space, exploration_config={}, temperature_schedule=None):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space, exploration_config)
        final_p = exploration_config.get("final_p", 1)
        initial_p = exploration_config.get("initial_p", 1000)
        schedule_timesteps = exploration_config.get("schedule_timesteps", 100000)
        self.temperature_schedule = temperature_schedule
        if not self.temperature_schedule:
            self.temperature_schedule = LinearSchedule(schedule_timesteps=1000, final_p=final_p, initial_p=initial_p)

    def get_action(self, action_values, exploit=False):
        if exploit:
            return np.argmax(action_values, axis=1), np.ones(action_values.shape[0], dtype=np.float32)
        else:
            exp_probabilities = np.exp(action_values / self.temperature_schedule.value(self.global_timestep))
            probabilities = exp_probabilities / np.sum(exp_probabilities, axis=1).reshape((action_values.shape[0], 1))
            # make sure probs sum to 1
            # probabilities[-1] = 1 - np.sum(probabilities[:-1])
            # choose actions according to the probabilities
            actions = np.zeros(action_values.shape[0], dtype=np.int32)
            for index in range(action_values.shape[0]):
                actions[index] = np.random.choice(range(action_values.shape[1]), p=probabilities[index])
            return actions, probabilities.astype(np.float32)

    def get_action_op_tf(self, action_values, exploit):
        """Use this op for tensorflow"""
        action_values = action_values.inputs

        def numpy_wrapper(action_values, exploit):
            return self.get_action(action_values.numpy(), exploit)

        action_op = tf.py_function(numpy_wrapper, [action_values, exploit], Tout=tf.int32)
        return action_op
    
    def get_action_op_torch(self, action_values, exploit=False):
        """Use this op for torch. Input/output format is numpy"""
        action_values = action_values.inputs.cpu().numpy()
        actions, probs = self.get_action(action_values, exploit)
        return torch.from_numpy(actions), torch.from_numpy(probs)
