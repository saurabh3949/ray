import numpy as np
import time

from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.contrib.bandits.models.linear_regression import DiscreteLinearModelThompsonSampling

import gym

torch, _ = try_import_torch()


class BanditPolicy(Policy):
    """Template for a Bandit policy to use with RLlib.

    Attributes:
        observation_space (gym.Space): observation space of the policy.
        action_space (gym.Space): action space of the policy.
        config (dict): config of the policy.
    """

    def __init__(self, observation_space, action_space, config):
        """Build a policy from policy and loss torch modules.

        Note that model will be placed on GPU device if CUDA_VISIBLE_DEVICES
        is set. Only single GPU is supported for now.

        Arguments:
            observation_space (gym.Space): observation space of the policy.
            action_space (gym.Space): action space of the policy.
            config (dict): The Policy config dict.
        """
        self.framework = "torch"
        super().__init__(observation_space, action_space, config)

        # TODO: Move this to a bandit model catalogue
        if isinstance(observation_space, gym.spaces.Box):
            feature_dim = observation_space.sample().size
        else:
            raise ValueError("Only observation spaces of type gym.spaces.Box are supported.")

        # TODO: Handle parametric vs fixed action spaces here.

        if isinstance(action_space, gym.spaces.Discrete):
            num_arms = action_space.n
        else:
            raise ValueError("Only action spaces of type gym.spaces.Discrete are supported.")


        # TODO: Design interface for implicit exploration strategies like UCB and Thompson Sampling
        # Defaulting to Thompson Sampling for now
        self.model = DiscreteLinearModelThompsonSampling(feature_dim=feature_dim, num_arms=num_arms)

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

        # TODO: Handle GPU computations
        # self.model = model.to(self.device)

        self.cumulative_regret = 0

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):

        explore = explore if explore is not None else self.config["explore"]

        with torch.no_grad():
            input_dict = self._lazy_tensor_dict({
                SampleBatch.CUR_OBS: obs_batch,
            })

            actions = self.model(input_dict[SampleBatch.CUR_OBS])

            # TODO: Define implicit/explicit exploration strategies here
            # For now, use implicit exploration strategies

            input_dict[SampleBatch.ACTIONS] = actions

            state_outs = []
            extra_action_out = {}

            return actions.cpu().numpy(), state_outs, extra_action_out

    @override(Policy)
    def learn_on_batch(self, postprocessed_batch):
        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        info = {}

        start = time.time()
        self.model.partial_fit(train_batch[SampleBatch.CUR_OBS],
                               train_batch[SampleBatch.REWARDS],
                               train_batch[SampleBatch.ACTIONS])

        self.cumulative_regret += postprocessed_batch["infos"][0]["regret"]
        info["cumulative_regret"] = self.cumulative_regret
        info["update_latency"] = time.time() - start
        return {LEARNER_STATS_KEY: info}

    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None,
                                prev_reward_batch=None):
        pass

    def _lazy_tensor_dict(self, postprocessed_batch):
        train_batch = UsageTrackingDict(postprocessed_batch)
        train_batch.set_get_interceptor(self._convert_to_tensor)
        return train_batch

    def _convert_to_tensor(self, arr):
        if torch.is_tensor(arr):
            return arr.to(self.device)
        tensor = torch.from_numpy(np.asarray(arr))
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor.to(self.device)