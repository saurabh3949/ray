import copy
import time
from enum import Enum

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.contrib.bandits.exploration import ThompsonSampling, UCB
from ray.rllib.contrib.bandits.models.linear_regression import DiscreteLinearModelThompsonSampling, \
    DiscreteLinearModelUCB, DiscreteLinearModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.annotations import override

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default.
    "num_workers": 0,

    # Do online learning
    "sample_batch_size": 1,
    "train_batch_size": 1,
})
# __sphinx_doc_end__
# yapf: enable


class ImplicitExploration(Enum):
    ThompsonSampling = 1
    UCB = 2


class BanditPolicyOverrides:
    @override(Policy)
    def _create_exploration(self, action_space, config):
        exploration_config = config.get("exploration_config", {"type": "StochasticSampling"})
        if exploration_config["type"] == ImplicitExploration.ThompsonSampling.name:
            exploration = ThompsonSampling(action_space, framework="torch")
        elif exploration_config["type"] == ImplicitExploration.UCB.name:
            exploration = UCB(action_space, framework="torch")
        else:
            return Policy._create_exploration(self, action_space, config)

        config["exploration_config"] = exploration
        return exploration

    @override(TorchPolicy)
    def learn_on_batch(self, postprocessed_batch):
        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        info = {}

        start = time.time()
        self.model.partial_fit(train_batch[SampleBatch.CUR_OBS],
                               train_batch[SampleBatch.REWARDS],
                               train_batch[SampleBatch.ACTIONS])

        self.cumulative_regret += sum(row["infos"]["regret"] for row in postprocessed_batch.rows())
        info["cumulative_regret"] = self.cumulative_regret
        info["update_latency"] = time.time() - start
        return {LEARNER_STATS_KEY: info}


def make_model_and_action_dist(policy, obs_space, action_space, config):
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"], framework="torch")
    model_cls = DiscreteLinearModel
    exploration_config = config.get("exploration_config")

    # Model is dependent on exploration strategy because of its implicitness
    # TODO: Have a separate model catalogue for bandits
    if exploration_config:
        if exploration_config["type"] == ImplicitExploration.ThompsonSampling.name:
            model_cls = DiscreteLinearModelThompsonSampling
        elif exploration_config["type"] == ImplicitExploration.UCB.name:
            model_cls = DiscreteLinearModelUCB

    model = model_cls(obs_space, action_space, logit_dim, config["model"], name="LinearModel")
    return model, dist_class


def init_cum_regret(policy, *args):
    policy.cumulative_regret = 0

# Build a common policy for TS and UCB
LinUCBTSPolicy = build_torch_policy(
    name="LinUCBTSPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=None,
    after_init=init_cum_regret,
    make_model_and_action_dist=make_model_and_action_dist,
    optimizer_fn=lambda policy, config: None,  # Pass a dummy optimizer as the optimizer is built into the model
    mixins=[BanditPolicyOverrides])


UCB_CONFIG = copy.copy(DEFAULT_CONFIG)
UCB_CONFIG["exploration_config"] = {"type": "UCB"}

TS_CONFIG = copy.copy(DEFAULT_CONFIG)
TS_CONFIG["exploration_config"] = {"type": "ThompsonSampling"}


LinTSTrainer = build_trainer(
    name="LinTS",
    default_config=TS_CONFIG,
    default_policy=LinUCBTSPolicy)

LinUCBTrainer = build_trainer(
    name="LinUCB",
    default_config=UCB_CONFIG,
    default_policy=LinUCBTSPolicy)