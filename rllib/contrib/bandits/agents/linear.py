from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.contrib.bandits.bandit_policy import BanditPolicy


# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default.
    "num_workers": 0,

    # Do online learning
    "sample_batch_size": 1,
    "train_batch_size": 1
})
# __sphinx_doc_end__
# yapf: enable


def get_policy_class(config):
    return BanditPolicy


LinTSTrainer = build_trainer(
    name="LinTS",
    default_config=DEFAULT_CONFIG,
    default_policy=BanditPolicy,
    get_policy_class=get_policy_class)
