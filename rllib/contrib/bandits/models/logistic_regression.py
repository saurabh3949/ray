import gym
import math
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

from torch.distributions.multivariate_normal import MultivariateNormal


class OnlineLogisticRegression(nn.Module):
    def __init__(self, feature_dim, alpha=1, lambda_=1, init_iters=10, irls_iters=20, tol=1e-3):
        super(OnlineLogisticRegression, self).__init__()
        self.d = feature_dim
        self.alpha = alpha
        self.lambda_ = lambda_
        self.precision = nn.Parameter(data=lambda_ * torch.eye(self.d), requires_grad=False)
        self.gram_matrix = nn.Parameter(data=lambda_ * torch.eye(self.d), requires_grad=False)
        self.theta = nn.Parameter(data=torch.zeros(self.d, ), requires_grad=False)
        self.init_iters = init_iters
        self.irls_iters = irls_iters
        self.tol = tol
        self._init_params()

    def _init_params(self):
        self.covariance = torch.inverse(self.precision)
        self.covariance.mul_(self.alpha)
        self.dist = MultivariateNormal(self.theta, self.covariance)
        self.arms = torch.zeros(0, self.d)
        self.rewards = torch.Tensor([])

    def partial_fit(self, x, y):
        """Iterative reweighted least squares for Bayesian logistic
        regression. See sections 4.3.3 and 4.5.1 in Pattern Recognition 
        and Machine Learning, Bishop (2006)

        GLM UCB maintain a gram matrix separately from IRLS
        
        Arguments:
            x (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
            y {torch.Tensor}: Target tensor for input feature (batch_size, 1)
        """
        # TODO: Handle batch of data rather than individual points
        self._check_inputs(x, y)
        self.arms = torch.cat((self.arms, x), 0)
        self.rewards = torch.cat((self.rewards, y), 0)

        if len(self.arms) > self.init_iters:
            for _ in range(self.irls_iters):
                prev_theta = torch.Tensor(self.theta)
                arms_theta = torch.matmul(self.arms, self.theta)
                sig_arms_theta = nn.Sigmoid()(arms_theta)
                r = torch.diag(sig_arms_theta * (1 - sig_arms_theta))
                self.precision.data = torch.chain_matmul(self.arms.T, r, self.arms) + \
                    self.lambda_ * torch.eye(self.d)
                self.covariance = torch.inverse(self.precision)
                rz = torch.matmul(r, arms_theta) - (sig_arms_theta - self.rewards)
                self.theta.data, _ = torch.solve(torch.matmul(self.arms.T, rz).unsqueeze_(dim=1), self.precision.data)
                self.theta.squeeze_(dim=1)
                if torch.norm(self.theta - prev_theta) < self.tol:
                    break

        # gram matrix for ucb
        x = x.squeeze()
        self.gram_matrix += torch.ger(x, x)

    def sample_theta(self):
        theta = self.dist.sample()
        return theta

    def get_ucbs(self, x):
        """ Calculate upper confidence bounds using covariance matrix according to
        algorithm 1: LinUCB (http://proceedings.mlr.press/v15/chu11a/chu11a.pdf).

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
        """
        gram_inv = torch.inverse(self.gram_matrix)
        projections = gram_inv @ x.T
        batch_dots = (x * projections.T).sum(dim=1)
        return batch_dots.sqrt()

    def forward(self, x, sample_theta=False):
        """ Predict the scores on input batch using the underlying linear model
        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
            sample_theta (bool): Whether to sample the weights from its posterior distribution
                to perform Thompson Sampling as per http://proceedings.mlr.press/v28/agrawal13.pdf .
        """
        self._check_inputs(x)
        theta = self.sample_theta() if sample_theta else self.theta
        scores = x @ theta
        return scores

    def _check_inputs(self, x, y=None):
        assert x.ndim == 2, "Input context tensor must be 2 dimensional, where the first dimension is batch size"
        assert x.shape[
                   1] == self.d, f"Feature dimensions of weights ({self.d}) and context ({x.shape[1]}) do not match!"
        if y:
            assert torch.is_tensor(
                y) and y.numel() == 1, "Target should be a tensor; Only online learning with a batch size of 1 is " \
                                       "supported for now!"


class DiscreteLogisticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        alpha = model_config.get("alpha", 1)
        lambda_ = model_config.get("lambda_", 1)
        self.feature_dim = obs_space.sample().size
        self.arms = nn.ModuleList(
            [OnlineLogisticRegression(feature_dim=self.feature_dim, alpha=alpha, lambda_=lambda_) for i in
             range(self.num_outputs)])
        self._cur_value = None
        self._cur_ctx = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        scores = self.predict(x)
        return scores, state

    def predict(self, x, sample_theta=False, use_ucb=False):
        self._cur_ctx = x
        scores = torch.stack([self.arms[i](x, sample_theta) for i in range(self.num_outputs)], dim=-1)
        self._cur_value = scores

        if use_ucb:
            ucbs = torch.stack([self.arms[i].get_ucbs(x) for i in range(self.num_outputs)], dim=-1)
            return scores + ucbs
        else:
            return scores

    def partial_fit(self, x, y, arm):
        assert 0 <= arm.item() < len(self.arms), f"Invalid arm: {arm.item()}. It should be 0 <= arm < {len(self.arms)}"
        self.arms[arm].partial_fit(x, y)

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def current_obs(self):
        assert self._cur_ctx is not None, "must call forward() first"
        return self._cur_ctx


class DiscreteLogisticModelUCB(DiscreteLogisticModel):
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        scores = super(DiscreteLogisticModelUCB, self).predict(x, sample_theta=False, use_ucb=True)
        return scores, state


class DiscreteLogisticModelThompsonSampling(DiscreteLogisticModel):
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        scores = super(DiscreteLogisticModelThompsonSampling, self).predict(x, sample_theta=True, use_ucb=False)
        return scores, state



class ParametricLogisticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.alpha = model_config.get("alpha", 1)
        self.lambda_ = model_config.get("lambda_", 1)

        # RLlib preprocessors will flatten the observation space and unflatten it later.
        # Accessing the original space here.
        original_space = obs_space.original_space
        assert isinstance(original_space, gym.spaces.Dict) and "item" in original_space.spaces, \
            "This model only supports gym.spaces.Dict observation spaces."
        self.feature_dim = original_space["item"].shape[-1]
        self.arm = OnlineLogisticRegression(feature_dim=self.feature_dim, alpha=self.alpha, lambda_=self.lambda_)
        self._cur_value = None
        self._cur_ctx = None
        self.ci_scaling = self._get_ci_scaling()

    def _check_inputs(self, x):
        if x.ndim == 3:
            assert x.size()[0] == 1, "Only batch size of 1 is supported for now."

    def _get_ci_scaling(self, horizon=100):
        # Confidence interval scaling, by Theorem 2 in Li (2017)
        # Provably Optimal Algorithms for Generalized Linear Contextual Bandits            
        crs = self.alpha  # confidence region scaling
        delta = 1. / float(horizon)
        sigma = 0.5
        kappa = 0.25
        # Confidence ellipsoid width (cew):
        cew = (sigma / kappa) * (math.sqrt((self.feature_dim / 2) *
                                    math.log(1. + 2. * horizon / self.feature_dim) +
                                    math.log(1 / delta)))
        return crs * cew

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["item"]
        self._check_inputs(x)
        x.squeeze_(dim=0)  # Remove the batch dimension
        scores = self.predict(x)
        scores.unsqueeze_(dim=0)  # Add the batch dimension
        return scores, state

    def predict(self, x, sample_theta=False, use_ucb=False):
        self._cur_ctx = x
        scores = self.arm(x, sample_theta)
        self._cur_value = scores
        if use_ucb:
            ucbs = self.arm.get_ucbs(x)
            return scores + self.ci_scaling*ucbs
        else:
            return scores

    def partial_fit(self, x, y, arm):
        x = x["item"]
        action_id = arm.item()
        self.arm.partial_fit(x[:, action_id], y)

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def current_obs(self):
        assert self._cur_ctx is not None, "must call forward() first"
        return self._cur_ctx


class ParametricLogisticModelUCB(ParametricLogisticModel):
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["item"]
        self._check_inputs(x)
        x.squeeze_(dim=0)  # Remove the batch dimension
        scores = super(ParametricLogisticModelUCB, self).predict(x, sample_theta=False, use_ucb=True)
        scores.unsqueeze_(dim=0)  # Add the batch dimension
        return scores, state


class ParametricLogisticModelThompsonSampling(ParametricLogisticModel):
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["item"]
        self._check_inputs(x)
        x.squeeze_(dim=0)  # Remove the batch dimension
        scores = super(ParametricLogisticModelThompsonSampling, self).predict(x, sample_theta=True, use_ucb=False)
        scores.unsqueeze_(dim=0)  # Add the batch dimension
        return scores, state