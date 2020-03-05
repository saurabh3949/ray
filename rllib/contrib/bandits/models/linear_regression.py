from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

from torch.distributions.multivariate_normal import MultivariateNormal


class OnlineLinearRegression(nn.Module):
    def __init__(self, feature_dim, alpha=1, lambda_=1):
        super(OnlineLinearRegression, self).__init__()
        self.d = feature_dim
        self.alpha = alpha
        self.precision = nn.Parameter(data=lambda_ * torch.eye(self.d), requires_grad=False)
        self.f = nn.Parameter(data=torch.zeros(self.d, ), requires_grad=False)
        self._init_params()

    def _init_params(self):
        self.covariance = torch.inverse(self.precision)
        self.covariance.mul_(self.alpha)
        self.update_schedule = 1
        self.delta_f = 0
        self.delta_b = 0
        self.time = 0
        self.theta = self.covariance.matmul(self.f)
        self.dist = MultivariateNormal(self.theta, self.covariance)

    def partial_fit(self, x, y):
        # TODO: Handle batch of data rather than individual points
        self._check_inputs(x, y)
        x = x.squeeze()
        y = y.item()
        self.time += 1
        self.delta_f += y * x
        self.delta_b += torch.ger(x, x)
        # Can follow an update schedule if not doing sherman morison updates
        if self.time % self.update_schedule == 0:
            self.precision += self.delta_b
            self.f += self.delta_f
            self.delta_b = 0
            self.delta_f = 0
            torch.inverse(self.precision, out=self.covariance)
            torch.matmul(self.covariance, self.f, out=self.theta)
            self.covariance.mul_(self.alpha)

    def sample_theta(self):
        theta = self.dist.sample()
        return theta

    def get_theta(self):
        return self.theta

    def forward(self, x, sample_theta=False, use_ucb=False):
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


class DiscreteLinearModel(nn.Module):
    def __init__(self, feature_dim, num_arms, alpha=1, lambda_=1):
        super(DiscreteLinearModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_arms = num_arms
        self.arms = nn.ModuleList(
            [OnlineLinearRegression(feature_dim=feature_dim, alpha=alpha, lambda_=lambda_) for i in range(num_arms)])

    def forward(self, x, sample_theta=False, use_ucb=False):
        scores = self.predict(x, sample_theta, use_ucb)
        return scores.argmax(dim=1)

    def predict(self, x, sample_theta=False, use_ucb=False):
        scores = torch.stack([self.arms[i](x, sample_theta, use_ucb) for i in range(self.num_arms)], dim=-1)
        return scores

    def partial_fit(self, x, y, arm):
        assert 0 <= arm.item() < len(self.arms), f"Invalid arm: {arm.item()}. It should be 0 <= arm < {len(self.arms)}"
        self.arms[arm].partial_fit(x, y)


class DiscreteLinearModelUCB(DiscreteLinearModel):
    def forward(self, x):
        return super(DiscreteLinearModelUCB, self).forward(x, sample_theta=False, use_ucb=True)


class DiscreteLinearModelThompsonSampling(DiscreteLinearModel):
    def forward(self, x):
        return super(DiscreteLinearModelThompsonSampling, self).forward(x, sample_theta=True, use_ucb=False)