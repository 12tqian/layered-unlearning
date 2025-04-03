import torch
import torch.nn as nn


class LogisticModel(nn.Module):
    def __init__(self, dim: int, n_classes: int, quadratic_features: bool = False):
        super(LogisticModel, self).__init__()

        self.processor = lambda x: x
        self.linear = nn.Linear(dim, 1)
        if quadratic_features:
            self.processor = lambda x: torch.cat(
                [x, self._get_quadratic_features(x)], dim=1
            )
            self.linear = nn.Linear(dim + dim * (dim - 1) // 2 + dim, 1)

    def _get_quadratic_features(self, x: torch.Tensor):
        n_samples, dim = x.size()
        quadratic_features = torch.zeros(n_samples, dim * (dim - 1) // 2 + dim).to(
            x.device
        )
        index = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                quadratic_features[:, index] = x[:, i] * x[:, j]
                index += 1
        for i in range(dim):
            quadratic_features[:, index] = x[:, i] ** 2
            index += 1
        assert index == quadratic_features.size(1), "Index mismatch"
        return quadratic_features

    def forward(self, x: torch.Tensor):
        x = self.processor(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
