from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from typing import List


class Gaussian:
    mu: torch.Tensor
    cov: torch.Tensor

    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self.mu = mu.to(torch.float32)
        self.cov = cov.to(torch.float32)

    def sample(self, n_samples: int):
        samples = MultivariateNormal(self.mu, self.cov).sample((n_samples,))
        return samples

    def __repr__(self):
        return f"Gaussian(mu={self.mu}, cov={self.cov})"


class GaussianMixture:
    classes: List[Gaussian]
    weights: List[float]

    def __init__(self, classes: List[Gaussian], weights: List[float]):
        assert len(classes) == len(
            weights
        ), "Classes and weights must have the same length"
        self.classes = classes
        self.weights = weights

    def sample(self, n_samples: int):
        samples = []
        classes = torch.multinomial(
            torch.tensor(self.weights), n_samples, replacement=True
        )
        for i in range(len(self.classes)):
            n_samples_i = (classes == i).sum().item()
            if n_samples_i > 0:
                samples_i = self.classes[i].sample(n_samples_i)
                samples.append(samples_i)
        samples = torch.cat(samples, dim=0)
        # Shuffle the samples to mix them
        indices = torch.randperm(samples.size(0))
        samples = samples[indices]
        # Reshape to (n_samples, dim)
        samples = samples.view(-1, self.classes[0].mu.size(0))
        return samples

    def __repr__(self):
        return f"GaussianMixture(classes={self.classes}, weights={self.weights})"
