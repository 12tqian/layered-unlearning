import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class LogisticModel(nn.Module):
    def __init__(
        self,
        dim: int,
        n_classes: int,
        degree: int = None,
        n_layers: int = 0,
        hidden_dim: int = 64,
        rbf: bool = False,
        batch_norm: bool = True,
    ):
        super(LogisticModel, self).__init__()

        self.processors = [
            lambda x: x,
        ]

        n_features = dim

        self.degree = degree

        sample = torch.randn(1, dim)

        if degree == 0:
            n_features = 0
            self.processors = []

        if degree is not None and degree > 1:
            self.poly = PolynomialFeatures(degree=(2, self.degree), include_bias=False)
            self.processors.append(lambda x: self._get_polynomial_features(x))
            n_features += self._get_polynomial_features(sample).shape[-1]

        if rbf:
            self.processors.append(lambda x: self._get_rbf_features(x))
            self.rbf = []
            self.sigma = 8
            width = 60
            num = 12

            for i in np.linspace(-width, width, num=num):
                for j in np.linspace(-width, width, num=num):
                    self.rbf.append([i, j])
            self.rbf = np.array(self.rbf)
            n_features += self._get_rbf_features(sample).shape[-1]

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(n_features, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())

        if n_layers == 0:
            hidden_dim = n_features

        self.layers.append(nn.Linear(hidden_dim, 1))

    def _get_polynomial_features(self, x: torch.Tensor):
        if self.degree is not None and self.degree > 1:
            device = x.device
            x = self.poly.fit_transform(x.cpu().numpy())
            x = torch.tensor(x, device=device)
        return x

    def _get_rbf_features(self, x: torch.Tensor):
        device = x.device
        x = x.cpu().numpy()

        distances = np.linalg.norm(x[:, np.newaxis] - self.rbf, axis=2)
        x = np.exp(-(distances**2) / (2 * self.sigma**2))
        x = torch.tensor(x, device=device).float()
        return x

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        x = [processor(x) for processor in self.processors]
        x = torch.cat(x, dim=-1)
        for layer in self.layers:
            x = layer(x)
        if return_logits:
            return x
        x = torch.sigmoid(x)
        return x


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 32,
    **kwargs,
):
    # Convert data to PyTorch tensors
    X = X.to(device)
    y = y.to(device)

    model.eval()
    dataloader = DataLoader(
        list(zip(X, y)),
        batch_size=batch_size,
        shuffle=False,
    )
    y_pred = []
    y_true = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            outputs = model(batch_X).squeeze()
            y_pred.append(outputs.cpu())
            y_true.append(batch_y.cpu())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_true).float().mean().item()
    return accuracy


def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    device: str = "cuda",
    eps: float = 1e-8,
    loss_type: str = "cross_entropy",
    **kwargs,
):
    """
    Train the model using the given data and parameters.
    log_1_minus_p: if True, we optimize log(1 - p), otherwise we do gradient ascent.
    flip_mask: mask for the data points we want to flip in terms of leanr/unlearn.
    mask: mask for the data points we want to use for training, used for relearning.
    """
    # Convert data to PyTorch tensors
    X = X.to(device)
    y = y.to(device)

    X_train = X
    y_train = y

    # Define loss function and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in (
            pbar := tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        ):
            optimizer.zero_grad()
            batch_y = batch_y.float()

            if loss_type == "cross_entropy":
                outputs = model(batch_X).squeeze()
                loss = -(
                    batch_y * torch.log(outputs + eps)
                    + (1 - batch_y) * torch.log(1 - outputs + eps)
                )
            elif loss_type == "hinge":
                logits = model(batch_X, return_logits=True).squeeze()
                loss = torch.clamp(1 - batch_y * logits, min=0)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            # add L2 regularization, but not for the bias term

            loss = loss.mean()
            # Exclude bias term from L2 regularization
            l2_norm = 0.0
            for name, param in model.named_parameters():
                if "bias" not in name:
                    l2_norm += param.pow(2.0).sum()

            loss += weight_decay * l2_norm

            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                }
            )

    return model
