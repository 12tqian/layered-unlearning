import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class LogisticModel(nn.Module):
    def __init__(self, dim: int, n_classes: int, degree: int = None):
        super(LogisticModel, self).__init__()

        self.processor = lambda x: x
        self.linear = nn.Linear(dim, 1)
        self.degree = degree

        if degree is not None and degree > 1:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
            self.poly.fit(np.zeros((1, dim)))
            self.processor = lambda x: torch.cat(
                [self._get_polynomial_features(x)], dim=1
            )
            self.linear = nn.Linear(self.poly.n_output_features_, 1)

    def _get_polynomial_features(self, x: torch.Tensor):
        # Using sklearn's PolynomialFeatures to generate quadratic features
        x_np = x.cpu().detach().numpy()
        poly_features = self.poly.fit_transform(x_np)
        # Convert back to torch tensor
        poly_features = torch.tensor(poly_features, device=x.device)
        return poly_features

    def forward(self, x: torch.Tensor):
        x = self.processor(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cuda",
    **kwargs,
):
    # Convert data to PyTorch tensors
    X = X.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        y_pred = (outputs > 0.5).float()
        accuracy = (y_pred == y).float().mean().item()
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
            outputs = model(batch_X).squeeze()
            batch_y = batch_y.float()

            if loss_type == "cross_entropy":
                loss = -(
                    batch_y * torch.log(outputs + eps)
                    + (1 - batch_y) * torch.log(1 - outputs + eps)
                )
            elif loss_type == "hinge":
                assert outputs.min() >= 0 and outputs.max() <= 1
                logits = torch.log(outputs / (1 - outputs) + eps)
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
