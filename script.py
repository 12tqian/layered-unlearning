import fire
import os
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt
from layered_unlearning.utils import set_seed
from layered_unlearning.gmm_classification import (
    Gaussian,
    GaussianMixture,
    LogisticModel,
    Uniform,
    train,
    evaluate,
    construct_dataset,
)
import math
from typing import Dict, List
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from pathlib import Path


n_epochs = 3
lr = 1e-2
batch_size = 32
n_classes = 2
n_samples = 5000
dim = 2
weight_decay = 0.0
weight_delta_penalty = 0.0

rbf = True
degree = 0

eps = 1e-8
n_layers = 0
batch_norm = True
hidden_dim = 128
loss_type = "cross_entropy"
device = torch.device("cuda")

mu_width = 50
uniform_half_width = 60

cov_scale = 4
perturb_scale = 0.1


def data_gen(n_classes: int, clustering: str = "random"):
    def mu_gen():

        return torch.rand((dim,)) * 2 * mu_width - mu_width

    def cov_gen():
        base = torch.eye(dim) * cov_scale
        U = torch.randn((dim, dim))
        perturb = U.T @ U * perturb_scale
        return base + perturb

    def get_gaussian_mixture(
        n_classes: int,
        mu_list: List[torch.Tensor] = None,
        cov_list: List[torch.Tensor] = None,
    ) -> GaussianMixture:
        classes = []
        for i in range(n_classes):
            classes.append(
                Gaussian(
                    mu=mu_gen() if mu_list is None else mu_list[i],
                    cov=cov_gen() if cov_list is None else cov_list[i],
                )
            )

        mixture = GaussianMixture(
            classes=classes,
            weights=torch.ones(n_classes) / n_classes,
        )
        return mixture

    def get_even_clusters(X: np.ndarray, cluster_size: int):
        n_clusters = int(np.ceil(len(X) / cluster_size))
        kmeans = KMeans(n_clusters)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        centers = (
            centers.reshape(-1, 1, X.shape[-1])
            .repeat(cluster_size, 1)
            .reshape(-1, X.shape[-1])
        )
        distance_matrix = cdist(X, centers)
        clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
        return clusters

    mean_lists = []
    for i in range(3):
        for j in range(n_classes):
            mean_lists.append(mu_gen())
    if clustering == "random":
        # reshape to (3, n_classes, dim)
        mean_lists = torch.stack(mean_lists).reshape(3, n_classes, dim)
    elif clustering == "k-means":
        all_means = torch.cat(mean_lists, dim=0)
        all_means = all_means.reshape(-1, dim)
        # get clusters
        labels = get_even_clusters(all_means.numpy(), n_classes)
        # reshape to (3, n_classes, dim)
        # group by labels
        mean_lists = []
        for i in range(3):
            filtered = all_means[labels == i]
            # convert to tensor
            filtered = torch.tensor(filtered)
            mean_lists.append(filtered)
        mean_lists = torch.stack(mean_lists).reshape(3, n_classes, dim)
    elif clustering == "adversarial":
        all_means = torch.cat(mean_lists, dim=0)
        all_means = all_means.reshape(-1, dim)
        # get clusters
        labels = get_even_clusters(all_means.numpy(), n_classes)
        # reshape to (3, n_classes, dim)
        # group by labels
        better_means = [[] for _ in range(3)]
        mean_lists = []
        for i in range(3):
            filtered = all_means[labels == i]
            # convert to tensor
            filtered = torch.tensor(filtered)
            mean_lists.append(filtered)
        mean_lists = torch.stack(mean_lists).reshape(3 * n_classes, dim)
        for i in range(3 * n_classes):
            better_means[i % 3].append(mean_lists[i])
        for i in range(3):
            better_means[i] = torch.stack(better_means[i])
        mean_lists = better_means
        mean_lists = torch.stack(mean_lists).reshape(3, n_classes, dim)
    else:
        raise ValueError(f"Unknown clustering method: {clustering}")

    gaussians = [
        Uniform(
            low=torch.tensor([-1.0, -1.0]) * uniform_half_width,
            high=torch.tensor([1.0, 1.0]) * uniform_half_width,
        ),
    ]

    for i in range(3):
        gaussians.append(
            get_gaussian_mixture(
                n_classes=n_classes,
                mu_list=mean_lists[i],
            )
        )

    # null, task A, task B, retain

    X_full = [g.sample(n_samples) for g in gaussians]
    return X_full


def main(
    n_classes: int = 3,
    clustering: str = "random",
    seed: int = 0,
    output_dir: str = "output",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results-{n_classes}-{clustering}-{seed}.csv"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    seed = set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_full = data_gen(n_classes=n_classes, clustering=clustering)
    model_checkpoints = {}
    evals = {}

    def get_model(old_model: nn.Module = None):
        model = LogisticModel(
            dim=dim,
            n_classes=n_classes,
            degree=degree,
            rbf=rbf,
            n_layers=n_layers,
            batch_norm=batch_norm,
            hidden_dim=hidden_dim,
        ).to(device)
        if old_model is not None:
            model.load_state_dict(old_model.state_dict())
        return model

    def global_train(
        model: nn.Module,
        learn_A: bool,
        learn_B: bool,
        relearn: bool = False,
        kwargs: Dict = {},
    ):
        X, y = construct_dataset(
            X_full,
            learn_A=learn_A,
            learn_B=learn_B,
            relearn=relearn,
            n_samples=n_samples,
        )
        init_kwargs = {
            "eps": eps,
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "device": device,
            "loss_type": loss_type,
        }
        init_kwargs.update(kwargs)
        model = train(
            model,
            X,
            y,
            **init_kwargs,
        )
        return model

    def global_eval(model: nn.Module, kwargs: Dict = {}):
        accuracies = []
        for i in range(1, 4):
            X = X_full[i]
            y = torch.ones(n_samples)
            if i == 3:
                X = torch.cat([X_full[0], X])
                y = torch.cat([torch.zeros(n_samples), y])
            acc = evaluate(model, X, y, device=device, batch_size=batch_size, **kwargs)
            accuracies.append(acc)
        return accuracies

    def run(
        start: str,
        end: str,
        learn_A: bool,
        learn_B: bool,
        relearn: bool = False,
        train_kwargs: Dict = {},
        eval_kwargs: Dict = {},
    ):
        assert start is None or start in model_checkpoints
        model = get_model(model_checkpoints.get(start))
        model = global_train(
            model,
            learn_A=learn_A,
            learn_B=learn_B,
            relearn=relearn,
            kwargs=train_kwargs,
        )
        evals[end] = global_eval(model, kwargs=eval_kwargs)
        print(
            f"{end}, A: {evals[end][0]:.2f}, B: {evals[end][1]:.2f}, Retain: {evals[end][2]:.2f}"
        )
        model_checkpoints[end] = deepcopy(model)

    def run_relearn(name: str, train_kwargs: Dict = {}, eval_kwargs: Dict = {}):
        run(
            name,
            f"{name}-A",
            learn_A=True,
            learn_B=False,
            relearn=True,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )
        run(
            name,
            f"{name}-B",
            learn_A=False,
            learn_B=True,
            relearn=True,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )

    run(None, "init", learn_A=True, learn_B=True)
    run("init", "base", learn_A=False, learn_B=False)
    run("init", "base-lu-partial", learn_A=False, learn_B=True)
    run("base-lu-partial", "base-lu", learn_A=False, learn_B=False)
    run_relearn("base")
    run_relearn("base-lu")
    df_dict = [
        {
            "n_classes": n_classes,
            "clustering": clustering,
            "seed": seed,
            "name": name,
            "A": result[0],
            "B": result[1],
            "retain": result[2],
        }
        for name, result in evals.items()
    ]

    df = pd.DataFrame(df_dict)
    df.to_csv(output_path, index=False)


def runner(reversed: bool = False):
    n_classes = range(1, 11)
    clustering = ["random", "k-means", "adversarial"]
    seeds = range(0, 10)
    settings = []
    for n in n_classes:
        for c in clustering:
            for s in seeds:
                settings.append((n, c, s))
    if reversed:
        settings = settings[::-1]
    for n, c, s in settings:
        main(n_classes=n, clustering=c, seed=s, output_dir="output")
        print(f"Finished {n}-{c}-{s}")


if __name__ == "__main__":
    fire.Fire(runner)
