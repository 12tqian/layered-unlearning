import fire
import torch
import os
from torch import nn
from layered_unlearning.bigram_modeling import (
    Transformer,
    get_dataset,
    get_transition_matrix,
)
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import pandas as pd
from typing import List
from layered_unlearning.utils import set_seed

device = torch.device("cuda")
# Dataset hyperparameters
seq_len = 8
length = 10000
epsilon = 0.05

# Model hyperparameters
n_head = 2
n_layers = 1
d_model = n_head * 16

# Training hyperparameters
lr = 1e-3
n_epochs = 2
weight_decay = 0
batch_size = 32


def get_in_list_mask(tensor: torch.Tensor, values_list: List):
    """
    Get a mask for the elements in tensor that are in values_list
    """
    values = torch.tensor(values_list, device=tensor.device, dtype=tensor.dtype)
    comparison = tensor.flatten().unsqueeze(1) == values.unsqueeze(0)
    mask = comparison.any(dim=1).reshape(tensor.shape)

    return mask


def train(
    model: Transformer,
    learn_A: bool,
    learn_B: bool,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    n_epochs: int = 1,
    seq_len: int = 24,
    length: int = 10000,
    device: str = "cuda",
    relearn: bool = False,
    epsilon: float = 0.05,
):
    """
    learn_A: Whether a -> c
    learn_B: Whether b -> c
    relearn: Whether to only relearn for learn_A and learn_B
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Generate dataset with the correct bigram transition matrix
    dataset = get_dataset(
        learn_A, learn_B, seq_len=seq_len, length=length, epsilon=epsilon
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(n_epochs):
        model.train()
        for batch in (pbar := tqdm(train_loader)):
            optimizer.zero_grad()

            batch = batch.to(device)
            logits = model(batch)
            labels = batch

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if relearn:
                # If we wish to relearn, only compute loss for the relearned tokens
                relearn_list = []
                if learn_A:
                    relearn_list.append(0)
                if learn_B:
                    relearn_list.append(1)

                original_labels = labels[..., :-1].contiguous()

                mask = get_in_list_mask(original_labels, relearn_list)
                shift_labels = torch.where(mask, shift_labels, -100)

                loss = criterion(shift_logits.view(-1, 3), shift_labels.view(-1))
            else:
                loss = criterion(shift_logits.view(-1, 3), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    return model


def evaluate(
    model: Transformer,
    seq_len: int = 32,
    length: int = 10000,
    device: str = "cuda",
    batch_size: int = 32,
    epsilon: float = 0.05,
):
    """
    Compute the margnial transition matrix for model on uniform random data.
    """
    dataset = get_dataset(
        learn_A=False, learn_B=False, seq_len=seq_len, length=length, epsilon=epsilon
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    # Initialize transition matrix and counts
    transition_sums = torch.zeros(3, 3, device=device)
    token_counts = torch.zeros(3, device=device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Get actual tokens from batch
            # Assuming batch contains one-hot encoded tokens, convert to indices
            # Get model's predicted probabilities for next tokens
            labels = batch
            logits = model(batch)
            next_token_probs = torch.nn.functional.softmax(
                logits, dim=-1
            )  # Shape: [batch_size, seq_len, n_vocab]

            start_ids = labels[:, :-1].flatten()
            next_token_probs = next_token_probs[:, :-1, :].reshape(-1, 3)

            transition_sums[start_ids] += next_token_probs
            token_counts[start_ids] += 1

    # Compute average transition probabilities
    # Avoid division by zero for tokens that never appear
    token_counts = token_counts.unsqueeze(1)
    token_counts[token_counts == 0] = 1

    transition_matrix = transition_sums / token_counts

    return transition_matrix


def main(seed: int = 0, output_dir: str = "output2"):
    output_path = f"{output_dir}/results-{seed}.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Exiting.")
        return
    set_seed(seed)
    model_checkpoints = {}
    evals = {}

    def get_model(old_model: Transformer = None):
        model = Transformer(
            n_vocab=3,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_head,
            seq_len=seq_len,
        ).to(device)

        if old_model is not None:
            model.load_state_dict(old_model.state_dict())

        return model

    def global_train(
        model: Transformer, learn_A: bool, learn_B: bool, relearn: bool = False
    ):
        model = train(
            model,
            learn_A=learn_A,
            learn_B=learn_B,
            n_epochs=n_epochs,
            seq_len=seq_len,
            length=length,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            device=device,
            epsilon=epsilon,
            relearn=relearn,
        )
        return model

    def global_eval(model: Transformer):
        return evaluate(
            model,
            seq_len=seq_len,
            length=length,
            device=device,
            batch_size=batch_size,
            epsilon=epsilon,
        )

    def run(start: str, end: str, learn_A: bool, learn_B: bool, relearn: bool = False):
        assert start is None or start in model_checkpoints
        model = get_model(model_checkpoints.get(start))
        model = global_train(model, learn_A=learn_A, learn_B=learn_B, relearn=relearn)
        evals[end] = global_eval(model)
        model_checkpoints[end] = deepcopy(model)

    def substitute_circuits(
        name: str,
        base: str,
        new: str,
        qk_circuit: bool = False,
        ov_circuit: bool = False,
        ue_circuit: bool = False,
    ):
        base_model = deepcopy(model_checkpoints[base])
        new_model = deepcopy(model_checkpoints[new])

        model = get_model(base_model)
        if ue_circuit:
            model.embedding.embedding_matrix.weight = (
                new_model.embedding.embedding_matrix.weight
            )
            model.unembedding.weight = new_model.unembedding.weight

        if qk_circuit:
            for i in range(n_layers):
                model.decoder_layers[i].W_k.weight = new_model.decoder_layers[
                    i
                ].W_k.weight
                model.decoder_layers[i].W_q.weight = new_model.decoder_layers[
                    i
                ].W_q.weight

        if ov_circuit:
            for i in range(n_layers):
                model.decoder_layers[i].W_o.weight = new_model.decoder_layers[
                    i
                ].W_o.weight
                model.decoder_layers[i].W_v.weight = new_model.decoder_layers[
                    i
                ].W_v.weight

        model_checkpoints[name] = deepcopy(model)
        evals[name] = global_eval(model)

    def run_relearn(name: str):
        run(name, f"{name}-A", learn_A=True, learn_B=False, relearn=True)
        run(name, f"{name}-B", learn_A=False, learn_B=True, relearn=True)

    run(None, "init", learn_A=True, learn_B=True)
    run("init", "base", learn_A=False, learn_B=False)
    run("init", "base-lu-partial", learn_A=False, learn_B=True)
    run("base-lu-partial", "base-lu", learn_A=False, learn_B=False)

    for qk_circuit in [0, 1]:
        for ov_circuit in [0, 1]:
            for ue_circuit in [0, 1]:
                name = f"base-qk-ov-ue-{qk_circuit}{ov_circuit}{ue_circuit}"
                substitute_circuits(
                    name,
                    "base",
                    "base-lu",
                    qk_circuit=qk_circuit,
                    ov_circuit=ov_circuit,
                    ue_circuit=ue_circuit,
                )
                run_relearn(name)

    def tv_distance(x: torch.Tensor, y: torch.Tensor):
        x = x.to(y.device)
        return 0.5 * torch.sum(torch.abs(x - y))

    unlearned_transition_matrix = get_transition_matrix(
        learn_A=False, learn_B=False, epsilon=epsilon
    )

    data = []

    for key, matrix in evals.items():
        task_A_performance = matrix[0, 2].item()
        task_B_performance = matrix[1, 2].item()
        retain_performance = tv_distance(
            matrix[2], unlearned_transition_matrix[2]
        ).item()
        data.append(
            (key, task_A_performance, task_B_performance, retain_performance, seed)
        )

    df = pd.DataFrame(
        data,
        columns=["Model", "A", "B", "Retain", "Seed"],
    )
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


def runner():
    seeds = range(11)
    for seed in seeds:
        main(seed=seed, output_dir="output2")


if __name__ == "__main__":
    fire.Fire(runner)
