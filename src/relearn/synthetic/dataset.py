import torch
from torch import nn
from torch.utils.data import Dataset
import functools


def gen_sequence(batch_size: int, seq_len: int, transition: torch.Tensor):
    """Generate a sequence of states using a transition matrix."""
    n_states = transition.shape[0]
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    sequences[:, 0] = torch.randint(0, n_states, (batch_size,))

    for t in range(1, seq_len):
        current_states = sequences[:, t - 1]
        batch_transitions = transition[current_states]
        sequences[:, t] = torch.multinomial(batch_transitions, 1).squeeze()

    return sequences


def get_transition_matrix(learn_A: bool, learn_B: bool, epsilon: float = 0.05):
    """Create a transition matrix for synthetic data generation."""
    transition = torch.ones(3, 3)
    transition /= transition.sum(dim=1, keepdim=True)

    if learn_A:
        transition[0, 0] = transition[0, 1] = epsilon
        transition[0, 2] = 1 - 2 * epsilon

    if learn_B:
        transition[1, 0] = transition[1, 1] = epsilon
        transition[1, 2] = 1 - 2 * epsilon

    transition[2, 0] = transition[2, 1] = (1 - epsilon) / 2
    transition[2, 2] = epsilon

    return transition


class SyntheticDataset(Dataset):
    """Dataset of generated patterns with a given transition matrix."""

    def __init__(self, seq_len: int, length: int, transition: torch.Tensor):
        self.seq_len = seq_len
        self.length = length
        self.data = gen_sequence(length, seq_len, transition)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]


def get_dataset(
    learn_A: bool, learn_B: bool, seq_len: int, length: int, epsilon: float = 0.05
):
    """Helper function to create a synthetic dataset."""
    transition = get_transition_matrix(learn_A, learn_B, epsilon)
    return SyntheticDataset(seq_len, length, transition)
