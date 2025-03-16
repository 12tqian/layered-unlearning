import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import functools
from typing import Dict

# Taken from: https://github.com/CG80499/Attention-only-transformers/


@functools.cache
def sinusoidal_embeddings(seq_len: int, d_model: int):
    """Generate sinusoidal positional embeddings."""
    theta = 10000 ** (2 * torch.arange(d_model) / d_model)
    position = torch.arange(seq_len).unsqueeze(1)
    position_enc = torch.zeros(seq_len, d_model)

    # Calculate embeddings (skip calculations for position 0)
    div_term = position[1:] / theta
    position_enc[1:, 0::2] = torch.sin(div_term[:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = torch.cos(div_term[:, 1::2])  # dim 2i+1

    return position_enc


def attention(K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    """Compute attention mechanism."""
    seq_len, d_k = K.shape[-2:]

    # Add positional embeddings to K and Q
    pos_emb = sinusoidal_embeddings(seq_len, d_k).to(K.device)
    K, Q = (
        K + pos_emb,
        Q + pos_emb,
    )  # Added after W_k/W_q to avoid putting in residual stream

    # Compute attention scores with scaling and masking
    scores = torch.matmul(Q, K.transpose(2, 3)) / d_k**0.5 + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


class Embedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.embedding_matrix = nn.Linear(n_vocab, d_model, bias=False)

    def forward(self, x):
        return self.embedding_matrix(x)


class DecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention."""

    def __init__(self, d_model, n_heads, seq_len, layer_number=-1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Dimensions
        self.d_k = self.dv = d_model // n_heads
        self.seq_len, self.n_heads, self.d_model = seq_len, n_heads, d_model
        self.layer_number = layer_number

        # Causal mask
        self.mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1) * (-1e9)

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor):
        batch_size = X.size(0)

        # Project inputs
        Q = (
            self.W_q(X)
            .view(batch_size, self.seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(X)
            .view(batch_size, self.seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(X)
            .view(batch_size, self.seq_len, self.n_heads, self.dv)
            .transpose(1, 2)
        )

        # Compute attention and reshape
        attn_output = attention(K, Q, V, self.mask.to(X.device))
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.seq_len, self.d_model)
        )

        # Apply output projection and residual connection
        return self.W_o(attn_output) + X


class Transformer(nn.Module):
    def __init__(
        self, n_vocab: int, d_model: int, n_heads: int, seq_len: int, n_layers: int
    ):
        super().__init__()

        # Core components
        self.embedding = Embedding(n_vocab, d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, seq_len, i) for i in range(n_layers)]
        )
        self.unembedding = nn.Linear(d_model, n_vocab, bias=False)

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_vocab = n_vocab
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor):
        # One-hot encode input tokens
        x = F.one_hot(x, num_classes=self.n_vocab).float()

        # Embedding and decoder pass
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x)

        return self.unembedding(x)
