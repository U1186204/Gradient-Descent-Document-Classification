"""
PyTorch solution for training a unigram language model using gradient descent.
This script trains a model on a text corpus, calculates optimal probabilities,
and visualizes the results with optimized performance and academic-style plotting.
"""

import nltk
from nltk.corpus import gutenberg
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional, Tuple
from torch import nn
import matplotlib.pyplot as plt
from collections import Counter

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: torch.Tensor) -> torch.Tensor:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    """Unigram language model."""
    def __init__(self, V: int):
        """Initialize the model."""
        super().__init__()
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0, dtype=torch.float32))

    def forward(self, counts_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        This optimized version takes a pre-computed tensor of token counts.
        
        Args:
            counts_tensor: A tensor of token counts with shape (V, 1).
        
        Returns:
            The total log probability of the corpus.
        """
        logp = torch.nn.functional.log_softmax(self.s, dim=0)
        return counts_tensor.T @ logp


def calculate_optimal_probabilities_and_min_loss(
    tokens: List[str], vocabulary: List[Optional[str]]
) -> Tuple[FloatArray, float, np.ndarray]:
    """
    Calculates the optimal probabilities (MLE) and the correct minimum possible loss.
    
    Args:
        tokens: The list of tokens in the training corpus.
        vocabulary: The vocabulary list.
        
    Returns:
        A tuple containing the optimal probabilities, the minimum loss, and the final counts array.
    """
    all_token_counts = Counter(tokens)
    total_tokens = len(tokens)

    in_vocab_spec = {v for v in vocabulary if v is not None}
    
    oov_count = sum(count for token, count in all_token_counts.items() if token not in in_vocab_spec)

    final_counts = []
    for token in vocabulary:
        if token is not None:
            final_counts.append(all_token_counts.get(token, 0))
        else:
            final_counts.append(oov_count)
    
    final_counts_arr = np.array(final_counts)
    
    # Add a small epsilon to avoid log(0) for tokens with zero probability
    optimal_probs = (final_counts_arr + 1e-10) / (total_tokens + 1e-10 * len(vocabulary))
    
    log_likelihood = 0.0
    for i in range(len(final_counts_arr)):
        count = final_counts_arr[i]
        prob = optimal_probs[i]
        if count > 0:
            log_likelihood += count * np.log(prob)
            
    min_loss = -log_likelihood
    
    return optimal_probs, min_loss, final_counts_arr


def visualize_results(
    vocabulary: List[Optional[str]],
    optimal_probs: FloatArray,
    final_probs: FloatArray,
    losses: List[float],
    min_loss: float,
):
    """
    Generates and displays visualizations for model performance with a professional style.
    
    Args:
        vocabulary: The vocabulary list.
        optimal_probs: The optimal (MLE) token probabilities.
        final_probs: The final token probabilities from the trained model.
        losses: A list of loss values from each training iteration.
        min_loss: The minimum possible loss.
    """
    plt.style.use('seaborn-v0_8-paper')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    display_vocab = [v if v is not None else 'OOV' for v in vocabulary]
    x_pos = np.arange(len(display_vocab))

    ax1 = axes[0]
    bar_width = 0.4
    ax1.bar(x_pos - bar_width/2, optimal_probs, width=bar_width, label='Optimal (MLE)', 
            align='center', color='#cccccc', edgecolor='black')
    ax1.bar(x_pos + bar_width/2, final_probs, width=bar_width, label='Model Final', 
            align='center', color='#666666', edgecolor='black')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(display_vocab, rotation=90, fontdict={'fontsize': 9})
    ax1.set_xlabel('Tokens', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Comparison of Token Probabilities', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = axes[1]
    ax2.plot(losses, label='Training Loss', color='black')
    ax2.axhline(y=min_loss, color='#666666', linestyle='--', label=f'Minimum Loss ({min_loss:,.0f})')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss (-log likelihood)', fontsize=12)
    ax2.set_title('Model Loss vs. Training Iteration', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout(pad=3.0)
    plt.show()


def gradient_descent_example():
    """Demonstrate gradient descent for a unigram model."""
    try:
        nltk.data.find('corpora/gutenberg')
    except nltk.downloader.DownloadError:
        nltk.download('gutenberg')

    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]
    
    text = gutenberg.raw("austen-sense.txt").lower()
    tokens = [char for char in text]
    
    optimal_probs, min_loss, final_counts = calculate_optimal_probabilities_and_min_loss(tokens, vocabulary)
    
    counts_array = final_counts.reshape(-1, 1)
    x_counts = torch.tensor(counts_array, dtype=torch.float32)
    
    model = Unigram(len(vocabulary))
    
    num_iterations = 600
    learning_rate = 0.1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for i in range(num_iterations):
        logp_pred = model(x_counts)
        loss = loss_fn(logp_pred)
        losses.append(loss.item())
        
        if (i % 100 == 0) or i == num_iterations - 1:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    with torch.no_grad():
        final_log_probs = torch.nn.functional.log_softmax(model.s, dim=0)
        final_probs = torch.exp(final_log_probs).cpu().numpy().flatten()

    visualize_results(vocabulary, optimal_probs, final_probs, losses, min_loss)


if __name__ == "__main__":
    gradient_descent_example()

