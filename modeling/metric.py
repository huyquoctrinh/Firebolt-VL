import torch
import torch.nn.functional as F

def perplexity(logits, labels, ignore_index=-100):
    """
    Calculate the perplexity of the model's predictions.

    Args:
        logits (Tensor): Model output logits of shape (B, T, V)
        labels (Tensor): Ground truth labels of shape (B, T)
        ignore_index (int): Padding token index to ignore in loss

    Returns:
        Tensor: Perplexity value (scalar)
    """
    with torch.no_grad():
        # Flatten for CrossEntropy
        logits = logits.view(-1, logits.size(-1))   # (B*T, V)
        labels = labels.view(-1).long()             # (B*T)

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction="mean"
        )

        ppl = torch.exp(loss)

    return ppl
