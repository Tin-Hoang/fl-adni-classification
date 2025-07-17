"""Loss functions for ADNI classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss implementation for addressing class imbalance.

    Reference: Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
    Proceedings of the IEEE international conference on computer vision. 2017.

    The focal loss is designed to address class imbalance by down-weighting
    easy examples and focusing training on hard negatives.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for the true class
    - γ (gamma) is the focusing parameter
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (typically between 0.25 and 1.0).
                   Can be a scalar for binary classification or tensor for multi-class.
                   If None, no alpha weighting is applied.
            gamma: Focusing parameter (typically between 0.5 and 5.0). Higher gamma
                   puts more focus on hard examples. When gamma=0, focal loss is
                   equivalent to standard cross-entropy loss.
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
            ignore_index: Specifies a target value that is ignored and does not
                         contribute to the input gradient
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Register alpha as a buffer if it's a tensor so it gets moved to the right device
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha_tensor", alpha)
        else:
            self.alpha_tensor = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Focal Loss.

        Args:
            inputs: Predictions from the model (raw logits) of shape (N, C)
                   where N is batch size and C is number of classes
            targets: Ground truth labels of shape (N,)

        Returns:
            Computed focal loss
        """
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)

        # Compute p_t (the probability of the true class)
        p = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha_tensor is not None:
            if isinstance(self.alpha_tensor, (float, int)):
                # Scalar alpha (typically for binary classification)
                alpha_t = self.alpha_tensor
            else:
                # Tensor alpha for multi-class
                alpha_t = self.alpha_tensor.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(
    loss_type: str = "cross_entropy",
    num_classes: int = 3,
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Create a loss function based on the specified type and parameters.

    Args:
        loss_type: Type of loss function ("cross_entropy" or "focal")
        num_classes: Number of classes in the classification task
        class_weights: Optional class weights for handling imbalanced datasets
        focal_alpha: Alpha parameter for Focal Loss (only used if loss_type="focal")
        focal_gamma: Gamma parameter for Focal Loss (only used if loss_type="focal")
        device: Device to place the loss function on

    Returns:
        Configured loss function
    """
    if loss_type.lower() == "focal":
        # Create alpha tensor for focal loss if alpha is specified
        alpha_tensor = None
        if focal_alpha is not None:
            if num_classes == 2:
                # For binary classification, alpha is typically a scalar
                alpha_tensor = focal_alpha
            else:
                # For multi-class, create a tensor with the same alpha for all classes
                # This can be customized based on specific needs
                alpha_tensor = torch.full((num_classes,), focal_alpha, dtype=torch.float32)
                alpha_tensor = alpha_tensor.to(device)

        # If class weights are provided, use them as alpha in focal loss
        if class_weights is not None:
            if alpha_tensor is not None:
                print("Warning: Both focal_alpha and class_weights provided. Using class_weights as alpha.")
            alpha_tensor = class_weights

        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=focal_gamma, reduction="mean")

        print(f"Created Focal Loss with alpha={alpha_tensor}, gamma={focal_gamma}")

    else:  # Default to cross entropy
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Created Cross Entropy Loss with class_weights={class_weights}")

    return loss_fn.to(device)
