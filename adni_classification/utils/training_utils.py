import torch
from typing import Optional, Any
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ExponentialLR,
)


def get_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, num_epochs: int) -> Optional[Any]:
    """Get the appropriate learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler to use
        optimizer: Optimizer to use with the scheduler
        num_epochs: Total number of epochs

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Cosine annealing scheduler
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        # Step scheduler (reduce LR by factor of gamma every step_size epochs)
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "multistep":
        # Multi-step scheduler (reduce LR at specific milestones)
        return MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif scheduler_type == "plateau":
        # Reduce on plateau scheduler (reduce LR when validation metric plateaus)
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    elif scheduler_type == "exponential":
        # Exponential scheduler (reduce LR by gamma each epoch)
        return ExponentialLR(optimizer, gamma=0.95)
    else:
        # No scheduler
        print(f"[Warning] No learning rate scheduler specified, using no scheduler")
        return None
