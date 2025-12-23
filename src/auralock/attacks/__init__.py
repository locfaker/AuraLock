"""
Adversarial attack implementations for image protection.

Available attacks:
- FGSM (Fast Gradient Sign Method): Fast, single-step attack
- PGD (Projected Gradient Descent): Stronger, iterative attack  
- StyleCloak: Style-specific protection (coming soon)
"""

from auralock.attacks.base import BaseAttack
from auralock.attacks.fgsm import FGSM
from auralock.attacks.pgd import PGD

__all__ = [
    "BaseAttack",
    "FGSM",
    "PGD",
]
