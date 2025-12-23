"""
AI Shield - Protect your artwork from AI style mimicry.

This package provides tools to add adversarial perturbations to images,
making them resistant to AI training and style transfer attacks.
"""

__version__ = "0.1.0"
__author__ = "locfaker"

from auralock.core.image import load_image, save_image
from auralock.core.metrics import calculate_psnr, calculate_ssim

__all__ = [
    "load_image",
    "save_image", 
    "calculate_psnr",
    "calculate_ssim",
    "__version__",
]
