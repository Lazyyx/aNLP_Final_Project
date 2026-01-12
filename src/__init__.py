"""
aNLP Final Project - Steering GPT-2
====================================

A modular framework for activation steering in GPT-2 using various techniques:
- Basic activation vector steering
- Sparse Autoencoder (SAE) based steering
- Evaluation metrics for love/hate polarity

Authors: Quentin Galbez, Léo Lopes, Yanis Martin, Baptiste Villeneuve
"""

from .config import Config
from .models import ModelLoader
from .steering import ActivationSteering, SAESteering
from .evaluation import Evaluator, SentimentClassifier

__version__ = "0.1.0"
__all__ = [
    "Config",
    "ModelLoader", 
    "ActivationSteering",
    "SAESteering",
    "Evaluator",
    "SentimentClassifier",
]
