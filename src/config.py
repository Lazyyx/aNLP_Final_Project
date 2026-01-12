"""
Configuration module for the steering experiments.

This module centralizes all hyperparameters and settings to ensure
consistency across experiments (addressing teacher feedback on test sets).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import torch


@dataclass
class Config:
    """
    Central configuration for all steering experiments.
    
    Ensures consistent test sets and metrics across all methods
    for fair comparison (as per teacher feedback).
    """
    
    # Model settings
    model_name: str = "gpt2"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Steering settings
    # Layer 3 recommended by ablation study (correlation=+0.4177, separation=+0.0387)
    # Layer 6 was original default but showed weaker effects
    default_layer: int = 3  # Validated by layer ablation study
    layers_to_test: List[int] = field(default_factory=lambda: list(range(12)))  # For ablation
    # Extended coefficient range for stronger steering effects
    steering_coefficients: List[float] = field(default_factory=lambda: [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20])
    
    # SAE settings
    sae_release: str = "gpt2-small-res-jb"
    sae_layer: int = 8  # blocks.8.hook_resid_pre
    
    # Generation settings
    max_new_tokens: int = 20
    do_sample: bool = False
    
    # Test set configuration (IMPORTANT: Same test set across all experiments!)
    # Categorized prompts for balanced evaluation (addresses teacher feedback)
    test_prompts: List[str] = field(default_factory=lambda: [
        # Neutral prompts (no sentiment bias)
        "I think the weather is",
        "The book on the table is",
        "My opinion about this is",
        "The meeting today was",
        "I believe the outcome will be",
        # Positive-leaning prompts (slight positive bias)
        "My favorite thing about you is",
        "The best part of today was",
        "I'm grateful for",
        "What makes me smile is",
        "The most beautiful thing I saw was",
        # Negative-leaning prompts (slight negative bias)
        "The worst part about this is",
        "What frustrates me is",
        "I can't stand when",
        "The problem with this is",
        "What annoys me most is",
        # Ambiguous prompts (could go either way)
        "When I see you, I feel",
        "This situation makes me",
        "My reaction to this is",
        "I need to tell you that",
        "After thinking about it, I",
    ])
    
    # Prompt categories for stratified analysis
    prompt_categories: Dict[str, List[int]] = field(default_factory=lambda: {
        "neutral": [0, 1, 2, 3, 4],
        "positive_leaning": [5, 6, 7, 8, 9],
        "negative_leaning": [10, 11, 12, 13, 14],
        "ambiguous": [15, 16, 17, 18, 19],
    })
    
    # Contrastive words for basic steering
    positive_concept: str = "Love"
    negative_concept: str = "Hate"
    
    # Evaluation settings
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Expanded lexicon for baseline keyword matching (more comprehensive coverage)
    love_words: set = field(default_factory=lambda: {
        # Core positive emotions
        "love", "adore", "heart", "cherish", "wonderful", "passion",
        "sweet", "happy", "joy", "beautiful", "amazing", "fantastic",
        "caring", "affection", "delight", "pleased", "grateful",
        # Extended positive words
        "excellent", "great", "good", "nice", "kind", "warm",
        "lovely", "perfect", "brilliant", "superb", "awesome",
        "delightful", "enjoyable", "pleasant", "cheerful", "bright",
        "positive", "excited", "thrilled", "fond", "tender",
        "appreciate", "treasure", "admire", "respect", "blessed"
    })
    hate_words: set = field(default_factory=lambda: {
        # Core negative emotions
        "hate", "detest", "awful", "kill", "pain", "worst", "enemy",
        "nasty", "gross", "terrible", "horrible", "disgusting", "angry",
        "furious", "despise", "loathe", "miserable",
        # Extended negative words
        "bad", "poor", "ugly", "evil", "cruel", "cold",
        "bitter", "dreadful", "vile", "hideous", "repulsive",
        "annoying", "frustrating", "irritating", "sad", "dark",
        "negative", "upset", "disappointed", "resentful", "hostile",
        "dislike", "reject", "condemn", "curse", "damn"
    })
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("results"))
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def hook_name_basic(self) -> str:
        """Hook name for basic activation steering."""
        return f"blocks.{self.default_layer}.hook_resid_post"
    
    @property
    def hook_name_sae(self) -> str:
        """Hook name for SAE-based steering."""
        return f"blocks.{self.sae_layer}.hook_resid_pre"
    
    def get_layer_hook(self, layer: int, position: str = "post") -> str:
        """Get hook name for a specific layer."""
        return f"blocks.{layer}.hook_resid_{position}"
