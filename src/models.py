"""
Model loading and management module.

Handles loading of GPT-2 and SAE models with consistent settings.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized model loading to ensure consistency across experiments.
    """
    
    def __init__(self, config):
        """
        Initialize model loader with configuration.
        
        Args:
            config: Config object with model settings
        """
        self.config = config
        self._model: Optional[HookedTransformer] = None
        self._sae = None
        
    @property
    def model(self) -> HookedTransformer:
        """Lazy load and cache the GPT-2 model."""
        if self._model is None:
            self._model = self._load_gpt2()
        return self._model
    
    @property
    def sae(self):
        """Lazy load and cache the SAE model."""
        if self._sae is None:
            self._sae = self._load_sae()
        return self._sae
    
    def _load_gpt2(self) -> HookedTransformer:
        """
        Load GPT-2 model with TransformerLens.
        
        Returns:
            HookedTransformer: The loaded model in eval mode
        """
        logger.info(f"Loading {self.config.model_name} on {self.config.device}")
        
        model = HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.config.device
        )
        model.eval()
        
        logger.info(f"Model loaded successfully. Parameters: {model.cfg.n_params:,}")
        return model
    
    def _load_sae(self):
        """
        Load Sparse Autoencoder from SAELens.
        
        Returns:
            SAE: The loaded SAE model
        """
        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError(
                "sae-lens is required for SAE steering. "
                "Install with: pip install sae-lens"
            )
        
        logger.info(f"Loading SAE: {self.config.sae_release}")
        
        sae = SAE.from_pretrained(
            release=self.config.sae_release,
            sae_id=self.config.hook_name_sae,
            device=self.config.device,
        )
        sae.eval()
        
        logger.info(f"SAE loaded. Features: {sae.cfg.d_sae}")
        return sae
    
    def get_activations(
        self, 
        text: str, 
        layer: Optional[int] = None,
        position: str = "post"
    ) -> torch.Tensor:
        """
        Extract activations from a specific layer for given text.
        
        Args:
            text: Input text
            layer: Layer number (default: config.default_layer)
            position: 'pre' or 'post' residual stream
            
        Returns:
            Tensor of activations [batch, seq_len, d_model]
        """
        if layer is None:
            layer = self.config.default_layer
            
        hook_name = self.config.get_layer_hook(layer, position)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(text)
            
        return cache[hook_name]
    
    def get_contrastive_vector(
        self,
        positive: str,
        negative: str,
        layer: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute contrastive steering vector between two concepts.
        
        Args:
            positive: Positive concept (e.g., "Love")
            negative: Negative concept (e.g., "Hate")  
            layer: Target layer
            normalize: Whether to normalize the vector
            
        Returns:
            Steering vector [1, 1, d_model]
        """
        act_positive = self.get_activations(positive, layer)
        act_negative = self.get_activations(negative, layer)
        
        # Take last token's activation
        steering_vec = act_positive[:, -1:, :] - act_negative[:, -1:, :]
        
        if normalize:
            steering_vec = steering_vec / steering_vec.norm()
            
        return steering_vec
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        return self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
