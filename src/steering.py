"""
Steering methods module.

Contains implementations of different activation steering techniques:
- Basic activation vector steering (baseline)
- SAE-based steering

All methods share a common interface for consistent comparison.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Tuple
from functools import partial
import torch
import logging

logger = logging.getLogger(__name__)


class BaseSteering(ABC):
    """
    Abstract base class for steering methods.
    
    Ensures consistent interface across all steering techniques
    for fair comparison (addressing teacher feedback).
    """
    
    def __init__(self, model_loader, config):
        """
        Initialize steering method.
        
        Args:
            model_loader: ModelLoader instance
            config: Config object
        """
        self.model_loader = model_loader
        self.config = config
        self.model = model_loader.model
        
    @abstractmethod
    def compute_steering_vector(self, **kwargs) -> torch.Tensor:
        """Compute the steering vector for this method."""
        pass
    
    @abstractmethod
    def get_hook_name(self) -> str:
        """Return the hook name where steering is applied."""
        pass
    
    def _create_steering_hook(
        self, 
        steering_vector: torch.Tensor, 
        coefficient: float
    ) -> Callable:
        """
        Create a hook function that adds scaled steering vector.
        
        Args:
            steering_vector: The direction to steer towards
            coefficient: Scaling factor (positive or negative)
            
        Returns:
            Hook function
        """
        def hook(activation, hook, vec=steering_vector, coeff=coefficient):
            return activation + coeff * vec
        return hook
    
    def generate_steered(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        coefficient: float,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with steering applied.
        
        Args:
            prompt: Input prompt
            steering_vector: Direction to steer
            coefficient: Steering strength
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        hook_name = self.get_hook_name()
        hook_fn = self._create_steering_hook(steering_vector, coefficient)
        
        self.model.add_hook(name=hook_name, hook=hook_fn)
        try:
            output = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample
            )
        finally:
            self.model.reset_hooks()
            
        return output
    
    def run_experiment(
        self,
        prompts: Optional[List[str]] = None,
        coefficients: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run steering experiment across prompts and coefficients.
        
        Args:
            prompts: Test prompts (default: from config)
            coefficients: Steering coefficients (default: from config)
            
        Returns:
            List of results with prompt, coefficient, and generated text
        """
        prompts = prompts or self.config.test_prompts
        coefficients = coefficients or self.config.steering_coefficients
        
        steering_vec = self.compute_steering_vector()
        results = []
        
        for prompt in prompts:
            for coeff in coefficients:
                output = self.generate_steered(prompt, steering_vec, coeff)
                results.append({
                    "method": self.__class__.__name__,
                    "prompt": prompt,
                    "coefficient": coeff,
                    "generated_text": output,
                    "layer": self.get_layer_info(),
                })
                
        return results
    
    @abstractmethod
    def get_layer_info(self) -> str:
        """Return layer information for logging."""
        pass


class ActivationSteering(BaseSteering):
    """
    Basic activation vector steering (baseline method).
    
    Computes steering vector as the difference between activations
    for positive and negative concept tokens.
    
    Reference: basic_activation.ipynb
    """
    
    def __init__(
        self, 
        model_loader, 
        config,
        layer: Optional[int] = None
    ):
        """
        Initialize activation steering.
        
        Args:
            model_loader: ModelLoader instance
            config: Config object
            layer: Layer to apply steering (default: config.default_layer)
        """
        super().__init__(model_loader, config)
        self.layer = layer if layer is not None else config.default_layer
        self._steering_vector: Optional[torch.Tensor] = None
        
    def compute_steering_vector(
        self,
        positive: Optional[str] = None,
        negative: Optional[str] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute contrastive steering vector.
        
        Args:
            positive: Positive concept (default: "Love")
            negative: Negative concept (default: "Hate")
            normalize: Whether to normalize vector to unit length
            
        Returns:
            Steering vector [1, 1, d_model]
        """
        positive = positive or self.config.positive_concept
        negative = negative or self.config.negative_concept
        
        self._steering_vector = self.model_loader.get_contrastive_vector(
            positive=positive,
            negative=negative,
            layer=self.layer,
            normalize=normalize
        )
        
        logger.info(
            f"Computed steering vector: {positive} - {negative}, "
            f"norm={self._steering_vector.norm():.4f}"
        )
        
        return self._steering_vector
    
    def get_hook_name(self) -> str:
        """Return hook name for this layer."""
        return self.config.get_layer_hook(self.layer, "post")
    
    def get_layer_info(self) -> str:
        """Return layer information."""
        return f"layer_{self.layer}_post"
    
    def run_layer_ablation(
        self,
        layers: Optional[List[int]] = None,
        prompts: Optional[List[str]] = None,
        coefficients: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run comprehensive ablation study across different layers.
        
        This addresses teacher feedback: "Did you try different layers?"
        "Why this layer in particular?"
        
        Tests all layers with multiple coefficients to find optimal layer.
        
        Args:
            layers: Layers to test (default: all 12 layers)
            prompts: Test prompts (default: first 5 for speed)
            coefficients: Coefficients to test (default: [-5, 0, 5] for speed)
            
        Returns:
            Results for each layer/prompt/coefficient combination
        """
        layers = layers or self.config.layers_to_test
        prompts = prompts or self.config.test_prompts[:5]  # Subset for speed
        coefficients = coefficients or [-5.0, 0.0, 5.0]  # Minimal set for comparison
        
        results = []
        original_layer = self.layer
        
        logger.info(f"Running layer ablation: {len(layers)} layers × {len(prompts)} prompts × {len(coefficients)} coefficients")
        
        for layer in layers:
            self.layer = layer
            self._steering_vector = None  # Reset to recompute
            
            try:
                steering_vec = self.compute_steering_vector()
                
                for prompt in prompts:
                    for coeff in coefficients:
                        output = self.generate_steered(prompt, steering_vec, coeff)
                        results.append({
                            "method": "layer_ablation",
                            "layer": layer,
                            "prompt": prompt,
                            "coefficient": coeff,
                            "generated_text": output,
                        })
                        
            except Exception as e:
                logger.warning(f"Layer {layer} failed: {e}")
                continue
                
        # Restore original layer
        self.layer = original_layer
        self._steering_vector = None
        
        logger.info(f"Layer ablation complete: {len(results)} results")
        
        return results
    
    def analyze_layer_ablation(
        self,
        results: List[Dict[str, Any]],
        evaluator=None
    ) -> Dict[str, Any]:
        """
        Analyze layer ablation results to determine optimal layer.
        
        Args:
            results: Results from run_layer_ablation
            evaluator: Optional Evaluator instance (created if not provided)
            
        Returns:
            Analysis dictionary with best layer and statistics
        """
        if evaluator is None:
            from src.evaluation import Evaluator
            evaluator = Evaluator(self.config, use_classifier=False)
        
        df = evaluator.evaluate_batch(results, show_progress=False)
        
        # Compute effectiveness per layer
        # Effectiveness = correlation between coefficient and sentiment score
        layer_effectiveness = {}
        
        for layer in df['layer'].unique():
            layer_df = df[df['layer'] == layer]
            
            if len(layer_df) < 3:
                continue
                
            # Correlation between coefficient and lexicon score
            corr = layer_df['coefficient'].corr(layer_df['lexicon_score'])
            
            # Mean scores at extreme coefficients
            pos_coeff_mean = layer_df[layer_df['coefficient'] > 0]['lexicon_score'].mean()
            neg_coeff_mean = layer_df[layer_df['coefficient'] < 0]['lexicon_score'].mean()
            
            layer_effectiveness[layer] = {
                'correlation': corr,
                'positive_coeff_mean': pos_coeff_mean,
                'negative_coeff_mean': neg_coeff_mean,
                'separation': pos_coeff_mean - neg_coeff_mean,  # How well steering separates
            }
        
        if not layer_effectiveness:
            return {"error": "No valid layer results"}
        
        # Find best layer by separation (difference between positive and negative steering)
        best_layer = max(layer_effectiveness.keys(), 
                         key=lambda l: layer_effectiveness[l]['separation'])
        
        return {
            'best_layer': best_layer,
            'best_layer_stats': layer_effectiveness[best_layer],
            'all_layers': layer_effectiveness,
            'recommendation': f"Layer {best_layer} shows the strongest steering effect "
                             f"(separation={layer_effectiveness[best_layer]['separation']:.4f})"
        }


class SAESteering(BaseSteering):
    """
    Sparse Autoencoder based steering.
    
    Uses SAE decoder directions for more interpretable steering.
    SAE features are often more meaningful than raw activation differences.
    
    Reference: GPT2_SAE_STEERING.ipynb
    """
    
    def __init__(self, model_loader, config):
        """
        Initialize SAE steering.
        
        Args:
            model_loader: ModelLoader instance  
            config: Config object
        """
        super().__init__(model_loader, config)
        self.sae = model_loader.sae
        self._feature_idx: Optional[int] = None
        self._steering_vector: Optional[torch.Tensor] = None
        
    def compute_steering_vector(
        self,
        feature_idx: Optional[int] = None,
        method: str = "decoder"
    ) -> torch.Tensor:
        """
        Compute steering vector from SAE.
        
        Args:
            feature_idx: SAE feature index to use (uses stored if not provided)
            method: "decoder" to use W_dec directly, "activation" to find from text
            
        Returns:
            Steering vector
        """
        # If already computed and no new feature_idx, return cached
        if self._steering_vector is not None and feature_idx is None:
            return self._steering_vector
            
        # Use provided feature_idx or fall back to stored one
        if feature_idx is not None:
            self._feature_idx = feature_idx
        
        if self._feature_idx is None:
            raise ValueError("feature_idx must be provided for SAE steering")
            
        self._steering_vector = self.sae.W_dec[self._feature_idx]
            
        return self._steering_vector
    
    def find_features_for_concept(
        self,
        text: str,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Find SAE features that activate for a given concept.
        
        Args:
            text: Text to analyze (e.g., "I love you so much")
            top_k: Number of top features to return
            
        Returns:
            Tensor of top feature indices
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(text, prepend_bos=False)
            activations = cache[self.sae.cfg.metadata.hook_name]
            encoded = self.sae.encode(activations)
            
        # Get top activating features across all positions
        max_activations, _ = torch.max(encoded[0], dim=0)
        top_features = torch.topk(max_activations, top_k).indices
        
        return top_features
    
    def find_contrastive_features(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find SAE features that differentiate positive from negative concepts.
        
        This method addresses the issue where find_features_for_concept returns
        similar features for both love and hate (because they share semantic space).
        Instead, we find features that DIFFER between concepts.
        
        Args:
            positive_texts: List of positive sentiment texts
            negative_texts: List of negative sentiment texts
            top_k: Number of features to return
            
        Returns:
            Tuple of (positive_specific, negative_specific, differential_features)
        """
        def get_mean_activations(texts: List[str]) -> torch.Tensor:
            """Get mean feature activations across multiple texts."""
            all_activations = []
            with torch.no_grad():
                for text in texts:
                    _, cache = self.model.run_with_cache(text, prepend_bos=False)
                    activations = cache[self.sae.cfg.metadata.hook_name]
                    encoded = self.sae.encode(activations)
                    # Mean across positions
                    mean_act = encoded[0].mean(dim=0)
                    all_activations.append(mean_act)
            # Mean across all texts
            return torch.stack(all_activations).mean(dim=0)
        
        pos_activations = get_mean_activations(positive_texts)
        neg_activations = get_mean_activations(negative_texts)
        
        # Differential activation: features that activate MORE for positive than negative
        differential = pos_activations - neg_activations
        
        # Top features that are MORE active for positive (sentiment steering)
        positive_specific = torch.topk(differential, top_k).indices
        
        # Top features that are MORE active for negative
        negative_specific = torch.topk(-differential, top_k).indices
        
        # Top differential features (highest absolute difference)
        differential_features = torch.topk(differential.abs(), top_k).indices
        
        logger.info(f"Contrastive feature analysis:")
        logger.info(f"  Positive-specific features: {positive_specific.tolist()}")
        logger.info(f"  Negative-specific features: {negative_specific.tolist()}")
        logger.info(f"  Max differential: {differential.max():.4f}, Min: {differential.min():.4f}")
        
        return positive_specific, negative_specific, differential_features
    
    def compute_contrastive_steering_vector(
        self,
        positive_texts: Optional[List[str]] = None,
        negative_texts: Optional[List[str]] = None,
        use_top_k: int = 5,
        method: str = "mean_differential"
    ) -> torch.Tensor:
        """
        Compute steering vector by contrasting positive vs negative SAE activations.
        
        This is more robust than using a single feature because it:
        1. Uses multiple texts to reduce noise
        2. Finds features that DIFFER between concepts (not just activate)
        3. Can ensemble multiple differential features
        
        Args:
            positive_texts: Texts representing positive sentiment
            negative_texts: Texts representing negative sentiment
            use_top_k: How many top differential features to use
            method: "mean_differential" (default) or "single_best"
            
        Returns:
            Steering vector
        """
        # Default texts if not provided
        if positive_texts is None:
            positive_texts = [
                "I love this so much",
                "This is wonderful and amazing",
                "I feel so happy and joyful",
                "This makes me incredibly happy",
                "What a beautiful and lovely experience",
            ]
        if negative_texts is None:
            negative_texts = [
                "I hate this so much",
                "This is terrible and awful",
                "I feel so sad and angry",
                "This makes me incredibly upset",
                "What a horrible and disgusting experience",
            ]
        
        pos_features, neg_features, diff_features = self.find_contrastive_features(
            positive_texts, negative_texts, top_k=use_top_k
        )
        
        if method == "single_best":
            # Use the single most differential feature
            best_feature = pos_features[0].item()
            self._feature_idx = best_feature
            self._steering_vector = self.sae.W_dec[best_feature]
            logger.info(f"Using single best positive feature: {best_feature}")
        else:
            # Mean of top positive-specific decoder directions
            top_pos_decoders = self.sae.W_dec[pos_features]
            self._steering_vector = top_pos_decoders.mean(dim=0)
            self._feature_idx = pos_features[0].item()  # Store primary for logging
            logger.info(f"Using mean of top {use_top_k} differential features")
        
        # Normalize
        self._steering_vector = self._steering_vector / self._steering_vector.norm()
        
        return self._steering_vector
    
    def find_features_by_embedding(
        self,
        word: str,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Find SAE features by similarity to word embedding.
        
        Args:
            word: Word to find features for
            top_k: Number of top features
            
        Returns:
            Tensor of top feature indices
        """
        token_ids = self.model.tokenizer.encode(word)
        embedding = self.model.W_E[token_ids][0]
        
        # Compute similarity between word embedding and SAE decoder directions
        similarities = self.sae.W_dec @ embedding
        top_features = torch.topk(similarities, top_k).indices
        
        return top_features
    
    def get_hook_name(self) -> str:
        """Return SAE hook name."""
        return self.sae.cfg.metadata.hook_name
    
    def get_layer_info(self) -> str:
        """Return layer information."""
        return f"sae_layer_{self.config.sae_layer}_feature_{self._feature_idx}"
    
    def _create_steering_hook(
        self,
        steering_vector: torch.Tensor,
        coefficient: float
    ) -> Callable:
        """
        Create SAE-specific steering hook.
        
        Only modifies the last token position.
        """
        def hook(activation, hook, vec=steering_vector, coeff=coefficient):
            activation[:, -1, :] += coeff * vec
            return activation
        return hook
    
    def get_neuronpedia_urls(self, feature_indices: torch.Tensor) -> List[str]:
        """
        Get Neuronpedia URLs for feature interpretation.
        
        Args:
            feature_indices: SAE feature indices
            
        Returns:
            List of Neuronpedia URLs
        """
        try:
            from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
            return get_neuronpedia_quick_list(self.sae, feature_indices.tolist())
        except ImportError:
            logger.warning("Neuronpedia integration not available")
            return []
