"""
Data loading utilities for test sets and datasets.
"""

from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_love_hate_dataset(
    dataset_name: str = "dair-ai/emotion",
    split: str = "test",
    max_samples: int = 100
) -> Tuple[List[str], List[str]]:
    """
    Load a dataset with love/hate or positive/negative labels.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        max_samples: Maximum samples per class
        
    Returns:
        Tuple of (love_texts, hate_texts)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Common label mappings
    love_labels = {'love', 'joy', 'positive', 1, '1'}
    hate_labels = {'anger', 'sadness', 'fear', 'negative', 0, '0'}
    
    love_texts = []
    hate_texts = []
    
    for item in dataset:
        label = item.get('label', item.get('sentiment', None))
        text = item.get('text', item.get('sentence', ''))
        
        if label in love_labels and len(love_texts) < max_samples:
            love_texts.append(text)
        elif label in hate_labels and len(hate_texts) < max_samples:
            hate_texts.append(text)
            
        if len(love_texts) >= max_samples and len(hate_texts) >= max_samples:
            break
    
    logger.info(f"Loaded {len(love_texts)} love and {len(hate_texts)} hate samples")
    return love_texts, hate_texts


def create_test_prompts(
    base_prompts: Optional[List[str]] = None
) -> List[str]:
    """
    Create standardized test prompts.
    
    Args:
        base_prompts: Optional list of custom prompts
        
    Returns:
        List of test prompts
    """
    default_prompts = [
        # Neutral starting points
        "I think dogs are",
        "My opinion about this movie is that it's",
        "When I see you, I feel",
        "This restaurant makes me",
        "The weather today makes me feel",
        
        # Relationship contexts
        "My relationship with my family is",
        "When I think about my friends, I",
        "Meeting new people makes me",
        
        # Life/work contexts
        "I believe that life is",
        "Working on this project makes me",
        "When I think about the future, I",
        
        # Direct sentiment triggers
        "This person is absolutely",
        "The situation right now is",
        "I would describe my day as",
    ]
    
    return base_prompts or default_prompts


def get_standardized_test_set() -> Dict[str, List[str]]:
    """
    Get the standardized test set for fair comparison.
    
    This ensures ALL experiments use the SAME prompts.
    (Addressing teacher feedback on consistent test sets)
    
    Returns:
        Dictionary with test configuration
    """
    return {
        "prompts": create_test_prompts(),
        "coefficients": [-10, -5, -2, -1, 0, 1, 2, 5, 10],
        "version": "1.0",
        "description": "Standardized test set for love/hate steering comparison"
    }
