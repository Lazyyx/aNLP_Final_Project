# aNLP Project - Steering GPT-2

**Authors:** Quentin Galbez, Léo Lopes, Yanis Martin, Baptiste Villeneuve  
**Course:** Advanced NLP - EPITA SCIA 2026

A modular framework for activation steering in GPT-2, comparing Basic Activation Steering vs SAE-based Steering with rigorous statistical evaluation.

## Key Results

| Method | Classifier Correlation | P-value | Effect Size | Significant? |
|--------|----------------------|---------|-------------|--------------|
| **Basic Activation** | +0.2163 | 0.00125 | 0.8377 (large) | ✅ YES |
| **SAE Steering** | +0.1492 | 0.0269 | 0.5868 (medium) | ✅ YES |

**Conclusion:** Basic Activation Steering outperforms SAE Steering with stronger correlation and larger effect size. Both methods achieve statistical significance (p < 0.05).

## Project Structure

```
aNLP_Final_Project/
├── src/                      # Main source code
│   ├── __init__.py          # Package exports
│   ├── config.py            # Centralized configuration
│   ├── models.py            # Model loading utilities
│   ├── steering.py          # Steering method implementations
│   ├── evaluation.py        # Evaluation metrics & statistical tests
│   └── visualization.py     # Plotting utilities
├── run_experiments.py        # Main experiment runner
├── basic_activation.ipynb    # Original notebook (reference)
├── GPT2_SAE_STEERING.ipynb   # Original notebook (reference)
├── evaluation.ipynb          # Original notebook (reference)
├── results/                  # Experiment outputs (JSON)
├── pyproject.toml            # Project configuration
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Lazyyx/aNLP_Final_Project.git
cd aNLP_Final_Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### Run All Experiments
```bash
python run_experiments.py
```

### Run Specific Experiments
```bash
# Skip SAE (faster, no SAE download needed)
python run_experiments.py --no-sae

# Skip classifier (faster, no transformer download)
python run_experiments.py --no-classifier

# Only run layer ablation
python run_experiments.py --no-basic --no-sae
```

### Use as Library
```python
from src import Config, ModelLoader, ActivationSteering, Evaluator

# Initialize
config = Config()
model_loader = ModelLoader(config)
evaluator = Evaluator(config)

# Basic steering
steering = ActivationSteering(model_loader, config)
results = steering.run_experiment()

# Evaluate
df = evaluator.evaluate_batch(results)
print(evaluator.generate_report(results, "Basic Steering"))
```

## Steering Methods

### 1. Basic Activation Steering (Baseline)
Computes steering vector as the difference between activations for "Love" and "Hate":
```
steering_vector = activation("Love") - activation("Hate")
```

### 2. SAE-based Steering
Uses Sparse Autoencoder decoder directions for more interpretable steering. 

**Improved Contrastive Feature Selection:** Instead of finding features that activate highly for a concept, we find features that *differentiate* between positive and negative concepts:
```python
differential = pos_activations - neg_activations
top_features = topk(differential, k=5)  # Features unique to positive sentiment
```

## Evaluation Metrics

### Lexicon-based Score (Baseline)
Simple keyword matching:
```
score = (love_words - hate_words) / total_words
```

### Transformer Classifier (Primary)
Fine-tuned RoBERTa sentiment model for robust evaluation:
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Returns continuous polarity score from -1 to +1

### Statistical Testing
- **Pearson correlation** between steering coefficient and sentiment score
- **P-value** for significance testing (α = 0.05)
- **Cohen's d** effect size (small < 0.2, medium < 0.8, large ≥ 0.8)

## Configuration

All settings are centralized in `src/config.py`:

```python
@dataclass
class Config:
    model_name: str = "gpt2"
    default_layer: int = 3  # Validated by layer ablation study
    steering_coefficients: List[float] = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
    test_prompts: List[str] = [...]  # 20 standardized prompts (4 categories)
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

## Addressing Teacher Feedback

| Feedback | Solution | Implementation |
|----------|----------|----------------|
| "Which test set?" | Standardized 20 prompts across all methods | `Config.test_prompts` |
| "Which metrics?" | Lexicon + RoBERTa classifier + statistics | `evaluation.py` |
| "Fine-tuned encoder" | RoBERTa sentiment classifier | `SentimentClassifier` class |
| "Why this layer?" | Layer ablation study (all 12 layers) | `run_layer_ablation()` |
| "Statistical significance" | Pearson correlation + p-value + Cohen's d | `compute_statistical_metrics()` |

### Layer Ablation Results

| Layer | Correlation | Separation |
|-------|-------------|------------|
| 0 | +0.0405 | +0.0027 |
| 1 | +0.2548 | +0.0204 |
| 2 | -0.2523 | -0.0160 |
| **3** | **+0.4177** | **+0.0387** ← Best |
| 6 | +0.4239 | +0.0377 |
| 11 | +0.0961 | +0.0074 |

**Recommendation:** Layer 3 selected as default based on highest separation score.

## Results

Results are automatically saved to `results/` with timestamps (JSON format).

### Sample Outputs

**Coefficient: -20 (negative steering)**
```
Prompt: "I think the weather is"
Output: "I think the weather is going to be a bit more intense than the last few days..."
Classifier: -0.457 (negative)
```

**Coefficient: +20 (positive steering)**
```
Prompt: "I think the weather is"
Output: "I think the weather is good. I think the weather is good. I think the weather is good..."
Classifier: +0.980 (positive)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- TransformerLens >= 2.0.0
- SAE-Lens >= 3.0.0
- transformers (for sentiment classifier)

## License

MIT License