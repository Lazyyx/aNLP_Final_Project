"""
Evaluation module for steering experiments.

Contains multiple evaluation metrics addressing teacher feedback:
1. Lexicon-based scoring (baseline)
2. Fine-tuned sentiment classifier (recommended by teacher)
3. Statistical significance testing
4. Aggregate metrics for comparison

All methods use the SAME test set for fair comparison.
"""
import json
import ast
import os
import sys
import re 

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import numpy as np
import pandas as pd
import logging
from scipy import stats
import warnings
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    text: str
    lexicon_score: float
    classifier_score: Optional[float]
    classifier_label: Optional[str]
    classifier_confidence: Optional[float]
    love_word_count: int
    hate_word_count: int
    total_words: int


@dataclass
class StatisticalMetrics:
    """Container for statistical analysis results."""
    correlation: float
    p_value: float
    mean_shift: float  # Difference between max and min coefficient means
    effect_size: float  # Cohen's d
    is_significant: bool


class LexiconEvaluator:
    """
    Baseline lexicon-based sentiment scoring.
    
    Simple keyword matching approach - serves as baseline
    but has known limitations (no context understanding).
    """
    
    def __init__(self, config):
        """
        Initialize lexicon evaluator.
        
        Args:
            config: Config object with word lists
        """
        self.love_words = config.love_words
        self.hate_words = config.hate_words
        
    def score(self, text: str) -> Tuple[float, int, int, int]:
        """
        Calculate lexicon-based sentiment score.
        
        Score = (love_count - hate_count) / total_words
        Range: approximately -1 to 1
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (score, love_count, hate_count, total_words)
        """
        if not text:
            return 0.0, 0, 0, 0
            
        # Normalize text
        words = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0, 0, 0, 0
            
        love_count = sum(1 for w in words if w in self.love_words)
        hate_count = sum(1 for w in words if w in self.hate_words)
        
        score = (love_count - hate_count) / total_words
        
        return score, love_count, hate_count, total_words


class SentimentClassifier:
    """
    Fine-tuned transformer-based sentiment classifier.
    
    Addresses teacher feedback:
    "you might also consider a finetuned encoder 
    (your task is very similar to sentiment analysis)"
    
    Uses a pre-trained RoBERTa model fine-tuned on sentiment data.
    """
    
    # Label mapping for different model formats
    LABEL_MAPPING = {
        # Twitter RoBERTa format
        'positive': 1.0,
        'negative': -1.0,
        'neutral': 0.0,
        # Alternative formats
        'POSITIVE': 1.0,
        'NEGATIVE': -1.0,
        'NEUTRAL': 0.0,
        'POS': 1.0,
        'NEG': -1.0,
        'NEU': 0.0,
        # 5-star format
        '1 star': -1.0,
        '2 stars': -0.5,
        '3 stars': 0.0,
        '4 stars': 0.5,
        '5 stars': 1.0,
    }
    
    def __init__(self, config, model_name: Optional[str] = None):
        """
        Initialize sentiment classifier.
        
        Args:
            config: Config object
            model_name: HuggingFace model name (default: from config)
        """
        self.config = config
        self.model_name = model_name or config.sentiment_model
        self._pipeline = None
        self._loaded = False
        self._load_error = None
        
    def _load_model(self):
        """Lazy load the sentiment model."""
        if self._loaded:
            return
        if self._load_error:
            raise self._load_error
            
        try:
            from transformers import pipeline
            
            logger.info(f"Loading sentiment classifier: {self.model_name}")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Return all class scores
            )
            self._loaded = True
            logger.info("Sentiment classifier loaded successfully")
            
        except ImportError as e:
            self._load_error = ImportError(
                "transformers is required for sentiment classification. "
                "Install with: pip install transformers"
            )
            raise self._load_error
        except Exception as e:
            self._load_error = e
            logger.error(f"Failed to load classifier: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if classifier can be loaded."""
        try:
            self._load_model()
            return True
        except Exception:
            return False
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict sentiment for text.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (label, confidence, all_scores)
        """
        self._load_model()
        
        # Truncate if too long (RoBERTa max length)
        text = text[:512] if text else "neutral"
        
        try:
            results = self._pipeline(text)
            
            # Handle different output formats
            if isinstance(results[0], list):
                results = results[0]
                
            # Get top prediction
            top = max(results, key=lambda x: x['score'])
            all_scores = {r['label']: r['score'] for r in results}
            
            return top['label'], top['score'], all_scores
            
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return "neutral", 0.5, {}
    
    def get_polarity_score(self, text: str) -> Tuple[float, float]:
        """
        Get a continuous polarity score from -1 (negative) to 1 (positive).
        
        Maps classifier outputs to a continuous scale for easier comparison.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (polarity_score, confidence) where:
            - polarity_score: Float from -1 to 1
            - confidence: Confidence in the prediction (0-1)
        """
        self._load_model()
        
        label, confidence, scores = self.predict(text)
        
        # Calculate weighted polarity score
        polarity = 0.0
        total_weight = 0.0
        
        for lbl, score in scores.items():
            if lbl in self.LABEL_MAPPING:
                polarity += self.LABEL_MAPPING[lbl] * score
                total_weight += score
        
        if total_weight > 0:
            polarity = polarity / total_weight
        else:
            # Fallback: use label directly
            polarity = self.LABEL_MAPPING.get(label, 0.0)
        
        return polarity, confidence


class Evaluator:
    """
    Combined evaluator using multiple metrics.
    
    Ensures consistent evaluation across all steering methods
    (addressing teacher feedback on metrics).
    
    Includes:
    - Lexicon-based scoring (baseline)
    - Transformer classifier scoring
    - Statistical significance testing
    - Effect size computation
    """
    
    def __init__(self, config, use_classifier: bool = True):
        """
        Initialize combined evaluator.
        
        Args:
            config: Config object
            use_classifier: Whether to use transformer classifier
        """
        self.config = config
        self.lexicon = LexiconEvaluator(config)
        self._use_classifier = use_classifier
        self._classifier = None
        
    @property
    def classifier(self) -> Optional[SentimentClassifier]:
        """Lazy load classifier only when needed."""
        if self._use_classifier and self._classifier is None:
            self._classifier = SentimentClassifier(self.config)
        return self._classifier
        
    def evaluate_single(self, text: str) -> EvaluationResult:
        """
        Evaluate a single text with all metrics.
        
        Args:
            text: Text to evaluate
            
        Returns:
            EvaluationResult with all scores
        """
        # Lexicon scoring
        lex_score, love_count, hate_count, total = self.lexicon.score(text)
        
        # Classifier scoring
        clf_score = None
        clf_label = None
        clf_confidence = None
        
        if self.classifier:
            try:
                clf_label, clf_confidence, _ = self.classifier.predict(text)
                clf_score, _ = self.classifier.get_polarity_score(text)
            except Exception as e:
                logger.warning(f"Classifier error: {e}")
                
        return EvaluationResult(
            text=text,
            lexicon_score=lex_score,
            classifier_score=clf_score,
            classifier_label=clf_label,
            classifier_confidence=clf_confidence,
            love_word_count=love_count,
            hate_word_count=hate_count,
            total_words=total
        )
    
    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate a batch of steering results.
        
        Args:
            results: List of dicts with 'generated_text' and metadata
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with all results and evaluation metrics
        """
        evaluated = []
        
        iterator = results
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(results, desc="Evaluating")
            except ImportError:
                pass
        
        for r in iterator:
            text = r.get('generated_text', '')
            eval_result = self.evaluate_single(text)
            
            evaluated.append({
                **r,
                'lexicon_score': eval_result.lexicon_score,
                'classifier_score': eval_result.classifier_score,
                'classifier_label': eval_result.classifier_label,
                'classifier_confidence': eval_result.classifier_confidence,
                'love_words': eval_result.love_word_count,
                'hate_words': eval_result.hate_word_count,
            })
            
        return pd.DataFrame(evaluated)
    
    def compute_statistical_metrics(
        self,
        df: pd.DataFrame,
        score_column: str = 'lexicon_score'
    ) -> StatisticalMetrics:
        """
        Compute comprehensive statistical metrics.
        
        Addresses teacher feedback on metrics by providing:
        - Pearson correlation with p-value
        - Effect size (Cohen's d)
        - Significance testing
        
        Args:
            df: DataFrame with results
            score_column: Column to analyze
            
        Returns:
            StatisticalMetrics dataclass
        """
        if 'coefficient' not in df.columns or score_column not in df.columns:
            return StatisticalMetrics(
                correlation=0.0, p_value=1.0, mean_shift=0.0,
                effect_size=0.0, is_significant=False
            )
        
        # Filter valid data
        valid_df = df[[score_column, 'coefficient']].dropna()
        
        if len(valid_df) < 3:
            return StatisticalMetrics(
                correlation=0.0, p_value=1.0, mean_shift=0.0,
                effect_size=0.0, is_significant=False
            )
        
        # Pearson correlation with p-value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation, p_value = stats.pearsonr(
                valid_df['coefficient'],
                valid_df[score_column]
            )
        
        # Mean shift between extreme coefficients
        coeff_groups = valid_df.groupby('coefficient')[score_column]
        coeff_means = coeff_groups.mean()
        
        max_coeff = coeff_means.index.max()
        min_coeff = coeff_means.index.min()
        mean_shift = coeff_means[max_coeff] - coeff_means[min_coeff]
        
        # Effect size (Cohen's d) between extreme coefficient groups
        max_group = valid_df[valid_df['coefficient'] == max_coeff][score_column]
        min_group = valid_df[valid_df['coefficient'] == min_coeff][score_column]
        
        pooled_std = np.sqrt(
            ((len(max_group) - 1) * max_group.std()**2 + 
             (len(min_group) - 1) * min_group.std()**2) /
            (len(max_group) + len(min_group) - 2)
        )
        
        effect_size = mean_shift / pooled_std if pooled_std > 0 else 0.0
        
        # Significance at α = 0.05
        is_significant = p_value < 0.05
        
        return StatisticalMetrics(
            correlation=correlation,
            p_value=p_value,
            mean_shift=mean_shift,
            effect_size=effect_size,
            is_significant=is_significant
        )
    
    def compute_aggregate_metrics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics for analysis.
        
        Args:
            df: DataFrame from evaluate_batch
            
        Returns:
            Dictionary of aggregate statistics
        """
        metrics = {}
        
        # Statistical metrics for both scoring methods
        metrics['lexicon_stats'] = self.compute_statistical_metrics(df, 'lexicon_score')
        
        if 'classifier_score' in df.columns and df['classifier_score'].notna().any():
            metrics['classifier_stats'] = self.compute_statistical_metrics(df, 'classifier_score')
        
        # Group by coefficient
        if 'coefficient' in df.columns:
            coeff_groups = df.groupby('coefficient')
            
            metrics['by_coefficient'] = {
                'lexicon_mean': coeff_groups['lexicon_score'].mean().to_dict(),
                'lexicon_std': coeff_groups['lexicon_score'].std().to_dict(),
                'lexicon_sem': coeff_groups['lexicon_score'].sem().to_dict(),  # Standard error
                'count': coeff_groups.size().to_dict(),
            }
            
            if 'classifier_score' in df.columns and df['classifier_score'].notna().any():
                metrics['by_coefficient']['classifier_mean'] = coeff_groups['classifier_score'].mean().to_dict()
                metrics['by_coefficient']['classifier_std'] = coeff_groups['classifier_score'].std().to_dict()
                metrics['by_coefficient']['classifier_sem'] = coeff_groups['classifier_score'].sem().to_dict()
        
        # Correlation between lexicon and classifier
        if 'classifier_score' in df.columns and df['classifier_score'].notna().any():
            valid = df[['lexicon_score', 'classifier_score']].dropna()
            if len(valid) > 2:
                corr, p = stats.pearsonr(valid['lexicon_score'], valid['classifier_score'])
                metrics['lexicon_classifier_correlation'] = {
                    'correlation': corr,
                    'p_value': p
                }
                
        return metrics
    
    def analyze_by_prompt_category(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze results by prompt category (neutral, positive-leaning, etc.).
        
        Useful for understanding how steering affects different types of inputs.
        
        Args:
            df: DataFrame with results
            
        Returns:
            DataFrame with category-level statistics
        """
        if not hasattr(self.config, 'prompt_categories'):
            return pd.DataFrame()
        
        # Map prompts to categories
        prompt_to_category = {}
        for category, indices in self.config.prompt_categories.items():
            for idx in indices:
                if idx < len(self.config.test_prompts):
                    prompt_to_category[self.config.test_prompts[idx]] = category
        
        df = df.copy()
        df['category'] = df['prompt'].map(prompt_to_category)
        
        # Compute stats by category and coefficient
        category_stats = df.groupby(['category', 'coefficient']).agg({
            'lexicon_score': ['mean', 'std', 'count']
        }).round(4)
        
        return category_stats
    
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        method_name: str,
        include_samples: bool = True
    ) -> str:
        """
        Generate a comprehensive text report for steering results.
        
        Addresses teacher feedback on metrics documentation.
        
        Args:
            results: Steering experiment results
            method_name: Name of the steering method
            include_samples: Whether to include sample outputs
            
        Returns:
            Formatted report string
        """
        df = self.evaluate_batch(results, show_progress=False)
        metrics = self.compute_aggregate_metrics(df)
        
        report = [
            "=" * 70,
            f"  EVALUATION REPORT: {method_name}",
            "=" * 70,
            "",
            f"Experiment Configuration:",
            f"  • Total samples: {len(df)}",
            f"  • Unique prompts: {df['prompt'].nunique() if 'prompt' in df.columns else 'N/A'}",
            f"  • Coefficients tested: {sorted(df['coefficient'].unique().tolist()) if 'coefficient' in df.columns else 'N/A'}",
            "",
        ]
        
        # Lexicon-based metrics
        report.append("-" * 70)
        report.append("LEXICON-BASED EVALUATION (Baseline)")
        report.append("-" * 70)
        
        if 'lexicon_stats' in metrics:
            stats = metrics['lexicon_stats']
            report.append(f"  Correlation (coeff → score): {stats.correlation:+.4f}")
            report.append(f"  P-value:                     {stats.p_value:.2e}")
            report.append(f"  Statistical significance:    {'YES ✓' if stats.is_significant else 'NO ✗'} (α=0.05)")
            report.append(f"  Mean shift (max-min coeff):  {stats.mean_shift:+.4f}")
            report.append(f"  Effect size (Cohen's d):     {stats.effect_size:+.4f}")
            
            # Interpret effect size
            if abs(stats.effect_size) < 0.2:
                effect_interp = "negligible"
            elif abs(stats.effect_size) < 0.5:
                effect_interp = "small"
            elif abs(stats.effect_size) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            report.append(f"  Effect interpretation:       {effect_interp}")
        
        # Classifier-based metrics
        if 'classifier_stats' in metrics:
            report.append("")
            report.append("-" * 70)
            report.append("CLASSIFIER-BASED EVALUATION (RoBERTa Sentiment)")
            report.append("-" * 70)
            
            stats = metrics['classifier_stats']
            report.append(f"  Correlation (coeff → score): {stats.correlation:+.4f}")
            report.append(f"  P-value:                     {stats.p_value:.2e}")
            report.append(f"  Statistical significance:    {'YES ✓' if stats.is_significant else 'NO ✗'} (α=0.05)")
            report.append(f"  Mean shift (max-min coeff):  {stats.mean_shift:+.4f}")
            report.append(f"  Effect size (Cohen's d):     {stats.effect_size:+.4f}")
            
            # Lexicon-classifier correlation
            if 'lexicon_classifier_correlation' in metrics:
                lcc = metrics['lexicon_classifier_correlation']
                report.append("")
                report.append(f"  Lexicon ↔ Classifier corr:   {lcc['correlation']:+.4f} (p={lcc['p_value']:.2e})")
        
        # Detailed breakdown by coefficient
        report.append("")
        report.append("-" * 70)
        report.append("BREAKDOWN BY STEERING COEFFICIENT")
        report.append("-" * 70)
        
        if 'by_coefficient' in metrics:
            header = "  Coeff  |  Lexicon Mean ± SEM  |  Classifier Mean ± SEM  |  N"
            report.append(header)
            report.append("  " + "-" * 65)
            
            coeffs = sorted(metrics['by_coefficient']['lexicon_mean'].keys())
            for coeff in coeffs:
                lex_mean = metrics['by_coefficient']['lexicon_mean'][coeff]
                lex_sem = metrics['by_coefficient'].get('lexicon_sem', {}).get(coeff, 0)
                n = metrics['by_coefficient'].get('count', {}).get(coeff, 0)
                
                line = f"  {coeff:+5.1f}  |  {lex_mean:+.4f} ± {lex_sem:.4f}   |"
                
                if 'classifier_mean' in metrics['by_coefficient']:
                    clf_mean = metrics['by_coefficient']['classifier_mean'].get(coeff, float('nan'))
                    clf_sem = metrics['by_coefficient'].get('classifier_sem', {}).get(coeff, 0)
                    line += f"  {clf_mean:+.4f} ± {clf_sem:.4f}      |"
                else:
                    line += "        N/A              |"
                    
                line += f"  {n}"
                report.append(line)
        
        # Sample outputs
        if include_samples and len(df) > 0:
            report.append("")
            report.append("-" * 70)
            report.append("SAMPLE OUTPUTS")
            report.append("-" * 70)
            
            # Show one sample per extreme coefficient
            if 'coefficient' in df.columns:
                for coeff in [df['coefficient'].min(), 0.0, df['coefficient'].max()]:
                    subset = df[df['coefficient'] == coeff]
                    if len(subset) > 0:
                        sample = subset.iloc[0]
                        report.append(f"\n  Coefficient: {coeff:+.1f}")
                        report.append(f"  Prompt: \"{sample.get('prompt', 'N/A')}\"")
                        report.append(f"  Output: \"{sample.get('generated_text', 'N/A')[:100]}...\"")
                        report.append(f"  Lexicon: {sample.get('lexicon_score', 0):+.4f}, Classifier: {sample.get('classifier_score', 'N/A')}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

# pour évaluer sémantique + sentiment
class LoveHateEvaluator:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print(f"Initialisation sur {'GPU' if device==0 else 'CPU'}...")
        # On évite de recharger si déjà en mémoire (utile en notebook)
        self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            top_k=None
        )

    def evaluate(self, prompt, generated_text):
        if not generated_text: return 0.0, 0.0
        
        # Sémantique
        emb1 = self.sim_model.encode(prompt, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.sim_model.encode(generated_text, convert_to_tensor=True, show_progress_bar=False)
        semantic_score = util.cos_sim(emb1, emb2).item()

        # Sentiment
        results = self.sentiment_pipe(generated_text[:512])[0]
        pos = next(x['score'] for x in results if x['label'] == 'positive')
        neg = next(x['score'] for x in results if x['label'] == 'negative')
        sentiment_score = pos - neg

        return {"semantic_score": round(semantic_score, 4), "sentiment_score": round(sentiment_score, 4)}

# Fonction pour nettoyer les strings corrompues
def clean_python_string(s):
    if "'dataframe':" in s:
        # On garde tout ce qui est AVANT 'dataframe':
        # On suppose que la structure est { ... , 'dataframe': ... }
        # On coupe au dernier séparateur virgule avant dataframe
        s_clean = s.split("'dataframe':")[0]
        s_clean = s_clean.strip()
        if s_clean.endswith(','):
            s_clean = s_clean[:-1] # Enlève la virgule finale
        
        # On referme l'accolade si besoin
        if not s_clean.endswith('}'):
            s_clean += '}'
            
        return s_clean
    return s

def load_and_fix_json(filepath):
    print(f"Lecture du fichier : {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    clean_data = {}
    for key, value_str in raw_data.items():
        if isinstance(value_str, str):
            try:
                # 1. Tentative normale
                parsed_val = ast.literal_eval(value_str)
                clean_data[key] = parsed_val
            except Exception:
                # 2. TENTATIVE DE SAUVETAGE
                try:
                    # On nettoie la chaîne
                    repaired_str = clean_python_string(value_str)
                    parsed_val = ast.literal_eval(repaired_str)
                    clean_data[key] = parsed_val
                    print(f" Clé '{key}' réparée avec succès (Dataframe supprimé).")
                except Exception as e:
                    print(f" Clé '{key}' irrécupérable : {e}")
        else:
            clean_data[key] = value_str
            
    return clean_data

def extract_timestamp_from_filename(filename):
    # Cherche un motif YYYYMMDD_HHMMSS (ex: 20260112_152928)
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d_%H%M%S") # Fallback si pas trouvé

def run_evaluation_result():
    print("Lancement de l'évaluation des résultats...")
    results_dir = "results"
    # Création du dossier results s'il n'existe pas (juste au cas où)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Dossier results/ vide.")
        return

    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.json')]
    if not files:
        print("Aucun fichier JSON trouvé.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f" Traitement de : {latest_file}")
    file_id = extract_timestamp_from_filename(latest_file)
    data = load_and_fix_json(latest_file)
    evaluator = LoveHateEvaluator()
    
    methods = ['basic', 'sae', 'ablation']
    
    for method in methods:
        filename = f"results/EVAL_LOVE_HATE_{method}_{file_id}.csv"

        # On vérifie si le fichier existe déjà
        if os.path.exists(filename):
            print(f"\n  Méthode '{method.upper()}' déjà évaluée.")
            print(f"    Fichier existant : {filename}")
            print("    On passe au suivant.")
            continue # On saute cette itération de la boucle
        
        # Sécurité : on vérifie que la méthode existe et que c'est bien un dictionnaire
        if method in data and isinstance(data[method], dict) and 'results' in data[method]:
            print(f"\nMéthode : {method.upper()}")
            results_list = data[method]['results']
            
            enriched = []
            print(f"   Calcul sur {len(results_list)} phrases...")
            
            for i, row in enumerate(results_list):
                prompt = row.get('prompt', '')
                text = row.get('generated_text', '')
                
                scores = evaluator.evaluate(prompt, text)
                
                new_row = row.copy()
                new_row.update(scores)
                new_row['length'] = len(text)
                enriched.append(new_row)
                
                if i % 50 == 0 and i > 0: print(f"   ... {i} faits")
            
            df = pd.DataFrame(enriched)
            
            df.to_csv(filename, index=False)
            print(f"Sauvegardé : {filename}")
        else:
            # Si on passe ici, c'est que la clé n'existe pas ou que le parsing a échoué
            if method in data and not isinstance(data[method], dict):
                 print(f"Méthode '{method}' ignorée (Format de données invalide).")

    print("\nÉvaluation terminée.")