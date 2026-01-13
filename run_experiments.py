"""
Main experiment runner for steering comparison.

This script runs all steering methods with the SAME test set
and evaluation metrics for fair comparison.
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from src.config import Config
from src.models import ModelLoader
from src.steering import ActivationSteering, SAESteering
from src.evaluation import Evaluator
from src.visualization import run_visualizations
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_basic_steering_experiment(
    model_loader: ModelLoader,
    config: Config,
    evaluator: Evaluator
) -> dict:
    """
    Run basic activation steering experiment.
    
    Args:
        model_loader: ModelLoader instance
        config: Config object
        evaluator: Evaluator instance
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 60)
    logger.info("Running Basic Activation Steering Experiment")
    logger.info("=" * 60)
    
    steering = ActivationSteering(model_loader, config)
    results = steering.run_experiment()
    
    # Evaluate
    df = evaluator.evaluate_batch(results)
    report = evaluator.generate_report(results, "Basic Activation Steering")
    print(report)
    
    return {
        "method": "basic_activation",
        "results": results,
        "dataframe": df,
        "report": report
    }


def run_layer_ablation(
    model_loader: ModelLoader,
    config: Config,
    evaluator: Evaluator
) -> dict:
    """
    Run comprehensive layer ablation study.
    
    Addresses teacher feedback: "Did you try different layers?"
    "Why this layer in particular?"
    
    Tests all 12 GPT-2 layers to find optimal steering position.
    
    Args:
        model_loader: ModelLoader instance
        config: Config object
        evaluator: Evaluator instance
        
    Returns:
        Dictionary with ablation results and analysis
    """
    logger.info("=" * 60)
    logger.info("Running Layer Ablation Study")
    logger.info("This addresses teacher feedback: 'Why this layer in particular?'")
    logger.info("=" * 60)
    
    steering = ActivationSteering(model_loader, config)
    results = steering.run_layer_ablation()
    
    # Evaluate
    df = evaluator.evaluate_batch(results, show_progress=True)
    
    # Analyze which layer works best
    analysis = steering.analyze_layer_ablation(results, evaluator)
    
    logger.info(f"\n{'='*60}")
    logger.info("LAYER ABLATION RESULTS")
    logger.info(f"{'='*60}")
    
    # Print all layer scores
    logger.info("\nLayer-by-Layer Effectiveness (correlation & separation):")
    logger.info("-" * 50)
    
    for layer in sorted(analysis.get('all_layers', {}).keys()):
        stats = analysis['all_layers'][layer]
        marker = " ← BEST" if layer == analysis.get('best_layer') else ""
        logger.info(
            f"  Layer {layer:2d}: correlation={stats['correlation']:+.4f}, "
            f"separation={stats['separation']:+.4f}{marker}"
        )
    
    logger.info("-" * 50)
    logger.info(f"\nRECOMMENDATION: {analysis.get('recommendation', 'N/A')}")
    logger.info(f"{'='*60}\n")
    
    return {
        "method": "layer_ablation",
        "results": results,
        "dataframe": df,
        "analysis": analysis,
        "best_layer": analysis.get('best_layer'),
        "layer_scores": analysis.get('all_layers', {})
    }


def run_sae_steering_experiment(
    model_loader: ModelLoader,
    config: Config,
    evaluator: Evaluator,
    use_contrastive: bool = True  # Use improved contrastive feature selection
) -> dict:
    """
    Run SAE-based steering experiment.
    
    Args:
        model_loader: ModelLoader instance
        config: Config object
        evaluator: Evaluator instance
        use_contrastive: Use improved contrastive feature selection (recommended)
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 60)
    logger.info("Running SAE Steering Experiment")
    logger.info("=" * 60)
    
    steering = SAESteering(model_loader, config)
    
    if use_contrastive:
        # NEW: Use contrastive feature selection for better sentiment steering
        logger.info("Using CONTRASTIVE feature selection (improved method)...")
        logger.info("This finds features that DIFFER between positive/negative sentiment")
        
        steering.compute_contrastive_steering_vector(
            positive_texts=[
                "I love this so much",
                "This is wonderful and amazing", 
                "I feel so happy and joyful",
                "This makes me incredibly happy",
                "What a beautiful and lovely experience",
            ],
            negative_texts=[
                "I hate this so much",
                "This is terrible and awful",
                "I feel so sad and angry", 
                "This makes me incredibly upset",
                "What a horrible and disgusting experience",
            ],
            use_top_k=5,
            method="mean_differential"
        )
        feature_info = f"contrastive_top5_mean"
    else:
        # Old method: single feature (kept for comparison)
        logger.info("Finding love-related SAE features (old method)...")
        love_features = steering.find_features_for_concept("I love you I love you so much")
        hate_features = steering.find_features_for_concept("I hate you I hate you so much")
        
        logger.info(f"Top love features: {love_features.tolist()}")
        logger.info(f"Top hate features: {hate_features.tolist()}")
        
        # Use a feature that appears in love but not hate (if any)
        love_set = set(love_features.tolist())
        hate_set = set(hate_features.tolist())
        unique_love = love_set - hate_set
        
        if unique_love:
            feature_idx = list(unique_love)[0]
            logger.info(f"Using unique love feature: {feature_idx}")
        else:
            feature_idx = love_features[0].item()
            logger.warning(f"No unique love features found, using {feature_idx}")
        
        steering.compute_steering_vector(feature_idx=feature_idx)
        feature_info = f"feature_{feature_idx}"
    
    results = steering.run_experiment()
    
    # Evaluate
    df = evaluator.evaluate_batch(results)
    report = evaluator.generate_report(results, f"SAE Steering ({feature_info})")
    print(report)
    
    return {
        "method": "sae_steering",
        "feature_info": feature_info,
        "contrastive_method": use_contrastive,
        "results": results,
        "dataframe": df,
        "report": report
    }


def compare_methods(
    basic_results: dict,
    sae_results: dict,
    evaluator: Evaluator,
    ablation_results: Optional[dict] = None
) -> dict:
    """
    Compare results across methods with statistical analysis.
    
    Args:
        basic_results: Results from basic steering
        sae_results: Results from SAE steering
        evaluator: Evaluator instance
        ablation_results: Optional results from layer ablation
        
    Returns:
        Comparison dictionary with statistics
    """
    logger.info("=" * 70)
    logger.info("  METHOD COMPARISON (Statistical Analysis)")
    logger.info("=" * 70)
    
    basic_df = basic_results['dataframe']
    sae_df = sae_results['dataframe']
    
    comparison = {}
    
    # Compute statistical metrics for both methods
    basic_stats = evaluator.compute_statistical_metrics(basic_df, 'lexicon_score')
    sae_stats = evaluator.compute_statistical_metrics(sae_df, 'lexicon_score')
    
    comparison['basic'] = {
        'correlation': basic_stats.correlation,
        'p_value': basic_stats.p_value,
        'effect_size': basic_stats.effect_size,
        'significant': basic_stats.is_significant
    }
    comparison['sae'] = {
        'correlation': sae_stats.correlation,
        'p_value': sae_stats.p_value,
        'effect_size': sae_stats.effect_size,
        'significant': sae_stats.is_significant
    }
    
    print("\n" + "-" * 70)
    print("LEXICON-BASED EVALUATION")
    print("-" * 70)
    print(f"{'Method':<20} {'Correlation':>12} {'P-value':>12} {'Effect Size':>12} {'Significant':>12}")
    print("-" * 70)
    
    print(f"{'Basic Activation':<20} {basic_stats.correlation:>+12.4f} {basic_stats.p_value:>12.2e} {basic_stats.effect_size:>+12.4f} {'YES ✓' if basic_stats.is_significant else 'NO ✗':>12}")
    print(f"{'SAE Steering':<20} {sae_stats.correlation:>+12.4f} {sae_stats.p_value:>12.2e} {sae_stats.effect_size:>+12.4f} {'YES ✓' if sae_stats.is_significant else 'NO ✗':>12}")
    
    # Check if classifier scores are available
    if 'classifier_score' in basic_df.columns and basic_df['classifier_score'].notna().any():
        basic_clf_stats = evaluator.compute_statistical_metrics(basic_df, 'classifier_score')
        sae_clf_stats = evaluator.compute_statistical_metrics(sae_df, 'classifier_score')
        
        comparison['basic']['classifier_correlation'] = basic_clf_stats.correlation
        comparison['sae']['classifier_correlation'] = sae_clf_stats.correlation
        
        print("\n" + "-" * 70)
        print("CLASSIFIER-BASED EVALUATION (RoBERTa Sentiment)")
        print("-" * 70)
        print(f"{'Method':<20} {'Correlation':>12} {'P-value':>12} {'Effect Size':>12} {'Significant':>12}")
        print("-" * 70)
        print(f"{'Basic Activation':<20} {basic_clf_stats.correlation:>+12.4f} {basic_clf_stats.p_value:>12.2e} {basic_clf_stats.effect_size:>+12.4f} {'YES ✓' if basic_clf_stats.is_significant else 'NO ✗':>12}")
        print(f"{'SAE Steering':<20} {sae_clf_stats.correlation:>+12.4f} {sae_clf_stats.p_value:>12.2e} {sae_clf_stats.effect_size:>+12.4f} {'YES ✓' if sae_clf_stats.is_significant else 'NO ✗':>12}")
    
    # Winner determination
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    
    # Compare by correlation strength
    if abs(basic_stats.correlation) > abs(sae_stats.correlation):
        winner = "Basic Activation"
        margin = abs(basic_stats.correlation) - abs(sae_stats.correlation)
    else:
        winner = "SAE Steering"
        margin = abs(sae_stats.correlation) - abs(basic_stats.correlation)
    
    print(f"Higher correlation: {winner} (margin: {margin:.4f})")
    
    if abs(basic_stats.effect_size) > abs(sae_stats.effect_size):
        print(f"Larger effect size: Basic Activation ({basic_stats.effect_size:+.4f})")
    else:
        print(f"Larger effect size: SAE Steering ({sae_stats.effect_size:+.4f})")
    
    # Include layer recommendation if available
    if ablation_results and 'best_layer' in ablation_results:
        print(f"\nRecommended layer (from ablation): {ablation_results['best_layer']}")
        comparison['recommended_layer'] = ablation_results['best_layer']
    
    print("=" * 70 + "\n")
    
    return comparison


import numpy as np # Assurez-vous d'avoir cet import en haut, sinon ajoutez-le ici

def make_serializable(obj):
    """
    Nettoie récursivement un objet pour qu'il soit compatible JSON.
    Gère : NumPy int/float, NumPy arrays, Pandas DataFrames, et clés de dict.
    """
    if isinstance(obj, dict):
        # On convertit TOUTES les clés en string pour éviter l'erreur "keys must be str..."
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, 'to_dict'): 
        # C'est un DataFrame ou une Series Pandas -> on convertit en liste de dicts
        # orient='records' donne format : [{'col': val}, {'col': val}]
        return obj.to_dict(orient='records')
    elif hasattr(obj, 'item'): 
        # C'est un scalaire NumPy (int64, float32, etc) -> on convertit en Python natif
        return obj.item()
    elif hasattr(obj, 'tolist'): 
        # C'est un array NumPy -> on convertit en liste
        return obj.tolist()
    return obj

def save_results(results, output_dir, filename_prefix):
    """
    Sauvegarde les résultats en JSON en nettoyant tous les types incompatibles.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    
    print(f"🧹 Nettoyage des données avant sauvegarde...")
    
    # On passe tout le dictionnaire dans la moulinette de nettoyage
    clean_results = make_serializable(results)
            
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Résultats sauvegardés avec succès dans : {path}")
    except Exception as e:
        print(f"❌ Erreur critique lors de la sauvegarde JSON : {e}")
        # On tente un debug pour voir où ça coince
        print("   Type de clean_results:", type(clean_results))


def main(
    run_basic: bool = True,
    run_sae: bool = True,
    run_ablation: bool = True,
    use_classifier: bool = True,
    save: bool = True,
    visuel: bool = True
):
    """
    Main experiment runner.
    
    Args:
        run_basic: Whether to run basic activation steering
        run_sae: Whether to run SAE steering
        run_ablation: Whether to run layer ablation
        use_classifier: Whether to use sentiment classifier
        save: Whether to save results
    """
    # Initialize
    print("\n Beginning Steering Experiments\n")
    config = Config()
    model_loader = ModelLoader(config)
    evaluator = Evaluator(config, use_classifier=use_classifier)
    
    all_results = {}
    
    # Run experiments
    if run_basic:
        print("\n Starting Basic Activation Steering Experiment\n")
        all_results['basic'] = run_basic_steering_experiment(
            model_loader, config, evaluator
        )
        print("\n Basic Activation Steering Experiment Completed\n")
    
    if run_ablation:
        print("\n Starting Layer Ablation Study\n")
        all_results['ablation'] = run_layer_ablation(
            model_loader, config, evaluator
        )
        print("\n Layer Ablation Study Completed\n")
    
    if run_sae:
        print("\n Starting SAE Steering Experiment\n")
        all_results['sae'] = run_sae_steering_experiment(
            model_loader, config, evaluator
        )
        print("\n SAE Steering Experiment Completed\n")
    
    # Compare methods
    if run_basic and run_sae:
        print("\n Starting Method Comparison\n")
        ablation = all_results.get('ablation')
        all_results['comparison'] = compare_methods(
            all_results['basic'],
            all_results['sae'],
            evaluator,
            ablation_results=ablation
        )
        print("\n Method Comparison Completed\n")
    
    # Save results
    if save:
        print("\n Saving Results\n")
        save_results(all_results, config.output_dir, "steering_comparison")
        print(" Results saved successfully.\n")
    
    if visuel:
        print("\n Generating Visualizations\n")
        run_visualizations()
        print(" Visualizations completed.\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steering experiments")
    parser.add_argument("--no-basic", action="store_true", help="Skip basic steering")
    parser.add_argument("--no-sae", action="store_true", help="Skip SAE steering")
    parser.add_argument("--no-ablation", action="store_true", help="Skip layer ablation")
    parser.add_argument("--no-classifier", action="store_true", help="Skip sentiment classifier")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-visuel", action="store_true", help="Skip visualization")
    args = parser.parse_args()
    
    main(
        run_basic=not args.no_basic,
        run_sae=not args.no_sae,
        run_ablation=not args.no_ablation,
        use_classifier=not args.no_classifier,
        save=not args.no_save,
        visuel=not args.no_visuel
    )
