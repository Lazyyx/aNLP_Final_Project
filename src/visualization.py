"""
Utility functions for visualization and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

def plot_steering_effect(
    df: pd.DataFrame,
    metric: str = 'lexicon_score',
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot steering coefficient vs sentiment score.
    
    Args:
        df: DataFrame with 'coefficient' and metric columns
        metric: Column name for y-axis
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and std per coefficient
    grouped = df.groupby('coefficient')[metric].agg(['mean', 'std']).reset_index()
    
    ax.errorbar(
        grouped['coefficient'], 
        grouped['mean'],
        yerr=grouped['std'],
        marker='o',
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8
    )
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('Steering Coefficient', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'Effect of Steering on {metric}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_layer_ablation(
    df: pd.DataFrame,
    metric: str = 'lexicon_score',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot layer ablation results.
    
    Args:
        df: DataFrame with 'layer' column
        metric: Metric to plot
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layer_means = df.groupby('layer')[metric].mean()
    layer_stds = df.groupby('layer')[metric].std()
    
    colors = ['green' if x > 0 else 'red' for x in layer_means]
    
    ax.bar(
        layer_means.index,
        layer_means.values,
        yerr=layer_stds.values,
        capsize=3,
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Layer Ablation Study: Which Layer Best Encodes Sentiment?', fontsize=14)
    ax.set_xticks(range(12))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best layer
    best_layer = layer_means.idxmax()
    ax.annotate(
        f'Best: Layer {best_layer}',
        xy=(best_layer, layer_means[best_layer]),
        xytext=(best_layer + 1.5, layer_means[best_layer] + 0.02),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_method_comparison(
    results: Dict[str, pd.DataFrame],
    metric: str = 'lexicon_score',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare multiple steering methods.
    
    Args:
        results: Dict mapping method name to DataFrame
        metric: Metric to compare
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10.colors
    
    for i, (method_name, df) in enumerate(results.items()):
        grouped = df.groupby('coefficient')[metric].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(
            grouped['coefficient'],
            grouped['mean'],
            yerr=grouped['std'],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            capsize=3,
            linewidth=2,
            markersize=8,
            label=method_name
        )
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('Steering Coefficient', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Comparison of Steering Methods', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_results_table(
    results: Dict[str, pd.DataFrame],
    metric: str = 'lexicon_score'
) -> pd.DataFrame:
    """
    Create summary table comparing methods.
    
    Args:
        results: Dict mapping method name to DataFrame
        metric: Metric to summarize
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for method_name, df in results.items():
        # Compute correlation
        if 'coefficient' in df.columns:
            corr = df['coefficient'].corr(df[metric])
        else:
            corr = np.nan
            
        # Effect size (difference between max and min coefficient means)
        if 'coefficient' in df.columns:
            coeff_means = df.groupby('coefficient')[metric].mean()
            effect_size = coeff_means.max() - coeff_means.min()
        else:
            effect_size = np.nan
        
        summary_data.append({
            'Method': method_name,
            'Correlation (coeff vs score)': corr,
            'Effect Size': effect_size,
            'Mean Score': df[metric].mean(),
            'Std Score': df[metric].std(),
            'N Samples': len(df)
        })
    
    return pd.DataFrame(summary_data)


def display_sample_outputs(
    df: pd.DataFrame,
    prompt: str,
    n_samples: int = 5
) -> None:
    """
    Display sample outputs for a given prompt at different coefficients.
    
    Args:
        df: Results DataFrame
        prompt: Prompt to filter by
        n_samples: Number of coefficients to show
    """
    subset = df[df['prompt'] == prompt].sort_values('coefficient')
    
    # Select evenly spaced coefficients
    coefficients = subset['coefficient'].unique()
    if len(coefficients) > n_samples:
        indices = np.linspace(0, len(coefficients)-1, n_samples, dtype=int)
        selected_coeffs = [coefficients[i] for i in indices]
        subset = subset[subset['coefficient'].isin(selected_coeffs)]
    
    print(f"\n{'='*60}")
    print(f"Prompt: \"{prompt}\"")
    print('='*60)
    
    for _, row in subset.iterrows():
        coeff = row['coefficient']
        text = row['generated_text']
        score = row.get('lexicon_score', 'N/A')
        
        direction = "→ LOVE" if coeff > 0 else "→ HATE" if coeff < 0 else "NEUTRAL"
        print(f"\n[Coeff {coeff:+.1f}] {direction}")
        print(f"Output: {text}")
        print(f"Lexicon Score: {score}")

def load_data():
    """Charge et fusionne tous les fichiers CSV d'évaluation trouvés."""
    # On cherche dans le dossier results/ à la racine du projet
    # Le chemin peut varier selon d'où on lance le script, on essaie de sécuriser
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    
    files = glob.glob(os.path.join(results_dir, "EVAL_LOVE_HATE_*.csv"))
    
    if not files:
        # Fallback : essaie le chemin relatif simple si le script est lancé depuis la racine
        files = glob.glob("results/EVAL_LOVE_HATE_*.csv")

    if not files:
        print(" Aucun fichier 'EVAL_LOVE_HATE_*.csv' trouvé dans results/.")
        return None

    dfs = []
    print(f" Visualisation : Chargement de {len(files)} fichiers...")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Extraction du nom de la méthode depuis le fichier
            filename = os.path.basename(f)
            parts = filename.split('_')
            # Ex: EVAL_LOVE_HATE_basic_... -> method = "basic"
            if len(parts) >= 5:
                method = parts[3] 
            else:
                method = "unknown"
            
            df['Method'] = method.upper()
            dfs.append(df)
        except Exception as e:
            print(f" Erreur lecture {f}: {e}")

    if not dfs:
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def plot_results(df):
    """Affiche les courbes (Style Loss/Accuracy classique)"""
    
    # Nettoyage des données
    if df is None or df.empty:
        print(" DataFrame vide, pas de graphique.")
        return

    # On trie par coefficient
    df = df.sort_values(by='coefficient')
    methods = df['Method'].unique()
    
    # Configuration du style général
    plt.style.use('seaborn-v0_8-whitegrid') # Style propre et académique
    
    # Création de la figure (1 ligne, 3 colonnes)
    plt.figure(figsize=(18, 5))
    
    # --- GRAPHIQUE 1 : Efficacité ---
    plt.subplot(1, 3, 1)
    for method in methods:
        subset = df[df['Method'] == method]
        # Moyenne au cas où doublons
        grouped = subset.groupby('coefficient')['sentiment_score'].mean()
        plt.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=method)

    plt.title('1. Efficacité (Sentiment)', fontsize=12, fontweight='bold')
    plt.xlabel('Force (Coefficient)')
    plt.ylabel('Score (-1 Haine ... +1 Amour)')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # --- GRAPHIQUE 2 : Qualité ---
    plt.subplot(1, 3, 2)
    for method in methods:
        subset = df[df['Method'] == method]
        grouped = subset.groupby('coefficient')['semantic_score'].mean()
        plt.plot(grouped.index, grouped.values, marker='s', linewidth=2, label=method)
    
    plt.title('2. Qualité (Sémantique)', fontsize=12, fontweight='bold')
    plt.xlabel('Force (Coefficient)')
    plt.ylabel('Score Sémantique (0 à 1)')
    plt.axhline(0.6, color='orange', linestyle='--', label='Seuil Cohérence')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # --- GRAPHIQUE 3 : Compromis ---
    plt.subplot(1, 3, 3)
    for method in methods:
        subset = df[df['Method'] == method]
        plt.scatter(subset['sentiment_score'], subset['semantic_score'], label=method, alpha=0.7, s=50)
        
    plt.title('3. Compromis : Sentiment vs Qualité', fontsize=12, fontweight='bold')
    plt.xlabel('Sentiment Atteint')
    plt.ylabel('Qualité Sémantique')
    plt.axhline(0.6, color='orange', linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Sauvegarde
    plt.tight_layout()
    
    # On sauvegarde dans results/plots/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'resultats_steering_final.png')
    plt.savefig(save_path, dpi=300)
    print(f" Graphiques sauvegardés sous : {save_path}")
    
    # Affichage (sécurisé pour éviter les erreurs si pas d'écran)
    try:
        plt.show()
    except:
        pass

def run_visualizations():
    """Fonction principale appelée par run_experiments.py"""
    print("\n Lancement des visualisations...")
    df = load_data()

    if df is not None:
        plot_results(df)
    else:
        print(" Impossible de générer les graphiques : aucune donnée chargée.")