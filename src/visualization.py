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
import matplotlib.pyplot as plt
import glob
import os
import json
import ast
import re
from datetime import datetime
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# class pour calculer les scores IA
class Calculate_scores:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f" Chargement des modèles IA ({'GPU' if self.device==0 else 'CPU'})...")
        self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=self.device,
            top_k=None
        )

    def compute(self, prompt, text):
        if not text: return 0.0, 0.0
        # Sémantique
        emb1 = self.sim_model.encode(prompt, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.sim_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        sem = util.cos_sim(emb1, emb2).item()
        # Sentiment
        try:
            res = self.sentiment_pipe(text[:512])[0]
            pos = next(x['score'] for x in res if x['label'] == 'positive')
            neg = next(x['score'] for x in res if x['label'] == 'negative')
            sent = pos - neg
        except:
            sent = 0.0
        return round(sem, 4), round(sent, 4)

# parse les données stockées dans le JSON
def clean_python_string(s):
    if not isinstance(s, str): return s
    if "'dataframe':" in s:
        s_clean = s.split("'dataframe':")[0].strip()
        if s_clean.endswith(','): s_clean = s_clean[:-1]
        if not s_clean.endswith('}'): s_clean += '}'
        return s_clean
    return s

def parse_method_data(raw_value):
    if isinstance(raw_value, dict): return raw_value
    try:
        return ast.literal_eval(raw_value)
    except:
        try:
            return ast.literal_eval(clean_python_string(raw_value))
        except:
            return None

def extract_timestamp_from_filename(filename):
    # Cherche un motif YYYYMMDD_HHMMSS (ex: 20260112_152928)
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d_%H%M%S") # Fallback si pas trouvé

# charge le dernier JSON, calcule les scores IA manquants, sauvegarde et retourne un DataFrame
def load_enrich_and_save():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    json_files = glob.glob(os.path.join(results_dir, "steering_comparison_*.json"))
    
    if not json_files:
        print("Aucun JSON trouvé.")
        return None
        
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Lecture : {os.path.basename(latest_file)}")
    id_timestamped_files = extract_timestamp_from_filename(latest_file)
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dfs = []
    methods = ['basic', 'sae', 'ablation']
    scorer = None
    file_needs_update = False 

    for method in methods:
        if method in data:
            method_data = parse_method_data(data[method])
            if not method_data: continue

            # On cherche les données dans 'dataframe' (votre version réparée) ou 'results' (standard)
            raw_data = []
            if 'dataframe' in method_data and isinstance(method_data['dataframe'], list):
                raw_data = method_data['dataframe']
            elif 'results' in method_data and isinstance(method_data['results'], list):
                raw_data = method_data['results']
            
            if raw_data:
                df = pd.DataFrame(raw_data)
               
                # Vérification Scores IA
                cols_missing = 'classifier_score' not in df.columns or 'semantic_score' not in df.columns
                
                if cols_missing:
                    print(f"   Calcul Scores IA pour {method.upper()}...")
                    if scorer is None: scorer = Calculate_scores()
                    sem_scores, sent_scores = [], []
                    for _, row in tqdm(df.iterrows(), total=len(df)):
                        p = row.get('prompt', '')
                        t = row.get('generated_text', '')
                        sem, sent = scorer.compute(p, t)
                        sem_scores.append(sem)
                        sent_scores.append(sent)
                    df['semantic_score'] = sem_scores
                    df['classifier_score'] = sent_scores
                    
                    # Mise à jour des données en mémoire
                    method_data['dataframe'] = df.to_dict(orient='records')
                    # On garde aussi 'results' synchronisé
                    method_data['results'] = method_data['dataframe']
                    data[method] = method_data
                    file_needs_update = True
                
                df['Method'] = method.upper()
                dfs.append(df)

    if file_needs_update:
        print("Sauvegarde des nouveaux scores IA...")
        try:
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde : {e}")

    if not dfs: return None
    return pd.concat(dfs, ignore_index=True),id_timestamped_files

# Affichage du tableau de bord complet
def plot_results(df_all,id_timestamped_files):
    df = df_all[df_all['Method'] != 'ABLATION'].copy()
    if df is None or df.empty: return

    if 'coefficient' in df.columns:
        df = df.sort_values(by='coefficient')
        
    methods = df['Method'].unique()
    
    # Configuration du style et de la taille (2 lignes, 3 colonnes)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Tableau de Bord Complet du Steering', fontsize=16, fontweight='bold')
    try:
        fig.canvas.manager.set_window_title("Analyse 1 : Comparaison Basic vs SAE")
    except:
        pass 
    # 1. Efficacité (Sentiment)
    ax = axes[0, 0]
    for method in methods:
        subset = df[df['Method'] == method]
        col = 'classifier_score' if 'classifier_score' in subset.columns else 'classifier_score'
        if col in subset.columns:
            grouped = subset.groupby('coefficient')[col].mean()
            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=method)
    ax.set_title('1. Efficacité IA (Sentiment)', fontweight='bold')
    ax.set_ylabel('Score (-1 Haine ... +1 Amour)')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True)

    # 2. Qualité (Sémantique)
    ax = axes[0, 1]
    for method in methods:
        subset = df[df['Method'] == method]
        if 'semantic_score' in subset.columns:
            grouped = subset.groupby('coefficient')['semantic_score'].mean()
            ax.plot(grouped.index, grouped.values, marker='s', linewidth=2, label=method)
    ax.set_title('2. Qualité Sémantique (Cohérence)', fontweight='bold')
    ax.set_ylabel('Score SBERT (0 à 1)')
    ax.legend()
    ax.grid(True)

    # 3. Compromis (Scatter)
    ax = axes[0, 2]
    for method in methods:
        subset = df[df['Method'] == method]
        col_sent = 'classifier_score' if 'classifier_score' in subset.columns else 'classifier_score'
        if col_sent in subset.columns and 'semantic_score' in subset.columns:
            ax.scatter(subset[col_sent], subset['semantic_score'], label=method, alpha=0.6)
    ax.set_title('3. Compromis (Trade-off)', fontweight='bold')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Qualité')
    ax.legend()
    ax.grid(True)

    # 4. Score Lexical (Global)
    ax = axes[1, 0]
    for method in methods:
        subset = df[df['Method'] == method]
        if 'lexicon_score' in subset.columns:
            grouped = subset.groupby('coefficient')['lexicon_score'].mean()
            ax.plot(grouped.index, grouped.values, marker='d', linewidth=2, linestyle='--', label=method)
    ax.set_title('4. Score Lexical (Mots-clés)', fontweight='bold')
    ax.set_xlabel('Force (Coefficient)')
    ax.set_ylabel('Ratio (Mots Love - Mots Hate)')
    ax.axhline(0, color='black', linestyle='--')
    ax.legend()
    ax.grid(True)

    # 5. Mots "Love" (Moyenne)
    ax = axes[1, 1]
    for method in methods:
        subset = df[df['Method'] == method]
        if 'love_words' in subset.columns:
            grouped = subset.groupby('coefficient')['love_words'].mean()
            ax.plot(grouped.index, grouped.values, marker='^', linewidth=2, label=f"{method} (Love)")
    ax.set_title('5. Apparition Mots "Love"', fontweight='bold')
    ax.set_xlabel('Force (Coefficient)')
    ax.set_ylabel('Nombre moyen par phrase')
    ax.legend()
    ax.grid(True)

    # 6. Mots "Hate" (Moyenne)
    ax = axes[1, 2]
    for method in methods:
        subset = df[df['Method'] == method]
        if 'hate_words' in subset.columns:
            grouped = subset.groupby('coefficient')['hate_words'].mean()
            ax.plot(grouped.index, grouped.values, marker='v', linewidth=2,  label=f"{method} (Hate)")
    ax.set_title('6. Apparition Mots "Hate"', fontweight='bold')
    ax.set_xlabel('Force (Coefficient)')
    ax.set_ylabel('Nombre moyen par phrase')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    # Sauvegarde
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "results", "plots", "dashboard_complete_"+id_timestamped_files+".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    print(f"\nTableau de bord sauvegardé sous : {save_path}")
    

def plot_ablation_study(df, id_timestamped_files):

    df_ab = df[df['Method'] == 'ABLATION'].copy()
    
    if df_ab.empty:
        return
        
    if 'layer' not in df_ab.columns:
        return

    df_ab = df_ab.sort_values(by='layer')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    try:
        fig.canvas.manager.set_window_title("Analyse 2 : Ablation par Couche")
    except:
        pass 
    # Axe Y1 : Sentiment (Bleu)
    color = 'tab:blue'
    ax1.set_xlabel('Couche du Modèle (Layer Index)')
    ax1.set_ylabel('Impact sur le Sentiment', color=color, fontweight='bold')
    
    grouped_sent = df_ab.groupby('layer')['classifier_score'].mean()
    ax1.plot(grouped_sent.index, grouped_sent.values, color=color, marker='o', linewidth=3, label="Sentiment")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Axe Y2 : Qualité (Vert)
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Qualité Sémantique', color=color, fontweight='bold')
    grouped_sem = df_ab.groupby('layer')['semantic_score'].mean()
    ax2.plot(grouped_sem.index, grouped_sem.values, color=color, marker='s', linestyle='--', linewidth=2, label="Qualité")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)

    plt.title("Étude d'Ablation : Impact par Couche", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "results", "plots", "etude_ablation_layers_"+id_timestamped_files+".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    print(f"Graphique Ablation sauvegardé : {save_path}")


def run_visualizations():
    print("\nGénération du Tableau de Bord Complet...")
    df,id_timestamped_files = load_enrich_and_save()
    if df is not None:
        plot_results(df,id_timestamped_files)
        plot_ablation_study(df, id_timestamped_files)
        try: plt.show()
        except: pass
    else:
        print("Aucune donnée disponible.")

if __name__ == "__main__":
    run_visualizations()