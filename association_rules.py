import pandas as pd
import numpy as np
import joblib
from itertools import combinations
import os

# ============================================================================
# LOAD DATA
# ============================================================================

if not os.path.exists('data/itens_transacao.csv'):
    print("Erro: Corre primeiro o script.py e processing_data.py!")
    exit()

df_transacoes = pd.read_csv('data/transacoes.csv')
df_itens = pd.read_csv('data/itens_transacao.csv')

# ============================================================================
# CALCULATE CO-PURCHASES WITH SYMMETRIC CONFIDENCES
# ============================================================================

print("="*80)
print("CALCULANDO REGRAS DE ASSOCIAÇÃO COM CONFIDÊNCIAS SIMÉTRICAS")
print("="*80)

# Create list of transactions with products
transactions = {}  # transaction_id -> set of products
for tid in df_itens['ID_Transacao'].unique():
    produtos = set(df_itens[df_itens['ID_Transacao'] == tid]['Produto'].unique())
    transactions[tid] = produtos

total_transactions = len(transactions)
print(f"\nTotal de transações: {total_transactions}")

# Find all product pairs that appear together
product_pairs = {}  # (prod_a, prod_b) -> count

for tid, produtos in transactions.items():
    # For each pair of products in this transaction
    for prod_a, prod_b in combinations(sorted(produtos), 2):
        pair_key = (prod_a, prod_b)
        product_pairs[pair_key] = product_pairs.get(pair_key, 0) + 1

print(f"Pares de produtos únicos: {len(product_pairs)}")

# Count individual product purchases
product_counts = {}
for tid, produtos in transactions.items():
    for prod in produtos:
        product_counts[prod] = product_counts.get(prod, 0) + 1

# ============================================================================
# CALCULATE METRICS FOR EACH PAIR
# ============================================================================

rules_data = []

for (prod_a, prod_b), count_both in product_pairs.items():
    count_a = product_counts[prod_a]
    count_b = product_counts[prod_b]
    
    # Co-purchase rate: % of all transactions with both products
    co_purchase_rate = (count_both / total_transactions) * 100
    
    # Directional confidences
    confidence_a_to_b = (count_both / count_a) * 100 if count_a > 0 else 0
    confidence_b_to_a = (count_both / count_b) * 100 if count_b > 0 else 0
    
    # Lift: how much more likely than random
    expected_if_independent = (count_a / total_transactions) * (count_b / total_transactions)
    actual = count_both / total_transactions
    lift = actual / expected_if_independent if expected_if_independent > 0 else 0
    
    rules_data.append({
        'Produto_A': prod_a,
        'Produto_B': prod_b,
        'Confianca_A_para_B': round(confidence_a_to_b, 2),
        'Confianca_B_para_A': round(confidence_b_to_a, 2),
        'Co_Purchase_Rate': round(co_purchase_rate, 2),
        'Lift': round(lift, 2),
        'Transacoes_com_Ambos': count_both
    })

# Create dataframe and sort by co-purchase rate
df_rules = pd.DataFrame(rules_data)
df_rules = df_rules.sort_values('Co_Purchase_Rate', ascending=False)

print(f"\nRegras de associação geradas: {len(df_rules)}")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("REGRAS DE ASSOCIAÇÃO COM CONFIDÊNCIAS SIMÉTRICAS")
print(f"{'='*80}\n")

display_df = df_rules[[
    'Produto_A', 'Produto_B', 'Confianca_A_para_B', 'Confianca_B_para_A', 
    'Co_Purchase_Rate', 'Lift'
]].copy()

for idx, (_, row) in enumerate(display_df.iterrows(), 1):
    print(f"{idx}. {row['Produto_A']} ↔ {row['Produto_B']}")
    print(f"   Se compra A → B: {row['Confianca_A_para_B']:.1f}%")
    print(f"   Se compra B → A: {row['Confianca_B_para_A']:.1f}%")
    print(f"   Co-purchase Rate: {row['Co_Purchase_Rate']:.1f}%")
    print(f"   Lift: {row['Lift']:.2f}x\n")

# ============================================================================
# SAVE RULES TO CSV
# ============================================================================

df_rules.to_csv('data/regras_associacao.csv', index=False)
print(f"✓ Regras salvas em 'data/regras_associacao.csv'")

# ============================================================================
# BUILD PRODUCT RECOMMENDATION DICT (for quick lookup in streamlit)
# ============================================================================

recommendations = {}

for _, row in df_rules.iterrows():
    prod_a = row['Produto_A']
    prod_b = row['Produto_B']
    conf_a_to_b = row['Confianca_A_para_B']
    conf_b_to_a = row['Confianca_B_para_A']
    lift = row['Lift']
    
    # Add both directions to recommendations
    if prod_a not in recommendations:
        recommendations[prod_a] = []
    recommendations[prod_a].append({
        'produto': prod_b,
        'confianca': conf_a_to_b,
        'lift': lift
    })
    
    if prod_b not in recommendations:
        recommendations[prod_b] = []
    recommendations[prod_b].append({
        'produto': prod_a,
        'confianca': conf_b_to_a,
        'lift': lift
    })

# Sort by confidence for each product
for produto in recommendations:
    recommendations[produto] = sorted(
        recommendations[produto],
        key=lambda x: x['confianca'],
        reverse=True
    )[:3]  # Keep top 3

# Save recommendations
joblib.dump(recommendations, 'models/modelo_recomendacoes.pkl')

print(f"✓ Modelo de recomendações guardado em 'models/modelo_recomendacoes.pkl'")
print(f"  - Produtos com recomendações: {len(recommendations)}")

print(f"\n{'='*80}")
print("✓ Sucesso: Modelos de associação treinados com confidências simétricas!")
print(f"{'='*80}\n")
