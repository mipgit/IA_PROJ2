import pandas as pd
import numpy as np
import joblib
from itertools import combinations
import os

if not os.path.exists('data/itens_transacao.csv'):
    print("Erro: Corre primeiro o script.py e processing_data.py!")
    exit()

df_transacoes = pd.read_csv('data/transacoes.csv')
df_itens = pd.read_csv('data/itens_transacao.csv')

# Calculate Co-Purchases

transactions = {} 
for tid in df_itens['ID_Transacao'].unique():
    produtos = set(df_itens[df_itens['ID_Transacao'] == tid]['Produto'].unique())
    transactions[tid] = produtos

total_transactions = len(transactions)

product_pairs = {}

for tid, produtos in transactions.items():
    for prod_a, prod_b in combinations(sorted(produtos), 2):
        pair_key = (prod_a, prod_b)
        product_pairs[pair_key] = product_pairs.get(pair_key, 0) + 1

# Count individual product purchases
product_counts = {}
for tid, produtos in transactions.items():
    for prod in produtos:
        product_counts[prod] = product_counts.get(prod, 0) + 1

rules_data = []

for (prod_a, prod_b), count_both in product_pairs.items():
    count_a = product_counts[prod_a]
    count_b = product_counts[prod_b]

    co_purchase_rate = (count_both / total_transactions) * 100

    confidence_a_to_b = (count_both / count_a) * 100 if count_a > 0 else 0
    confidence_b_to_a = (count_both / count_b) * 100 if count_b > 0 else 0

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

df_rules = pd.DataFrame(rules_data)
df_rules = df_rules.sort_values('Co_Purchase_Rate', ascending=False)


# Display results

display_df = df_rules[[
    'Produto_A', 'Produto_B', 'Confianca_A_para_B', 'Confianca_B_para_A', 
    'Co_Purchase_Rate', 'Lift'
]].copy()


# Save rules to csv

df_rules.to_csv('data/regras_associacao.csv', index=False)

recommendations = {}

for _, row in df_rules.iterrows():
    prod_a = row['Produto_A']
    prod_b = row['Produto_B']
    conf_a_to_b = row['Confianca_A_para_B']
    conf_b_to_a = row['Confianca_B_para_A']
    lift = row['Lift']

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

for produto in recommendations:
    recommendations[produto] = sorted(
        recommendations[produto],
        key=lambda x: x['confianca'],
        reverse=True
    )[:3]

joblib.dump(recommendations, 'models/modelo_recomendacoes.pkl')

print(f"Modelo de recomendações guardado em 'models/modelo_recomendacoes.pkl'")

