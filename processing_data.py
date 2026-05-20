import pandas as pd
import os
from itertools import combinations

# ============================================================================
# LOAD NORMALIZED DATA
# ============================================================================

if not os.path.exists('data/transacoes.csv') or not os.path.exists('data/itens_transacao.csv'):
    print("Erro: Corre primeiro o script.py")
    exit()

df_transacoes = pd.read_csv('data/transacoes.csv')
df_itens = pd.read_csv('data/itens_transacao.csv')

# Merge to create full transaction view
df = df_transacoes.merge(df_itens, on='ID_Transacao')

# Convert dates
df['Data'] = pd.to_datetime(df['Data'])

# ============================================================================
# FEATURE ENGINEERING FOR RECURRENCE MODEL
# ============================================================================

# Sort by customer and date
df = df.sort_values(['ID_Cliente', 'Produto', 'Data'])

# Calculate days between consecutive purchases of same product by same customer
df['Intervalo'] = df.groupby(['ID_Cliente', 'Produto'])['Data'].diff().dt.days

# Create feature set for recurrence model
features_recurrence = []

for (cid, prod), grupo in df.groupby(['ID_Cliente', 'Produto']):
    if len(grupo) >= 3:  # Need at least 3 purchases to establish a pattern
        intervalo_medio = grupo['Intervalo'].dropna().mean()
        ultima_compra = grupo['Data'].max()
        
        # Current date reference
        hoje = pd.Timestamp.now()
        dias_desde_ultima = (hoje - ultima_compra).days
        
        # Target: How many days until next purchase?
        target = intervalo_medio - dias_desde_ultima
        
        # FILTRO: Apenas incluir se última compra está dentro do intervalo médio
        # Isso garante que Target_Dias_Restantes sempre seja >= 0
        # (ou muito próximo, considerando variação)
        if dias_desde_ultima <= intervalo_medio * 1.1:  # Permite até 10% de variação
            # Get product info from last purchase
            categoria = grupo['Categoria'].iloc[-1]
            preco = grupo['Preco_Unitario'].iloc[-1]
            marca = grupo['Marca'].iloc[-1]
            
            features_recurrence.append({
                'ID_Cliente': cid,
                'Produto': prod,
                'Categoria': categoria,
                'Marca': marca,
                'Intervalo_Medio_Habito': round(intervalo_medio, 1),
                'Dias_Desde_Ultima_Compra': dias_desde_ultima,
                'Total_Compras_Historico': len(grupo),
                'Preco_Unitario': preco,
                'Target_Dias_Restantes': round(max(target, 0), 1),  # Garantir >= 0
                'Data_Ultima_Compra': ultima_compra.strftime('%Y-%m-%d')
            })

df_treino = pd.DataFrame(features_recurrence)
df_treino.to_csv('data/dados_treino_ia.csv', index=False)

# ============================================================================
# FEATURE ENGINEERING FOR CO-PURCHASE/ASSOCIATION RULES
# ============================================================================

# Extract product pairs from each transaction
cocompras = []

for tid, grupo_transacao in df_itens.groupby('ID_Transacao'):
    produtos = grupo_transacao['Produto'].unique()
    
    # Generate all pairs of products in this transaction
    if len(produtos) > 1:
        for prod_a, prod_b in combinations(sorted(produtos), 2):
            cocompras.append({
                'ID_Transacao': tid,
                'Produto_A': prod_a,
                'Produto_B': prod_b
            })

df_cocompras = pd.DataFrame(cocompras)

# Count frequency and calculate association metrics
if len(df_cocompras) > 0:
    # Frequency of each product pair
    pair_freq = df_cocompras.groupby(['Produto_A', 'Produto_B']).size().reset_index(name='Frequencia')
    
    # Total transactions
    total_transactions = len(df_transacoes)
    
    # Support: proportion of transactions containing this pair
    pair_freq['Suporte'] = (pair_freq['Frequencia'] / total_transactions * 100).round(2)
    
    # For each product, calculate how often it appears with the other
    # (This will be refined in association_rules.py with proper confidence/lift)
    pair_freq = pair_freq.sort_values('Frequencia', ascending=False)
    
    pair_freq.to_csv('data/dados_cocompra.csv', index=False)
else:
    print("Sem co-compras para analisar (transações com apenas 1 produto)")
