import pandas as pd
import os
import argparse
import numpy as np
from itertools import combinations

parser = argparse.ArgumentParser(description='Process normalized transactions into training features')
parser.add_argument('--noise', type=float, default=5.0, help='Stddev of gaussian noise to add to Target_Dias_Restantes (default: 5.0 for realistic variance)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for noise generation')
args = parser.parse_args()
noise_sigma = float(args.noise)
np.random.seed(int(args.seed))

if not os.path.exists('data/transacoes.csv') or not os.path.exists('data/itens_transacao.csv'):
    print("Erro: Corre primeiro o script.py")
    exit()

df_transacoes = pd.read_csv('data/transacoes.csv')
df_itens = pd.read_csv('data/itens_transacao.csv')

df = df_transacoes.merge(df_itens, on='ID_Transacao')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values(['ID_Cliente', 'Produto', 'Data'])

df['Intervalo'] = df.groupby(['ID_Cliente', 'Produto'])['Data'].diff().dt.days

features_recurrence = []

for (cid, prod), grupo in df.groupby(['ID_Cliente', 'Produto']):
    if len(grupo) >= 3:
        intervalo_medio = grupo['Intervalo'].dropna().mean()
        ultima_compra = grupo['Data'].max()

        hoje = pd.Timestamp.now()
        dias_desde_ultima = (hoje - ultima_compra).days

        target = intervalo_medio - dias_desde_ultima
        if noise_sigma > 0:
            target = target + np.random.normal(loc=0.0, scale=noise_sigma)

        if dias_desde_ultima <= intervalo_medio * 1.5: # permite até 50% de variação (inclui atrasados)
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
                'Target_Dias_Restantes': round(target, 1), 
                'Data_Ultima_Compra': ultima_compra.strftime('%Y-%m-%d')
            })

df_treino = pd.DataFrame(features_recurrence)
df_treino.to_csv('data/dados_treino.csv', index=False)

print(f"Dados de treino: {len(df_treino)} exemplos em 'data/dados_treino.csv'")

cocompras = []

for tid, grupo_transacao in df_itens.groupby('ID_Transacao'):
    produtos = grupo_transacao['Produto'].unique()

    if len(produtos) > 1:
        for prod_a, prod_b in combinations(sorted(produtos), 2):
            cocompras.append({
                'ID_Transacao': tid,
                'Produto_A': prod_a,
                'Produto_B': prod_b
            })

df_cocompras = pd.DataFrame(cocompras)

if len(df_cocompras) > 0:
    pair_freq = df_cocompras.groupby(['Produto_A', 'Produto_B']).size().reset_index(name='Frequencia')

    total_transactions = len(df_transacoes)
    pair_freq['Suporte'] = (pair_freq['Frequencia'] / total_transactions * 100).round(2)
    pair_freq = pair_freq.sort_values('Frequencia', ascending=False)

    pair_freq.to_csv('data/dados_cocompra.csv', index=False)
else:
    print("Sem co-compras para analisar (transações com apenas 1 produto)")
