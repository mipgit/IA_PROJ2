import pandas as pd
import os

# 1. Carregar os Dados Brutos
if not os.path.exists('data/dados_brutos.csv'):
    print("Erro: Corre primeiro o script.py!")
    exit()

df = pd.read_csv('data/dados_brutos.csv')
df['Data'] = pd.to_datetime(df['Data'])

# 2. Ordenar por cliente e data para calcular intervalos
df = df.sort_values(['ID_Cliente', 'Data'])

# 3. Calcular o intervalo (em dias) entre compras sucessivas
df['Intervalo'] = df.groupby(['ID_Cliente', 'Produto'])['Data'].diff().dt.days

# 4. Agrupar para criar o Dataset de Treino
# Queremos saber o hábito médio de cada cliente
features = []

for (cid, prod), grupo in df.groupby(['ID_Cliente', 'Produto']):
    if len(grupo) >= 3: # Precisamos de pelo menos 3 compras para ter um padrão
        intervalo_medio = grupo['Intervalo'].mean()
        ultima_compra = grupo['Data'].max()
        
        # Simular que estamos a processar isto "hoje"
        hoje = pd.Timestamp.now()
        dias_desde_ultima = (hoje - ultima_compra).days
        
        # O ALVO (TARGET): Quantos dias faltavam realmente para ele comprar?
        # Para o treino, usamos o intervalo real que ele costuma fazer
        target = intervalo_medio - dias_desde_ultima
        
        # Metadata extra do produto (pegamos no primeiro registo do grupo)
        categoria = grupo['Categoria'].iloc[0]
        preco = grupo['Preco_Unitario'].iloc[0]
        
        features.append({
            'ID_Cliente': cid,
            'Produto': prod,
            'Intervalo_Medio_Habito': round(intervalo_medio, 1),
            'Dias_Desde_Ultima_Compra': dias_desde_ultima,
            'Total_Compras_Historico': len(grupo),
            'Preco_Unitario': preco,
            'Target_Dias_Restantes': round(target, 1)
        })

# 5. Guardar o CSV Filtrado para a IA
df_treino = pd.DataFrame(features)
df_treino.to_csv('data/dados_treino_ia.csv', index=False)

print(f"Sucesso! Criado 'data/dados_treino_ia.csv' com {len(df_treino)} exemplos filtrados.")
print("Este ficheiro contém apenas a 'inteligência' extraída dos logs.")