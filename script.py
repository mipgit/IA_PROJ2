import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

if not os.path.exists('data'): os.makedirs('data')

np.random.seed(42)
n_clientes = 50
hoje = datetime.now()
data_list = []

# Catálogo de Produtos Enriquecido (Metadata)
catalogo = {
    'Lentes de Contacto (30 unidades)': {
        'ciclo_base': 30, 'categoria': 'Óptica', 'marca': 'Acuvue', 'preco': 24.99},
    'Creme Hidratante Facial 50ml': {
        'ciclo_base': 45, 'categoria': 'Dermocosmética', 'marca': 'Wells Care', 'preco': 12.50},
    'Multivitamínico (60 cápsulas)': {
        'ciclo_base': 60, 'categoria': 'Suplementos', 'marca': 'Centrum', 'preco': 18.90},
    'Soro Fisiológico (Pack 20)': {
        'ciclo_base': 20, 'categoria': 'Saúde', 'marca': 'Marca Própria', 'preco': 3.99}
}

for i in range(n_clientes):
    c_id = 10000 + i
    # Escolher um produto fixo para este cliente (simular fidelidade a um produto)
    prod_nome = np.random.choice(list(catalogo.keys()))
    info_prod = catalogo[prod_nome]
    
    # Ritmo de consumo individual
    ritmo = info_prod['ciclo_base'] + np.random.randint(-5, 6)
    
    data_compra = hoje - timedelta(days=np.random.randint(200, 500))
    
    for _ in range(np.random.randint(5, 12)):
        if data_compra <= hoje:
            data_list.append({
                'ID_Transacao': np.random.randint(100000, 999999),
                'ID_Cliente': c_id,
                'Data': data_compra.strftime('%Y-%m-%d'),
                'Produto': prod_nome,
                'Categoria': info_prod['categoria'],
                'Marca': info_prod['marca'],
                'Preco_Unitario': info_prod['preco'],
                'Loja': np.random.choice(['Lisboa-Rossio', 'Porto-NorteShopping', 'Online', 'Coimbra-Alma'])
            })
        data_compra += timedelta(days=ritmo + np.random.randint(-2, 3))

df = pd.DataFrame(data_list)
df.to_csv('data/dados_brutos.csv', index=False)
print(f"Sucesso: Dados enriquecidos gerados em 'data/dados_brutos.csv'.")