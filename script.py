import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

if not os.path.exists('data'): os.makedirs('data')

np.random.seed(42)
n_clientes = 40
hoje = datetime.now()
lojas = ['Lisboa-Rossio', 'Porto-NorteShopping', 'Online', 'Coimbra-Alma']

catalogo = {
    'Lentes de Contacto (30 unidades)': {
        'ciclo_base': 30, 'categoria': 'Óptica', 'marca': 'Acuvue', 'preco': 24.99,
        'affinities': ['Soro Fisiológico (Pack 20)', 'Gel Higienizante 500ml']
    },
    'Soro Fisiológico (Pack 20)': {
        'ciclo_base': 25, 'categoria': 'Saúde Visual', 'marca': 'Marca Própria', 'preco': 3.99,
        'affinities': ['Lentes de Contacto (30 unidades)']
    },
    'Óculos de Sol UV': {
        'ciclo_base': 180, 'categoria': 'Óptica', 'marca': 'Ray-Ban', 'preco': 89.99,
        'affinities': ['Protetor Solar SPF50 75ml']
    },
    'Creme Hidratante Facial 50ml': {
        'ciclo_base': 45, 'categoria': 'Dermocosmética', 'marca': 'Wells Care', 'preco': 12.50,
        'affinities': ['Sérum Anti-Rugas 30ml', 'Protetor Solar SPF50 75ml', 'Tónico Facial 200ml']
    },
    'Sérum Anti-Rugas 30ml': {
        'ciclo_base': 60, 'categoria': 'Dermocosmética', 'marca': 'Wells Premium', 'preco': 28.99,
        'affinities': ['Creme Hidratante Facial 50ml']
    },
    'Protetor Solar SPF50 75ml': {
        'ciclo_base': 90, 'categoria': 'Dermocosmética', 'marca': 'Sunsan', 'preco': 15.99,
        'affinities': ['Creme Hidratante Facial 50ml', 'Óculos de Sol UV']
    },
    'Tónico Facial 200ml': {
        'ciclo_base': 40, 'categoria': 'Dermocosmética', 'marca': 'Wells Care', 'preco': 9.99,
        'affinities': ['Creme Hidratante Facial 50ml']
    },
    'Multivitamínico (60 cápsulas)': {
        'ciclo_base': 60, 'categoria': 'Suplementos', 'marca': 'Centrum', 'preco': 18.90,
        'affinities': ['Ómega-3 (60 cápsulas)']
    },
    'Ómega-3 (60 cápsulas)': {
        'ciclo_base': 60, 'categoria': 'Suplementos', 'marca': 'Nutrilite', 'preco': 22.50,
        'affinities': ['Multivitamínico (60 cápsulas)']
    },
    'Proteína em Pó 500g': {
        'ciclo_base': 45, 'categoria': 'Suplementos', 'marca': 'Whey Gold', 'preco': 35.99,
        'affinities': ['Multivitamínico (60 cápsulas)']
    },
    'Gel Higienizante 500ml': {
        'ciclo_base': 20, 'categoria': 'Higiene', 'marca': 'Marca Própria', 'preco': 4.99,
        'affinities': ['Desodorizante Antitranspirante']
    },
    'Desodorizante Antitranspirante': {
        'ciclo_base': 25, 'categoria': 'Higiene', 'marca': 'Rexona', 'preco': 5.99,
        'affinities': ['Gel Higienizante 500ml']
    }
}

transactions_list = []
items_list = []

transaction_id = 100000

for cliente_idx in range(n_clientes):
    c_id = 10000 + cliente_idx

    produtos_primarios = np.random.choice(list(catalogo.keys()), size=np.random.randint(1, 3), replace=False)

    for prod_principal in produtos_primarios:
        info_prod = catalogo[prod_principal]
        ritmo = info_prod['ciclo_base'] + np.random.randint(-5, 6)

        dias_desde_ultima_compra = np.random.randint(0, int(ritmo))
        data_compra = hoje - timedelta(days=dias_desde_ultima_compra)

        num_purchases = np.random.randint(5, 12)

        for purchase_num in range(num_purchases):
            t_id = transaction_id
            transaction_id += 1

            loja = np.random.choice(lojas)

            transactions_list.append({
                'ID_Transacao': t_id,
                'ID_Cliente': c_id,
                'Data': data_compra.strftime('%Y-%m-%d'),
                'Loja': loja
            })

            items_list.append({
                'ID_Transacao': t_id,
                'Produto': prod_principal,
                'Categoria': info_prod['categoria'],
                'Marca': info_prod['marca'],
                'Preco_Unitario': info_prod['preco'],
                'Quantidade': np.random.randint(1, 3)
            })

            if np.random.random() < 0.7 and len(info_prod['affinities']) > 0:
                n_extras = np.random.randint(1, min(3, len(info_prod['affinities']) + 1))
                produtos_extra = np.random.choice(info_prod['affinities'], size=n_extras, replace=False)

                for prod_extra in produtos_extra:
                    info_extra = catalogo[prod_extra]
                    items_list.append({
                        'ID_Transacao': t_id,
                        'Produto': prod_extra,
                        'Categoria': info_extra['categoria'],
                        'Marca': info_extra['marca'],
                        'Preco_Unitario': info_extra['preco'],
                        'Quantidade': np.random.randint(1, 2)
                    })

            data_compra -= timedelta(days=ritmo + np.random.randint(-2, 3))

df_transactions = pd.DataFrame(transactions_list)
df_items = pd.DataFrame(items_list)

df_transactions.to_csv('data/transacoes.csv', index=False)
df_items.to_csv('data/itens_transacao.csv', index=False)
