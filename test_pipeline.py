#!/usr/bin/env python3
"""
End-to-end pipeline test
Verifies all components work together correctly
"""

import sys
import pandas as pd
import joblib
import numpy as np

print("="*80)
print("VERIFICAÇÃO DO PIPELINE WELLS - 4 Modelos ML")
print("="*80)

# Test 1: Check data files
print("\n✓ VERIFICANDO FICHEIROS DE DADOS...")
files_to_check = [
    'data/transacoes.csv',
    'data/itens_transacao.csv',
    'data/dados_treino_ia.csv',
    'data/regras_associacao.csv'
]

for f in files_to_check:
    try:
        df = pd.read_csv(f)
        print(f"  ✓ {f} ({len(df)} registos)")
    except Exception as e:
        print(f"  ✗ {f}: {e}")
        sys.exit(1)

# Test 2: Load trained models
print("\n✓ CARREGANDO MODELOS TREINADOS...")

models_to_test = {
    'models/modelo_wells.pkl': 'Modelo Principal (melhor)',
    'models/modelo_linear.pkl': 'Regressão Linear',
    'models/modelo_dt.pkl': 'Árvore Decisão',
    'models/modelo_rf.pkl': 'Random Forest',
    'models/modelo_gb.pkl': 'Gradient Boosting',
    'models/modelo_recomendacoes.pkl': 'Recomendações'
}

loaded_models = {}
for fname, desc in models_to_test.items():
    try:
        model = joblib.load(fname)
        loaded_models[fname] = model
        print(f"  ✓ {fname} ({desc})")
    except Exception as e:
        print(f"  ✗ {fname}: {e}")
        sys.exit(1)

# Test 3: Test all regression models
print("\n✓ TESTANDO TODOS OS MODELOS DE RECORRÊNCIA...")

df_treino = pd.read_csv('data/dados_treino_ia.csv')
X_test = df_treino[['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']].head(5)
y_test = df_treino['Target_Dias_Restantes'].head(5)

regression_models = {
    'models/modelo_linear.pkl': 'Regressão Linear',
    'models/modelo_dt.pkl': 'Árvore Decisão',
    'models/modelo_rf.pkl': 'Random Forest',
    'models/modelo_gb.pkl': 'Gradient Boosting',
    'models/modelo_wells.pkl': 'Melhor Modelo'
}

print(f"\n  Predições para os primeiros 5 exemplos:")
for fname, label in regression_models.items():
    model = loaded_models[fname]
    predictions = model.predict(X_test)
    print(f"  {label:20}: {[f'{p:.1f}' for p in predictions]}")

# Test 4: Test recommendations
print("\n✓ TESTANDO RECOMENDAÇÕES...")
recommendations = loaded_models['models/modelo_recomendacoes.pkl']
sample_produtos = list(recommendations.keys())[:3]
for prod in sample_produtos:
    recos = recommendations[prod]
    if recos:
        print(f"  - {prod}:")
        for r in recos[:2]:
            print(f"      → {r['produto']} ({r['confianca']:.0f}% confiança)")
    else:
        print(f"  - {prod}: sem recomendações")

# Test 5: Association rules
print("\n✓ VERIFICANDO REGRAS DE ASSOCIAÇÃO...")

df_regras = pd.read_csv('data/regras_associacao.csv')
print(f"  Total de pares únicos: {len(df_regras)}")
print(f"  Confiança média (A→B): {df_regras['Confianca_A_para_B'].mean():.1f}%")
print(f"  Co-purchase rate média: {df_regras['Co_Purchase_Rate'].mean():.1f}%")
print(f"  Top 3 pares (por co-purchase rate):")

for _, row in df_regras.nlargest(3, 'Co_Purchase_Rate').iterrows():
    print(f"    - {row['Produto_A']} ↔ {row['Produto_B']}")
    print(f"      A→B: {row['Confianca_A_para_B']:.1f}% | B→A: {row['Confianca_B_para_A']:.1f}% | Co-purchase: {row['Co_Purchase_Rate']:.1f}%")

# Test 6: Streamlit app import
print("\n✓ VERIFICANDO APP STREAMLIT...")

try:
    import streamlit as st
    print(f"  ✓ Streamlit versão {st.__version__} instalado")
except Exception as e:
    print(f"  ✗ Streamlit não encontrado: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ TUDO VERIFICADO COM SUCESSO! (4 modelos ML + recomendações)")
print("="*80)
print("\nPara iniciar a aplicação, execute:")
print("  streamlit run streamlit_app.py")
