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
print("VERIFICAÇÃO DO PIPELINE WELLS")
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

try:
    modelo = joblib.load('modelo_wells.pkl')
    print(f"  ✓ modelo_wells.pkl carregado")
except Exception as e:
    print(f"  ✗ modelo_wells.pkl: {e}")
    sys.exit(1)

try:
    recommendations = joblib.load('modelo_recomendacoes.pkl')
    print(f"  ✓ modelo_recomendacoes.pkl carregado ({len(recommendations)} produtos)")
except Exception as e:
    print(f"  ✗ modelo_recomendacoes.pkl: {e}")
    sys.exit(1)

# Test 3: Test recurrence model prediction
print("\n✓ TESTANDO MODELO DE RECORRÊNCIA...")

df_treino = pd.read_csv('data/dados_treino_ia.csv')
X_test = df_treino[['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']].head(5)

try:
    predictions = modelo.predict(X_test)
    print(f"  ✓ Previsões geradas: {len(predictions)} (média: {predictions.mean():.1f} dias)")
    
    # Show sample prediction
    print(f"\n  Exemplo de Previsão:")
    for i, (_, row) in enumerate(df_treino.head(3).iterrows()):
        X = np.array([[
            row['Dias_Desde_Ultima_Compra'],
            row['Intervalo_Medio_Habito'],
            row['Total_Compras_Historico']
        ]])
        pred = modelo.predict(X)[0]
        print(f"    - {row['Produto']}: {pred:.1f} dias até próxima compra")
except Exception as e:
    print(f"  ✗ Erro na previsão: {e}")
    sys.exit(1)

# Test 4: Test recommendations
print("\n✓ TESTANDO RECOMENDAÇÕES...")

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
print("✓ TUDO VERIFICADO COM SUCESSO!")
print("="*80)
print("\nPara iniciar a aplicação, execute:")
print("  streamlit run streamlit_app.py")
