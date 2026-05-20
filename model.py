import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Carregar os Dados Processados (Features)
try:
    df = pd.read_csv('data/dados_treino_ia.csv')
    
    # Definir as variáveis de entrada (X) e o objetivo (y)
    # Usamos o Hábito, a Recência e a Fidelidade (Total de Compras)
    X = df[['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']]
    y = df['Target_Dias_Restantes']

    # 2. Divisão para Avaliação (Train/Test Split)
    # Importante para o Slide 7: avaliar em dados que a IA nunca viu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treino do Modelo
    # Usamos Random Forest por ser excelente a captar padrões de comportamento humano
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Avaliação 
    previsoes = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, previsoes)
    r2 = r2_score(y_test, previsoes)

    # 5. Guardar o Modelo Final
    joblib.dump(modelo, 'modelo_wells.pkl')
    
    # 6. Guardar as Métricas em JSON
    metrics = {
        'mae': float(mae),
        'r2': float(r2),
        'mae_formatted': f"{mae:.2f}",
        'r2_percentage': f"{r2*100:.2f}"
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nSucesso: Modelo 'modelo_wells.pkl' pronto!")

except FileNotFoundError:
    print("Erro: O ficheiro 'data/dados_treino_ia.csv' não existe. Corre o 'processar_dados.py' primeiro.")