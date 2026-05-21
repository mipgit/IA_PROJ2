import pandas as pd
import numpy as np
import joblib
import shutil
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 20

FEATURES = ['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']
TARGET = 'Target_Dias_Restantes'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    df = pd.read_csv('data/dados_treino_ia.csv')
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

# ============================================================================
# MODELS REGISTRY — define all models and their hyperparameter grids
# ============================================================================

def baseline_predict(X_df):
    pred = (X_df['Intervalo_Medio_Habito'] - X_df['Dias_Desde_Ultima_Compra']).clip(lower=0)
    return pred.values

MODELS = [
    {
        'name': 'Regressão Linear',
        'key': 'LinearRegression',
        'model': make_pipeline(StandardScaler(), LinearRegression()),
        'params': None,
        'file': 'models/modelo_linear.pkl',
    },
    {
        'name': 'Árvore Decisão',
        'key': 'DecisionTree',
        'model': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
        },
        'file': 'models/modelo_dt.pkl',
    },
    {
        'name': 'Random Forest',
        'key': 'RandomForest',
        'model': RandomForestRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5],
        },
        'file': 'models/modelo_rf.pkl',
    },
    {
        'name': 'Gradient Boosting',
        'key': 'GradientBoosting',
        'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
        },
        'file': 'models/modelo_gb.pkl',
    },
]

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model_def, X_train, y_train):
    if model_def['params']:
        rs = RandomizedSearchCV(
            model_def['model'],
            model_def['params'],
            n_iter=N_ITER,
            scoring='neg_mean_absolute_error',
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rs.fit(X_train, y_train)
        best = rs.best_estimator_
        print(f"  ✓ {model_def['name']:20} best params: {rs.best_params_}")
        return best
    else:
        model_def['model'].fit(X_train, y_train)
        print(f"  ✓ {model_def['name']:20} (sem hyperparameter tuning)")
        return model_def['model']

def evaluate_model(model, X_full, y_full):
    y_pred = model.predict(X_full)
    return {
        'MAE': mean_absolute_error(y_full, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_full, y_pred)),
        'R2': r2_score(y_full, y_pred),
    }

def print_comparison(results):
    print('\n' + '='*70)
    print('COMPARAÇÃO DE ALGORITMOS (dataset completo)')
    print('='*70)
    print(f"{'Modelo':20} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print('-'*44)
    for name, m in results.items():
        print(f"{name:20} {m['MAE']:8.3f} {m['RMSE']:8.3f} {m['R2']:8.3f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    results = {}

    # Baseline
    y_pred_baseline = baseline_predict(X)
    results['Baseline'] = {
        'MAE': mean_absolute_error(y, y_pred_baseline),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_baseline)),
        'R2': r2_score(y, y_pred_baseline),
    }
    print(f"  ✓ Baseline (fórmula determinística)")

    # Train + evaluate each model
    for model_def in MODELS:
        trained = train_model(model_def, X_train, y_train)
        metrics = evaluate_model(trained, X, y)
        results[model_def['key']] = metrics
        joblib.dump(trained, model_def['file'])

    # Pick best and copy to modelo_wells.pkl
    valid = {k: v for k, v in results.items() if k != 'Baseline'}
    best_key = min(valid, key=lambda k: valid[k]['MAE'])
    best_file = next(m['file'] for m in MODELS if m['key'] == best_key)
    shutil.copy(best_file, 'models/modelo_wells.pkl')

    print_comparison(results)
    print(f'\nMelhor modelo: {best_key} (MAE={valid[best_key]["MAE"]:.3f})')
    print(f'Guardado como: models/modelo_wells.pkl')

if __name__ == '__main__':
    main()
