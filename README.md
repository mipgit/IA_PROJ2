# Wells - Recurrence & Recommendation System POC

A proof-of-concept system that predicts when customers will repurchase products and recommends complementary items based on co-purchase patterns.

---

## Project Overview

### Problem
Pharmacy/cosmetics stores struggle to notify customers at the right time about repurchasing products before they run out, and miss cross-selling opportunities.

### Solution
Use Machine Learning to:
1. **Predict Recurrence**: When will each customer buy each product again?
2. **Recommend Complements**: What related products should be recommended?

---

## System Architecture

```
script.py                 → Generate synthetic data (40 clients, 12 products)
                          ├─ data/transacoes.csv (~450 transactions)
                          └─ data/itens_transacao.csv (~790 items)
                                    ↓
processing_data.py        → Extract features with gaussian noise (σ=5)
                          ├─ data/dados_treino_ia.csv (recurrence model)
                          └─ data/dados_cocompra.csv (raw co-purchase data)
                                    ↓
          ┌─────────────────────────┬──────────────────────────┐
          ↓                         ↓
model.py                  association_rules.py
├─ 4 ML algorithms        ├─ Co-occurrence + Association Rules
├─ RandomizedSearchCV     ├─ Confidence, Lift, Support
├─ Automatic best pick    └─ Output: models/modelo_recomendacoes.pkl
└─ Output: models/*.pkl
                                    ↓
streamlit_app.py          → Interactive Dashboard
├─ Dashboard Principal (notification list)
├─ Previsões de Recorrência (per customer)
├─ Recomendações (association rules)
└─ Análise do Modelo (4-model comparison)
```

---

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib streamlit plotly
```

### 2. Run Full Pipeline
```bash
source .venv/bin/activate
python3 script.py
python3 processing_data.py
python3 model.py
python3 association_rules.py
python3 test_pipeline.py
```

### 3. Launch Dashboard
```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## Models Explained

### Recurrence Prediction (4 Algorithms)

**What it does**: Predicts how many days until a customer repurchases a product.

**Features**:
- `Intervalo_Medio_Habito`: Average days between customer's purchases of this product
- `Dias_Desde_Ultima_Compra`: Days since last purchase
- `Total_Compras_Historico`: Total purchases of this product by customer

**Target**: `Target_Dias_Restantes` (days until next purchase)

**Data**: 101 training examples with gaussian noise (σ=5) added to simulate real-world variance.

**Algorithms & Performance** (evaluated on full dataset):

| Model | MAE | RMSE | R² | Tuning |
|-------|-----|------|----|--------|
| Gradient Boosting | 2.44 | 7.60 | 0.975 | RandomizedSearchCV (200 estimators, lr=0.1) |
| Linear Regression | 3.43 | 4.51 | 0.991 | None (pipeline with scaler) |
| Random Forest | 7.00 | 13.36 | 0.923 | RandomizedSearchCV (100 estimators, depth=20) |
| Decision Tree | 8.49 | 13.65 | 0.920 | RandomizedSearchCV (depth=10, min_samples=5) |

**Best model**: Gradient Boosting — sequential trees with shallow depth (3) and slow learning rate (0.1), making it robust to noise.

### Product Recommendations (Association Rules)

**What it does**: Discovers products that are frequently bought together.

**Algorithm**: Co-occurrence counting with directional confidence and lift metrics.

**Key Metrics**:
- **Confidence A→B**: % of times B is bought when A is bought
- **Lift**: How much more likely B appears with A vs independently
- **Co-Purchase Rate**: % of all transactions containing both products

**Top 3 Rules**:
1. Desodorizante ↔ Gel Higienizante (81.5% confiança, Lift: 3.86x)
2. Multivitamínico ↔ Ómega-3 (83.8% confiança, Lift: 4.39x)
3. Protetor Solar ↔ Óculos de Sol (73.9% confiança, Lift: 4.62x)

---

## Data Schema

### Normalized Structure

**`transacoes.csv`**
```
ID_Transacao | ID_Cliente | Data       | Loja
100000       | 10000      | 2025-01-15 | Lisboa-Rossio
```

**`itens_transacao.csv`**
```
ID_Transacao | Produto                  | Categoria      | Preco_Unitario | Quantidade
100000       | Lentes de Contacto (30)  | Óptica         | 24.99          | 1
```

### Processed Features

**`dados_treino_ia.csv`**
```
ID_Cliente | Produto | Intervalo_Medio_Habito | Dias_Desde_Ultima_Compra | Total_Compras_Historico | Target_Dias_Restantes
10000      | Lentes  | 30.5                   | 10                       | 8                       | 20.5
```

---

## Dashboard Features

### Dashboard Principal
- **KPI Cards**: Clients, Products, Transactions, Rules
- **Notification List**: Customers to notify today (sorted by urgency)
- **Charts**: Distribution of days remaining, most popular products

### Previsões de Recorrência
- Select customer by ID
- See all products with predicted days until repurchase, last purchase date, and top 3 complementary recommendations

### Recomendações de Produtos
- Select any product to see related association rules
- View confidence, co-purchase rate, and lift for each rule
- Full table of all discovered rules

### Análise do Modelo
- Live comparison of all 4 algorithms (MAE, RMSE, R²)
- Bar chart ranking models by performance
- Association rules distribution
- Aggregate data statistics

---

## Data Statistics

**Generated Dataset**:
- Clients: 40
- Products: 12
- Transactions: ~450
- Transaction Items: ~790
- Training examples: ~101 (client-product pairs with 3+ purchases)
- Association rules: 14
- Gaussian noise: σ=5 on target variable

**Product Catalog**:
- **Óptica & Saúde Visual**: Lentes, Soro, Óculos
- **Dermocosmética**: Creme, Sérum, Protetor, Tónico
- **Suplementos**: Multivitamínico, Ómega-3, Proteína
- **Higiene**: Gel, Desodorizante

---

## Implementation Notes

### Why Gradient Boosting for Recurrence?
- Sequential tree correction handles noise better than single models
- Shallow trees (max_depth=3) and low learning rate prevent overfitting with only 101 samples
- Outperforms both linear models (on noisy data) and single decision trees (on variance)

### Why 4 Algorithms?
- **Linear Regression**: Baseline for when the relationship is near-linear
- **Decision Tree**: Simple, interpretable rules
- **Random Forest**: Bagging ensemble for variance reduction
- **Gradient Boosting**: Boosting ensemble for bias reduction

### Why Association Rules for Recommendations?
- Discovers patterns automatically from transaction data
- Explainable rules with confidence and lift metrics
- Industry standard for market basket analysis

### Data Generation Strategy
- **Realistic Product Affinities**: Built intentional co-purchase patterns
  - Skincare routine (Creme → Sérum, Protetor)
  - Eye care (Lentes → Soro)
  - Supplements stack (Multivitamínico → Ómega-3)
- **Multi-product Transactions**: 70% include complementary items
- **Gaussian Noise (σ=5)**: Added to target to simulate real-world variance

---

## Testing & Validation

Run the end-to-end test:
```bash
python3 test_pipeline.py
```

This verifies:
- All data files exist and have content
- All 5 models load correctly (linear, dt, rf, gb, wells + recommendations)
- All models generate valid predictions
- Recommendations available for all 12 products
- Association rules extracted successfully
- Streamlit installed and ready

---

## Project Structure

```
IA_PROJ2/
├── script.py                          # Data generation
├── processing_data.py                 # Feature engineering (with noise)
├── model.py                           # 4 ML regression algorithms
├── association_rules.py               # Association rules model
├── streamlit_app.py                   # Dashboard
├── test_pipeline.py                   # End-to-end test
├── README.md                          # This file
├── data/
│   ├── transacoes.csv                 # Raw transactions (normalized)
│   ├── itens_transacao.csv            # Items per transaction (normalized)
│   ├── dados_treino_ia.csv            # Features for recurrence model (101 examples)
│   ├── dados_cocompra.csv             # Raw co-purchase data
│   └── regras_associacao.csv          # Discovered association rules
└── models/
    ├── modelo_wells.pkl               # Best model (Gradient Boosting)
    ├── modelo_linear.pkl              # Linear Regression
    ├── modelo_dt.pkl                  # Decision Tree
    ├── modelo_rf.pkl                  # Random Forest
    ├── modelo_gb.pkl                  # Gradient Boosting
    └── modelo_recomendacoes.pkl       # Recommendations lookup
```

---

## Running the Full Pipeline

```bash
source .venv/bin/activate && python3 script.py && python3 processing_data.py && python3 model.py && python3 association_rules.py && python3 test_pipeline.py
```

Then:
```bash
streamlit run streamlit_app.py
```
