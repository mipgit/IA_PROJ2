# 💊 Wells - Recurrence & Recommendation System POC

A proof-of-concept system that predicts when customers will repurchase products and recommends complementary items based on co-purchase patterns.

---

## 📋 Project Overview

### Problem
Pharmacy/cosmetics stores struggle to notify customers at the right time about repurchasing products before they run out, and miss cross-selling opportunities.

### Solution
Use Machine Learning to:
1. **Predict Recurrence**: When will each customer buy each product again?
2. **Recommend Complements**: What related products should be recommended?

---

## 🏗️ System Architecture

```
script.py                 → Generate synthetic data (40 clients, 12 products)
                          ├─ data/transacoes.csv (379 transactions)
                          └─ data/itens_transacao.csv (678 items)
                                    ↓
processing_data.py        → Extract features (normalized schema)
                          ├─ data/dados_treino_ia.csv (recurrence model)
                          └─ data/dados_cocompra.csv (raw co-purchase data)
                                    ↓
         ┌─────────────────────────┬──────────────────────────┐
         ↓                         ↓
model.py                  association_rules.py
├─ Train: Random Forest   ├─ Train: Apriori Algorithm
├─ Features: 3            ├─ Input: Multi-product transactions
├─ Target: Days remain    ├─ Output: Association rules (12 rules)
└─ Output: modelo_wells   └─ Output: modelo_recomendacoes.pkl
                                    ↓
streamlit_app.py          → Interactive Dashboard
├─ Dashboard Principal (notification list)
├─ Previsões de Recorrência (per customer)
├─ Recomendações (association rules)
└─ Análise do Modelo (performance metrics)
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn joblib mlxtend streamlit plotly
```

### 2. Run Full Pipeline
```bash
# Generate synthetic data (40 clients, multi-product transactions)
python3 script.py

# Process data and extract features
python3 processing_data.py

# Train recurrence prediction model
python3 model.py

# Train association rules model
python3 association_rules.py

# Verify everything works
python3 test_pipeline.py
```

### 3. Launch Dashboard
```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 Models Explained

### Model 1: Recurrence Prediction (Random Forest)

**What it does**: Predicts how many days until a customer repurchases a product.

**Features**:
- `Intervalo_Medio_Habito`: Average days between customer's purchases of this product
- `Dias_Desde_Ultima_Compra`: Days since last purchase
- `Total_Compras_Historico`: Total purchases of this product by customer

**Target**: `Target_Dias_Restantes` (days until next purchase)

**Performance**:
- MAE: 10.19 days
- R²: 98.75% (excellent fit)

**Example**:
```
Customer usually buys every 45 days
Last bought: 10 days ago
Prediction: 35 days remaining
→ Notify in 30 days
```

---

### Model 2: Product Recommendations (Apriori + Association Rules)

**What it does**: Discovers products that are frequently bought together.

**Algorithm**: 
- Apriori: Find frequent itemsets (products often in same transaction)
- Association Rules: Generate IF-THEN rules with confidence metrics

**Key Metrics**:
- **Support**: % of transactions containing this pair
- **Confidence**: % of times B is bought when A is bought
- **Lift**: How much more likely B appears with A vs independently

**Example Rule**:
```
IF customer buys: Soro Fisiológico (Pack 20)
THEN recommend: Lentes de Contacto (30 unidades)
Confidence: 79% | Support: 17% | Lift: 3.56x
```

**Top 3 Rules Found**:
1. Soro Fisiológico + Gel → Lentes (100% confidence)
2. Desodorizante → Gel (89% confidence)
3. Óculos de Sol → Protetor Solar (85% confidence)

---

## 📁 Data Schema

### Normalized Structure 

**`transacoes.csv`** (Transactions table)
```
ID_Transacao | ID_Cliente | Data       | Loja
100000       | 10000      | 2025-01-15 | Lisboa-Rossio
100001       | 10001      | 2025-01-16 | Online
```

**`itens_transacao.csv`** (Items per transaction)
```
ID_Transacao | Produto                  | Categoria       | Preco_Unitario | Quantidade
100000       | Lentes de Contacto (30)  | Óptica          | 24.99          | 1
100000       | Soro Fisiológico (Pack20)| Saúde Visual    | 3.99           | 2
100001       | Creme Hidratante Facial  | Dermocosmética  | 12.50          | 1
```

### Processed Features

**`dados_treino_ia.csv`** (Training data for recurrence model)
```
ID_Cliente | Produto | Intervalo_Medio_Habito | Dias_Desde_Ultima_Compra | Total_Compras_Historico | Target_Dias_Restantes
10000      | Lentes  | 30.5                   | 10                       | 8                       | 20.5
10001      | Creme   | 45.2                   | 5                        | 9                       | 40.2
```

**`regras_associacao.csv`** (Co-purchase pairs)
```
Produto_A,Produto_B,Confianca_A_para_B,Confianca_B_para_A,Co_Purchase_Rate,Lift,Transacoes_com_Ambos

```

---

## 🎯 Dashboard Features

### 📊 Dashboard Principal
- **KPI Cards**: Clients, Products, Transactions, Rules
- **Notification List**: Customers to notify today (sorted by urgency)
- **Charts**: 
  - Distribution of days remaining
  - Most popular products

### 🔮 Previsões de Recorrência
- Select customer by ID
- See all their products with:
  - Predicted days until repurchase
  - Urgency indicator (🔴 High / 🟡 Medium / 🟢 Low)
  - Last purchase date
  - Top 3 complementary recommendations

### 🎯 Recomendações
- Select any product
- See all association rules where it's the antecedent
- View confidence, support, and lift for each rule
- Full table of all discovered rules

### 📈 Análise do Modelo
- Model specifications
- Performance metrics (MAE, R²)
- Distribution charts
- Aggregate statistics

---

## 📊 Sample Data Statistics

**Generated Dataset**:
- Clients: 40
- Products: 12
- Transactions: 379
- Transaction Items: 678
- Avg items per transaction: 1.79
- Training examples: 112 (client-product pairs with 3+ purchases)
- Association rules found: 12

**Product Catalog**:
- **Óptica & Saúde Visual**: Lentes, Soro, Óculos
- **Dermocosmética**: Creme, Sérum, Protetor, Tónico
- **Suplementos**: Multivitamínico, Ómega-3, Proteína
- **Higiene**: Gel, Desodorizante

---

## 📚 Implementation Notes

### Why Random Forest for Recurrence?
- Captures non-linear relationships in purchase patterns
- Handles customer heterogeneity well
- Excellent for small datasets (112 examples)
- Interpretable feature importance

### Why Apriori for Recommendations?
- Discovers patterns automatically from data
- Explainable rules (high confidence = reliable)
- Industry standard for market basket analysis
- Good interpretability for presentations

### Data Generation Strategy
- **Realistic Product Affinities**: Built intentional co-purchase patterns
  - Skincare routine (Creme → Sérum, Protetor)
  - Eye care (Lentes → Soro)
  - Supplements stack (Multivitamínico → Ómega-3)
- **Multi-product Transactions**: 70% of transactions include complementary items
- **Individual Patterns**: Each customer has their own purchase frequency

---

## 🧪 Testing & Validation

Run the end-to-end test:
```bash
python3 test_pipeline.py
```

This verifies:
- ✓ All data files exist and have content
- ✓ Models load correctly
- ✓ Predictions generate valid outputs
- ✓ Recommendations available for all products
- ✓ Association rules extracted successfully
- ✓ Streamlit installed and ready

---

## 📝 Project Structure

```
IA_PROJ2/
├── script.py                          # Data generation
├── processing_data.py                 # Feature engineering
├── model.py                           # Recurrence model
├── association_rules.py               # Association rules model
├── streamlit_app.py                   # Dashboard
├── test_pipeline.py                   # End-to-end test
├── README.md                          # This file
├── modelo_wells.pkl                   # Trained recurrence model
├── modelo_recomendacoes.pkl           # Trained recommendation rules
└── data/
    ├── transacoes.csv                 # Raw transactions (normalized)
    ├── itens_transacao.csv            # Items per transaction (normalized)
    ├── dados_treino_ia.csv            # Features for recurrence model
    ├── dados_cocompra.csv             # Raw co-purchase data
    └── regras_associacao.csv          # Discovered association rules
```

---

## 🔄 Running the Full Pipeline 

```bash
python3 script.py && python3 processing_data.py && python3 model.py && python3 association_rules.py && python3 test_pipeline.py
```

Then:
```bash
streamlit run streamlit_app.py
```


