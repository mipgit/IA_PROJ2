import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Wells - Sistema de Recomendações",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    modelo_wells = joblib.load('models/modelo_wells.pkl')
    modelo_linear = joblib.load('models/modelo_linear.pkl')
    modelo_dt = joblib.load('models/modelo_dt.pkl')
    modelo_rf = joblib.load('models/modelo_rf.pkl')
    modelo_gb = joblib.load('models/modelo_gb.pkl')
    recommendations = joblib.load('models/modelo_recomendacoes.pkl')
    return modelo_wells, modelo_linear, modelo_dt, modelo_rf, modelo_gb, recommendations

@st.cache_data
def load_data():
    df_transacoes = pd.read_csv('data/transacoes.csv')
    df_itens = pd.read_csv('data/itens_transacao.csv')
    df_treino = pd.read_csv('data/dados_treino_ia.csv')
    df_regras = pd.read_csv('data/regras_associacao.csv')
    
    df_transacoes['Data'] = pd.to_datetime(df_transacoes['Data'])
    df_treino['Data_Ultima_Compra'] = pd.to_datetime(df_treino['Data_Ultima_Compra'])
    
    return df_transacoes, df_itens, df_treino, df_regras

try:
    modelo_wells, modelo_linear, modelo_dt, modelo_rf, modelo_gb, recommendations = load_models()
    df_transacoes, df_itens, df_treino, df_regras = load_data()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")
    st.stop()

modelos_map = {
    'LinearRegression': ('Regressão Linear', modelo_linear),
    'DecisionTree': ('Árvore Decisão', modelo_dt),
    'RandomForest': ('Random Forest', modelo_rf),
    'GradientBoosting': ('Gradient Boosting', modelo_gb),
}

modelo_keys = list(modelos_map.keys())
default_idx = modelo_keys.index('GradientBoosting')

FEATURES = ['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================

st.title("Wells - Sistema de Recorrência & Recomendações")


# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

page = st.sidebar.radio("Selecione uma página:", [
    "Dashboard Principal",
    "Previsões de Recorrência",
    "Recomendações de Produtos",
    "Análise do Modelo"
])

if 'algo_key' not in st.session_state:
    st.session_state.algo_key = 'GradientBoosting'

algo_key = st.session_state.algo_key
modelo_ativo = modelos_map[algo_key][1]
modelo_label = modelos_map[algo_key][0]

# Precompute predictions for the selected model
df_pred = df_treino.copy()
df_pred['Previsao_Dias_Restantes'] = modelo_ativo.predict(df_treino[FEATURES])


# ============================================================================
# PAGE: DASHBOARD PRINCIPAL
# ============================================================================

if page == "Dashboard Principal":
    st.subheader("Visão Geral do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clientes", df_treino['ID_Cliente'].nunique())
    
    with col2:
        st.metric("Produtos Únicos", df_treino['Produto'].nunique())
    
    with col3:
        st.metric("Total de Transações", len(df_transacoes))
    
    with col4:
        st.metric("Regras de Associação", len(df_regras))
    
    st.divider()
    
    # Model selection
    st.subheader("⚙️ Selecionar Algoritmo de Previsão")
    col_algo = st.columns([2, 4])
    with col_algo[0]:
        novo_algo = st.selectbox(
            "Algoritmo:",
            options=modelo_keys,
            format_func=lambda k: modelos_map[k][0],
            index=modelo_keys.index(st.session_state.algo_key),
            key='algo_selector',
        )
        if novo_algo != st.session_state.algo_key:
            st.session_state.algo_key = novo_algo
            st.rerun()
    st.divider()
    
    # Notification list - customers to notify TODAY
    st.subheader("Clientes para Notificar HOJE")
    
    # Calculate urgency for each customer-product
    notification_threshold = st.slider(
        "Dias até recompra:",
        min_value=3,
        max_value=30,
        value=7,
        step=1
    )
    
    df_notify = df_pred[df_pred['Previsao_Dias_Restantes'] <= notification_threshold].copy()
    df_notify = df_notify.sort_values('Previsao_Dias_Restantes')
    
    if len(df_notify) > 0:
        # Get top recommendations for each product
        notification_data = []
        
        for _, row in df_notify.iterrows():
            produto = row['Produto']
            recos = recommendations.get(produto, [])
            
            top_recos = ", ".join([f"{r['produto']}" for r in recos[:2]]) if recos else "Sem recomendações"
            
            notification_data.append({
                'Cliente': row['ID_Cliente'],
                'Produto': produto,
                'Previsão (dias)': int(row['Previsao_Dias_Restantes']),
                'Última Compra': row['Data_Ultima_Compra'],
                'Urgência': '🔴 ALTA' if row['Previsao_Dias_Restantes'] <= 3 else '🟡 MÉDIA' if row['Previsao_Dias_Restantes'] <= 9 else '🟢 BAIXA',
                'Recomendações': top_recos
            })
        
        df_notify_display = pd.DataFrame(notification_data)
        st.dataframe(df_notify_display, use_container_width=True, hide_index=True)
        
        st.success(f"✓ {len(df_notify_display)} cliente(s) para notificar!")
    else:
        st.info(f"✓ Nenhum cliente necessita de notificação nos próximos {notification_threshold} dias")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Dias Restantes")
        fig = px.histogram(
            df_pred, x='Previsao_Dias_Restantes',
            nbins=20,
            title="Dias previstos até próxima compra",
            labels={'Previsao_Dias_Restantes': 'Dias Restantes (previsão)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top products by frequency
        st.subheader("Produtos Mais Comprados")
        top_produtos = df_treino.groupby('Produto')['Total_Compras_Historico'].sum().sort_values(ascending=False).head(8)
        fig = px.bar(
            x=top_produtos.index,
            y=top_produtos.values,
            title="Histórico Total de Compras",
            labels={'x': 'Produto', 'y': 'Total de Compras'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: PREVISÕES DE RECORRÊNCIA
# ============================================================================

elif page == "Previsões de Recorrência":
    st.subheader("Previsões de Recorrência por Cliente")
    st.caption(f"Modelo ativo: **{modelo_label}**")
    
    cliente_id = st.selectbox(
        "Selecione um cliente:",
        sorted(df_treino['ID_Cliente'].unique())
    )
    
    # Filter data for selected customer
    df_cliente = df_pred[df_pred['ID_Cliente'] == cliente_id].sort_values('Previsao_Dias_Restantes')
    
    if len(df_cliente) > 0:
        st.markdown(f"### Cliente {cliente_id}")
        
        # Display products for this customer
        for _, row in df_cliente.iterrows():
            with st.container(border=True):
                col_prod, col_info = st.columns([2, 3])
                
                with col_prod:
                    st.markdown(f"**{row['Produto']}**")
                    st.caption(f"{row['Categoria']} | {row['Marca']}")
                
                with col_info:
                    pred = row['Previsao_Dias_Restantes']
                    st.metric("Previsão", f"{pred:.0f} dias")
                
                # Additional info
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.caption(f"Intervalo Médio: {row['Intervalo_Medio_Habito']:.0f} dias")
                with col_b:
                    st.caption(f"Última Compra: {row['Data_Ultima_Compra']}")
                with col_c:
                    st.caption(f"Total de Compras: {row['Total_Compras_Historico']:.0f}x")
                
                # Get recommendations
                recos = recommendations.get(row['Produto'], [])
                if recos:
                    st.caption("**Recomendações complementares:**")
                    for i, reco in enumerate(recos[:3], 1):
                        st.caption(f"{i}. {reco['produto']} ({reco['confianca']:.0f}% confiança)")
    else:
        st.warning(f"Sem dados para cliente {cliente_id}")
    
 

# ============================================================================
# PAGE: RECOMENDAÇÕES
# ============================================================================

elif page == "Recomendações de Produtos":
    st.subheader("Recomendações de Produtos Complementares")
    
    st.markdown("""
         Este sistema utiliza **Association Rules** (Market Basket Analysis) para descobrir 
         quais produtos são frequentemente comprados juntos.
    """)
    
    # Select product
    produtos = sorted(df_treino['Produto'].unique())
    produto_selecionado = st.selectbox("Selecione um produto:", produtos)
    
    st.divider()
    
    # Find all rules related to this product (both directions)
    regras_relacionadas = []
    
    for _, rule in df_regras.iterrows():
        prod_a = rule['Produto_A']
        prod_b = rule['Produto_B']
        
        # If this product is involved in the rule
        if prod_a == produto_selecionado:
            # This product is A, looking at confidence A→B
            regras_relacionadas.append({
                'outro_produto': prod_b,
                'confianca': rule['Confianca_A_para_B'],
                'confianca_reversa': rule['Confianca_B_para_A'],
                'co_purchase': rule['Co_Purchase_Rate'],
                'lift': rule['Lift'],
                'direcao': f"Se compra {prod_a}"
            })
        elif prod_b == produto_selecionado:
            # This product is B, looking at confidence B→A
            regras_relacionadas.append({
                'outro_produto': prod_a,
                'confianca': rule['Confianca_B_para_A'],
                'confianca_reversa': rule['Confianca_A_para_B'],
                'co_purchase': rule['Co_Purchase_Rate'],
                'lift': rule['Lift'],
                'direcao': f"Se compra {prod_b}"
            })
    
    if len(regras_relacionadas) > 0:
        st.subheader(f"Frequentemente comprados juntos: **{produto_selecionado}**")
        
        for reg in sorted(regras_relacionadas, key=lambda x: x['confianca'], reverse=True):
            with st.container(border=True):
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown(f"**{reg['outro_produto']}**")
                
                with col2:
                    st.metric("Confiança", f"{reg['confianca']:.1f}%")
                
                # Explanation
                col_exp, col_co, col_lift = st.columns(3)
                with col_exp:
                    st.caption(f"Se cliente compra {produto_selecionado}, probabilidade de comprar também: **{reg['confianca']:.1f}%**")
                with col_co:
                    st.caption(f"Co-compra: {reg['co_purchase']:.1f}% das transações têm ambos")
                with col_lift:
                    st.caption(f"Lift: {reg['lift']:.2f}x (relação de força)")
    else:
        st.info(f"Sem recomendações descobertas para **{produto_selecionado}**")
    
    st.divider()
    
    # All rules table with both directional confidences
    st.subheader("Todas as Regras de Associação")
    
    df_regras_display = df_regras[[
        'Produto_A', 'Produto_B', 'Confianca_A_para_B', 'Confianca_B_para_A', 
        'Co_Purchase_Rate', 'Lift'
    ]].copy()
    
    df_regras_display.columns = [
        'Produto A', 'Produto B', 'Confiança A→B', 'Confiança B→A', 
        'Co-Purchase Rate (%)', 'Lift'
    ]
    
    st.dataframe(df_regras_display, use_container_width=True, hide_index=True)
    
    

# ============================================================================
# PAGE: ANÁLISE DO MODELO
# ============================================================================

elif page == "Análise do Modelo":
    st.subheader("Performance & Comparação de Modelos")
    
    st.markdown("### Modelos de Recorrência (4 Algoritmos)")
    st.markdown("""
    - **Features:** Intervalo Médio de Compra, Dias desde Última Compra, Total de Compras no Histórico
    - **Target:** Dias Restantes até Próxima Compra
    - **Train/Test Split:** 80/20 com validação cruzada (5-fold)
    - **Dados com ruído gaussiano (σ=5) para simular cenário real**
    """)
    
    st.divider()
    
    # Compute predictions for all models on a sample
    X_all = df_treino[['Dias_Desde_Ultima_Compra', 'Intervalo_Medio_Habito', 'Total_Compras_Historico']]
    y_true = df_treino['Target_Dias_Restantes']
    
    comparison_data = []
    for key, (label, model) in modelos_map.items():
        y_pred = model.predict(X_all)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        comparison_data.append({
            'Modelo': label,
            'MAE (dias)': round(mae, 2),
            'RMSE (dias)': round(rmse, 2),
            'R² Score': round(r2, 4)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('MAE (dias)')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    with col2:
        best_model = df_comparison.iloc[0]['Modelo']
        best_mae = df_comparison.iloc[0]['MAE (dias)']
        st.success(f"**Melhor modelo:** {best_model}")
        st.metric("MAE do melhor modelo", f"{best_mae:.2f} dias", "Menor = melhor")
    
    st.divider()
    
    # Bar chart comparing models
    st.subheader("Comparação Visual de Performance")
    fig = px.bar(
        df_comparison,
        x='Modelo',
        y='MAE (dias)',
        color='Modelo',
        title="MAE por Algoritmo (menor = melhor)",
        text='MAE (dias)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Modelo de Recomendações (Association Rules)")
        st.markdown(f"""
        - **Algoritmo:** Co-ocorrência + Association Rules
        - **Métrica:** Confidence (Confiança) e Lift
        - **Total de Regras:** {len(df_regras)}
        - **Produtos com Recomendações:** {df_treino['Produto'].nunique()}
        """)
        
        max_conf_ab = df_regras['Confianca_A_para_B'].max()
        max_conf_ba = df_regras['Confianca_B_para_A'].max()
        overall_max = max(max_conf_ab, max_conf_ba)
        st.metric("Top Confidence", f"{overall_max:.0f}%", "Melhor regra")
    
    with col2:
        st.subheader("Distribuição de Confiança das Regras")
        conf_data = pd.concat([
            df_regras[['Confianca_A_para_B']].rename(columns={'Confianca_A_para_B': 'Confianca'}),
            df_regras[['Confianca_B_para_A']].rename(columns={'Confianca_B_para_A': 'Confianca'})
        ])
        fig = px.histogram(
            conf_data,
            x='Confianca',
            nbins=10,
            title="Confiança das Regras (Ambas Direções)",
            labels={'Confianca': 'Confiança (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Estatísticas dos Dados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Recorrência**")
        st.write(f"Intervalo Médio (média): {df_treino['Intervalo_Medio_Habito'].mean():.1f} dias")
        st.write(f"Intervalo Médio (mediana): {df_treino['Intervalo_Medio_Habito'].median():.1f} dias")
    
    with col2:
        st.markdown("**Histórico de Compras**")
        st.write(f"Compras por Cliente-Produto (média): {df_treino['Total_Compras_Historico'].mean():.1f}x")
        st.write(f"Compras por Cliente-Produto (max): {df_treino['Total_Compras_Historico'].max():.0f}x")
    
    with col3:
        st.markdown("**Previsões**")
        st.write(f"Dias Restantes (média): {df_pred['Previsao_Dias_Restantes'].mean():.1f} dias")
        st.write(f"Dias Restantes (mediana): {df_pred['Previsao_Dias_Restantes'].median():.1f} dias")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    IA Assignment 2 - Group 19  
""")
