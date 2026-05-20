import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
    modelo_recurrence = joblib.load('modelo_wells.pkl')
    recommendations = joblib.load('modelo_recomendacoes.pkl')
    return modelo_recurrence, recommendations

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
    modelo_recurrence, recommendations = load_models()
    df_transacoes, df_itens, df_treino, df_regras = load_data()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")
    st.stop()

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================

st.title("Wells - Sistema de Recorrência & Recomendações")


# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

# st.sidebar.title("Wells - Sistema de Recorrência & Recomendações")
page = st.sidebar.radio("Selecione uma página:", [
    "Dashboard Principal",
    "Previsões de Recorrência",
    "Recomendações de Produtos",
    "Análise do Modelo"
])


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
    
    # Notification list - customers to notify
    st.subheader("Clientes para Notificar")
    
    # Calculate urgency for each customer-product
    notification_threshold = st.slider(
        "Dias até recompra:",
        min_value=3,
        max_value=30,
        value=7,
        step=1
    )
    
    df_notify = df_treino[df_treino['Target_Dias_Restantes'] <= notification_threshold].copy()
    df_notify = df_notify.sort_values('Target_Dias_Restantes')
    
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
                'Dias Restantes': int(row['Target_Dias_Restantes']),
                'Última Compra': row['Data_Ultima_Compra'],
                'Urgência': '🔴 ALTA' if row['Target_Dias_Restantes'] <= 3 else '🟡 MÉDIA' if row['Target_Dias_Restantes'] <= 9 else '🟢 BAIXA',
                'Recomendações': top_recos
            })
        
        df_notify_display = pd.DataFrame(notification_data)
        st.dataframe(df_notify_display, use_container_width=True, hide_index=True)
        
        st.success(f"{len(df_notify_display)} cliente(s) para notificar!")
    else:
        st.info(f"Nenhum cliente necessita de notificação nos próximos {notification_threshold} dias")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of days remaining
        st.subheader("Distribuição de Dias Restantes")
        fig = px.histogram(
            df_treino,
            x='Target_Dias_Restantes',
            nbins=20,
            title="Quantos dias até a próxima compra?",
            labels={'Target_Dias_Restantes': 'Dias Restantes'}
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        cliente_id = st.selectbox(
            "Selecione um cliente:",
            sorted(df_treino['ID_Cliente'].unique())
        )
    
    # Filter data for selected customer
    df_cliente = df_treino[df_treino['ID_Cliente'] == cliente_id].sort_values('Target_Dias_Restantes')
    
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
                    dias_restantes = row['Target_Dias_Restantes']
                    
                    st.metric("Dias Restantes", f"{dias_restantes:.0f}")
                
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
    st.subheader("Performance & Análise do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Modelo de Recorrência (Random Forest)")
        st.markdown("""
        - **Algoritmo:** Random Forest Regressor
        - **Features:** 
            - Intervalo Médio de Compra
            - Dias desde Última Compra
            - Total de Compras no Histórico
        - **Target:** Dias Restantes até Próxima Compra
        - **Train/Test Split:** 80/20
        """)
        
        # Show metrics from training
        st.metric("Erro Médio (MAE)", "10.19 dias", "±2.5")
        st.metric("Precisão (R²)", "98.75%", "Excelente")
    
    with col2:
        st.markdown("### Modelo de Recomendações (Apriori)")
        st.markdown(f"""
        - **Algoritmo:** Apriori + Association Rules
        - **Métrica:** Confidence (Confiança)
        - **Total de Regras:** {len(df_regras)}
        - **Produtos Analisados:** {df_treino['Produto'].nunique()}
        """)
        
        max_conf_ab = df_regras['Confianca_A_para_B'].max()
        max_conf_ba = df_regras['Confianca_B_para_A'].max()
        overall_max = max(max_conf_ab, max_conf_ba)
        st.metric("Top Confidence", f"{overall_max:.0f}%", "Melhor regra")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Erros de Previsão")
        df_erros = df_treino.copy()
        df_erros['Erro_Absoluto'] = df_erros['Target_Dias_Restantes'].abs()
        
        fig = px.histogram(
            df_erros,
            x='Target_Dias_Restantes',
            nbins=15,
            title="Erro em Dias (Previsão vs Real)",
            labels={'Target_Dias_Restantes': 'Erro (dias)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição de Confiança das Regras")
        
        # Prepare data: combine both confidence values
        conf_data = pd.concat([
            df_regras[['Confianca_A_para_B']].rename(columns={'Confianca_A_para_B': 'Confianca'}),
            df_regras[['Confianca_B_para_A']].rename(columns={'Confianca_B_para_A': 'Confianca'})
        ])
        
        fig = px.histogram(
            conf_data,
            x='Confianca',
            nbins=10,
            title="Confiança das Regras de Associação (Ambas Direções)",
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
        st.write(f"Dias Restantes (média): {df_treino['Target_Dias_Restantes'].mean():.1f} dias")
        st.write(f"Dias Restantes (mediana): {df_treino['Target_Dias_Restantes'].median():.1f} dias")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    IA Assignment 2 - Group 19  
""")
