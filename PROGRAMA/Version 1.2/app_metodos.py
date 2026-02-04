import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Analizador Numérico BTC/USDT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para un acabado profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #00ff88; }
    .stAlert { border-radius: 8px; }
    h1, h2, h3 { color: #ffffff; }
    .report-box { background-color: #1e2130; padding: 20px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA Y PROCESAMIENTO DE DATOS (FASE 1) ---
@st.cache_data
def cargar_y_limpiar_datos():
    df = pd.read_csv('CRYPTO_BTC_USD.csv', sep=';')
    df['open_time'] = pd.to_datetime(df['open_time']).dt.tz_localize(None)
    
    def normalizar_valor(texto):
        if isinstance(texto, str):
            return float(texto.replace('.', '')) / 100000000
        return texto

    columnas_precio = ['open_price', 'high_price', 'low_price', 'close_price']
    for col in columnas_precio:
        df[col] = df[col].apply(normalizar_valor)
    
    df = df.sort_values('open_time').reset_index(drop=True)
    return df

# --- COMPONENTES DE LA BARRA LATERAL (Panel de Navegación) ---
st.sidebar.header("Panel de Navegacion")
fase = st.sidebar.selectbox(
    "Seleccionar Fase del Proyecto",
    ["Fase 1: Preprocesamiento", 
     "Fase 2: Volatilidad", 
     "Fase 3: Regresion Polinomial", 
     "Fase 4: Splines Cubicos", 
     "Fase 5: Reporte y Recomendacion"]
)

st.sidebar.divider()
st.sidebar.header("Configuracion de Datos")

df_raw = cargar_y_limpiar_datos()
fecha_min, fecha_max = df_raw['open_time'].min().date(), df_raw['open_time'].max().date()

start_date = st.sidebar.date_input("Fecha de Inicio", fecha_min, min_value=fecha_min, max_value=fecha_max)
end_date = st.sidebar.date_input("Fecha de Fin", fecha_min + timedelta(days=5), min_value=fecha_min, max_value=fecha_max)

# Parámetros Globales
grado_poli = st.sidebar.slider("Grado del Polinomio", 1, 3, 2)
precio_objetivo = st.sidebar.number_input("Precio Objetivo (USD)", value=12000)

mask = (df_raw['open_time'].dt.date >= start_date) & (df_raw['open_time'].dt.date <= end_date)
df = df_raw.loc[mask].copy()
df['t'] = np.arange(len(df))

if df.empty:
    st.error("No hay datos en el rango seleccionado.")
    st.stop()

# --- TÍTULO PRINCIPAL ---
st.title("Análisis Numérico Aplicado: Mercado BTC/USDT")
st.caption(f"Visualización profesional basada en el dataset CRYPTO_BTC_USD | Rango: {start_date} a {end_date}")

# --- CONTENIDO POR FASES ---

if fase == "Fase 1: Preprocesamiento":
    st.header("Fase 1: Preprocesamiento y Aritmetica")
    st.markdown("""
    En esta etapa se transforman los datos crudos a un formato numerico estable. 
    Se aplica el estandar **IEEE 754** para garantizar precision en los calculos de punto flotante y se genera la variable temporal $t$.
    """)
    col_info, col_data = st.columns([1, 2])
    with col_info:
        st.info(f"**Muestras:** {len(df)} horas\n\n**Variable t:** Linealizada para modelos.")
        st.write("**Precios Normalizados:**")
        st.dataframe(df[['open_time', 'close_price']].head(10))
    with col_data:
        fig = go.Figure(go.Scatter(x=df['t'], y=df['close_price'], name="Precio", line=dict(color='#00ff88')))
        fig.update_layout(template="plotly_dark", height=400, xaxis_title="Tiempo (t)", yaxis_title="Precio (USD)")
        st.plotly_chart(fig, use_container_width=True)

elif fase == "Fase 2: Volatilidad":
    st.header("Fase 2: Dinamica de Precio y Volatilidad")
    st.markdown("""
    **Analisis Operativo:** El rango de volatilidad define la "zona de ruido" del mercado. 
    Una amplitud mayor entre las lineas de soporte y resistencia (sombras) indica una alta incertidumbre y riesgo, 
    mientras que rangos estrechos sugieren una fase de consolidacion de precio.
    """)
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df['open_time'], y=df['high_price'], mode='lines', line=dict(width=0), showlegend=False))
    fig_vol.add_trace(go.Scatter(x=df['open_time'], y=df['low_price'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', name="Rango Volatilidad"))
    fig_vol.add_trace(go.Scatter(x=df['open_time'], y=df['close_price'], name="BTC/USDT", line=dict(color='#f7931a', width=2)))
    fig_vol.update_layout(template="plotly_dark", height=500, xaxis_title="Cronologia", yaxis_title="Precio (USD)")
    st.plotly_chart(fig_vol, use_container_width=True)

elif fase == "Fase 3: Regresion Polinomial":
    st.header("Fase 3: Modelado de Tendencia (Minimos Cuadrados)")
    coef = np.polyfit(df['t'], df['close_price'], grado_poli)
    modelo = np.poly1d(coef)
    y_tendencia = modelo(df['t'])
    r2 = r2_score(df['close_price'], y_tendencia)
    
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=df['open_time'], y=df['close_price'], name="Real", mode='markers', marker=dict(size=4, opacity=0.3)))
    fig_reg.add_trace(go.Scatter(x=df['open_time'], y=y_tendencia, name="Tendencia", line=dict(color='red', width=3)))
    fig_reg.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_reg, use_container_width=True)
    
    st.subheader("Analisis Matematico")
    st.latex(f"P(t) = {modelo}")
    st.write(f"**Coeficiente R²:** `{r2:.4f}` | **Inercia:** {'Alcista' if coef[0] > 0 else 'Bajista'}")
    st.info("Este modelo de minimos cuadrados filtra el ruido horario para identificar la direccion macro del precio.")

elif fase == "Fase 4: Splines Cubicos":
    st.header("Fase 4: Interpolacion y Derivadas Numericas")
    t_vals, y_vals = df['t'].values, df['close_price'].values
    spline = CubicSpline(t_vals, y_vals, bc_type='natural')
    t_denso = np.linspace(t_vals.min(), t_vals.max(), 500)
    
    # Visualizacion uno bajo el otro
    st.subheader("1. Trayectoria Suavizada S(t)")
    st.markdown("Reconstruccion exacta punto a punto. Define el camino mas probable entre horas.")
    fig1 = go.Figure(go.Scatter(x=t_denso, y=spline(t_denso), name="Spline", line=dict(color='#00cf46')))
    fig1.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Velocidad de Mercado S'(t)")
    st.markdown("Indica el cambio de precio por unidad de tiempo (USD/h). Valores positivos indican momentum alcista.")
    v = spline(t_denso, 1)
    fig2 = go.Figure(go.Scatter(x=t_denso, y=v, name="Velocidad", line=dict(color='cyan')))
    fig2.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Aceleracion Numerica S''(t)")
    st.markdown("Muestra la variacion de la velocidad. Los cruces por cero indican cambios criticos en la fuerza del movimiento.")
    a = spline(t_denso, 2)
    fig3 = go.Figure(go.Scatter(x=t_denso, y=a, name="Aceleracion", line=dict(color='magenta')))
    fig3.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

elif fase == "Fase 5: Reporte y Recomendacion":
    st.header("Fase 5: Reporte Final de Metricas y Recomendacion")
    
    # Logica de validacion
    n_test = 24 if len(df) > 48 else int(len(df)*0.2)
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    coef_fut = np.polyfit(df['t'], df['close_price'], grado_poli)
    modelo_fut = np.poly1d(coef_fut)
    
    # Error
    y_pred_val = np.poly1d(np.polyfit(train['t'], train['close_price'], grado_poli))(test['t'])
    rmse = np.sqrt(mean_squared_error(test['close_price'], y_pred_val))
    error_rel = (rmse / test['close_price'].mean()) * 100
    
    # Grafica
    fig_f5 = go.Figure()
    fig_f5.add_trace(go.Scatter(x=df['open_time'], y=df['close_price'], name="Real", line=dict(color='white', width=1), opacity=0.3))
    t_proy = np.linspace(df['t'].max(), df['t'].max() + 24, 24)
    tiempos_fut = [df['open_time'].max() + timedelta(hours=int(i)) for i in range(1, 25)]
    fig_f5.add_trace(go.Scatter(x=tiempos_fut, y=modelo_fut(t_proy), name="Proyeccion 24h", line=dict(color='red', dash='dash')))
    fig_f5.add_hline(y=precio_objetivo, line_dash="dot", line_color="cyan", annotation_text="Precio Objetivo")
    fig_f5.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_f5, use_container_width=True)

    # Bloque de Reporte Completo
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.subheader("--- REPORTE FINAL DE CALIDAD ---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Error Absoluto (RMSE)", f"${rmse:,.2f}")
        st.write(f"**Error Relativo:** {error_rel:.2f}%")
    with c2:
        calidad = "Alta" if error_rel < 5 else "Aceptable" if error_rel < 15 else "Baja"
        st.write(f"**Analisis de Precision:** {calidad}")
        st.write(f"**Diferencia al Objetivo:** ${precio_objetivo - df['close_price'].iloc[-1]:,.2f}")
    with c3:
        st.write("**Metodo:** Validacion Hold-Out")
        st.write(f"**Muestras Test:** {n_test} horas")
    
    st.divider()
    
    distancia = precio_objetivo - df['close_price'].iloc[-1]
    va_hacia = (distancia > 0 and modelo_fut(t_proy[-1]) > df['close_price'].iloc[-1]) or \
               (distancia < 0 and modelo_fut(t_proy[-1]) < df['close_price'].iloc[-1])

    if error_rel < 15 and va_hacia:
        st.success("### RECOMENDACION FINAL: RECOMENDADO\nEl objetivo es coherente con la inercia del modelo y el error es tolerable.")
    else:
        st.warning("### RECOMENDACION FINAL: NO RECOMENDADO\nEl objetivo diverge de la trayectoria calculada o la fiabilidad es insuficiente.")
    st.markdown('</div>', unsafe_allow_html=True)