import streamlit as st
import pandas as pd
import altair as alt
import joblib  
import numpy as np
from sklearn.model_selection import train_test_split 
import json # <-- ¡IMPORTANTE! Importamos la librería JSON

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Ingresos y Empleo (Gran Mendoza)",
    page_icon="📊",
    layout="wide"
)

# --- Carga de Datos, Modelo y Gráficos ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Tabla_Final.csv')
    except FileNotFoundError:
        st.error("Error: No se encontró 'Tabla_Final.csv'. Asegúrate de que esté en la misma carpeta que app.py.")
        return (None,) * 10 # Retorna Nones para evitar más errores

    df_clean = df.dropna(subset=['IngresoPromedio']).copy()
    
    # --- ¡CORRECCIÓN APLICADA AQUÍ! ---
    # En lugar de alt.load_chart, leemos el JSON como un diccionario.
    try:
        with open('piramide_ingresos.json') as f:
            chart1 = json.load(f)
        with open('panel_brushing.json') as f:
            chart2 = json.load(f)
        with open('timeline_ingresos.json') as f:
            chart3 = json.load(f)
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró un archivo JSON esencial: {e}. Asegúrate de que los 3 archivos .json estén en la carpeta.")
        chart1, chart2, chart3 = None, None, None
    # --- FIN DE LA CORRECCIÓN ---
    
    niveles_educativos = df_clean['NivelEducativo'].unique()
    rangos_etarios = df_clean['RangoEtario'].unique()
    sexos = df_clean['Sexo'].unique()
    
    CATEGORICAL_FEATURES = ['NivelEducativo', 'RangoEtario', 'Sexo']
    NUMERIC_FEATURES = ['HorasTrabajoPromedio', 'TasaActividadPonderada', 'TasaEmpleoPonderada', 'Poblacion']
    FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    TARGET = 'IngresoPromedio'

    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return df_clean, chart1, chart2, chart3, niveles_educativos, rangos_etarios, sexos, X_test, y_test, FEATURES

@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_ridge.pkl')
        return model
    except FileNotFoundError:
        return None

# Cargar todo
(df_clean, chart1, chart2, chart3, 
 niveles_educativos, rangos_etarios, sexos, 
 X_test, y_test, FEATURES) = load_data()

# --- Barra Lateral (Sidebar) para el Modelo ---
st.sidebar.title("🤖 Probar el Modelo (Ridge Regression)")
st.sidebar.markdown("Ingresa datos de un segmento poblacional para predecir su ingreso promedio.")

if model is None:
    st.sidebar.error("Error Crítico: 'modelo_ridge.pkl' no encontrado.")
elif df_clean is not None:
    inputs = {}
    st.sidebar.header("Variables Categóricas")
    inputs['NivelEducativo'] = st.sidebar.selectbox("Nivel Educativo", options=niveles_educativos)
    inputs['RangoEtario'] = st.sidebar.selectbox("Rango Etario", options=rangos_etarios)
    inputs['Sexo'] = st.sidebar.selectbox("Sexo", options=sexos)

    st.sidebar.header("Variables Numéricas")
    inputs['HorasTrabajoPromedio'] = st.sidebar.number_input("Horas de Trabajo Promedio", min_value=0.0, max_value=80.0, value=40.0, step=0.1)
    inputs['TasaActividadPonderada'] = st.sidebar.slider("Tasa de Actividad Ponderada", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    inputs['TasaEmpleoPonderada'] = st.sidebar.slider("Tasa de Empleo Ponderada", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    inputs['Poblacion'] = st.sidebar.number_input("Población del Segmento", min_value=0, max_value=100000, value=1000, step=100)

    if st.sidebar.button("Predecir Ingreso Promedio"):
        input_df = pd.DataFrame([inputs], columns=FEATURES)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Datos de Entrada:")
        st.sidebar.dataframe(input_df)
        
        try:
            prediction = model.predict(input_df)
            st.sidebar.subheader("Resultado de la Predicción:")
            st.sidebar.success(f"Ingreso Promedio Estimado: **${prediction[0]:,.2f}**")
        except Exception as e:
            st.sidebar.error(f"Error al predecir: {e}")

# --- Cuerpo Principal de la Aplicación ---
st.title("📊 4ta Entrega: Visualización e Integración de Modelos")

if df_clean is not None:
    st.markdown(f"""
    Bienvenido al dashboard del proyecto. Esta aplicación integra los hallazgos de las entregas anteriores.
    - **Modelo Predictivo:** `Ridge Regression` (R² Test: 0.680, RMSE Test: $21,102.53)
    - **Datos:** `Tabla_Final.csv` ({len(df_clean)} segmentos analizados)
    """)

    # --- Sección 1: Visualizaciones Interactivas (Altair) ---
    st.header("1. Visualizaciones Interactivas (Altair)")

    if chart1:
        st.subheader("Pirámide Educativa de Ingresos y Brecha de Género")
        # st.altair_chart ahora recibe un diccionario, ¡y funciona!
        st.altair_chart(chart1, use_container_width=True) 
        st.markdown("""
        Este gráfico compara el ingreso promedio (USD) entre varones y mujeres para cada nivel educativo. Las barras, presentadas de forma opuesta, ilustran una clara **brecha de género**: en casi todos los niveles, el ingreso promedio de los varones (en azul) es superior al de las mujeres (en naranja).

        Además, se reafirma la tendencia de que **a mayor nivel educativo, mayor es el ingreso** promedio para ambos sexos, siendo la diferencia de ingresos particularmente pronunciada en los niveles de posgrado.
        """)

    if chart3:
        st.subheader("Evolución del Ingreso Promedio por Edad y Nivel Educativo")
        st.altair_chart(chart3, use_container_width=True)
        st.markdown("""
        Esta visualización muestra la **trayectoria de ingresos** a lo largo de los diferentes rangos etarios, segmentada por nivel educativo.

        Se observa claramente cómo la "curva de la experiencia" impacta positivamente en los ingresos, especialmente para quienes poseen **estudios universitarios o de posgrado**, alcanzando picos de ingreso entre los 45 y 59 años. Por el contrario, los niveles educativos más bajos (primario y secundario) muestran un aplanamiento de ingresos a una edad mucho más temprana.
        """)

    if chart2:
        st.subheader("Panel Interactivo: Ingreso vs. Horas de Trabajo y Nivel Educativo")
        st.altair_chart(chart2, use_container_width=True)
        st.markdown("""
        Este panel doble permite una **exploración interactiva** de los datos. El gráfico de dispersión (arriba) muestra la relación entre las horas trabajadas y el ingreso promedio, donde se observa una correlación positiva general, pero con una alta dispersión.

        **Instrucción:** Use el mouse para **seleccionar un área rectangular** del gráfico de dispersión. El gráfico de barras (abajo) se actualizará automáticamente para mostrar la composición educativa y la brecha de género solo del subconjunto seleccionado, permitiendo un análisis focalizado.
        """)

    # --- Sección 2: Evaluación del Modelo (Adicional) ---
    st.header("2. Evaluación del Modelo de Regresión")
    st.subheader("Valores Reales vs. Valores Predichos (Datos de Test)")
    st.markdown("Esta visualización (creada con el conjunto de *test*) compara el ingreso real (Eje Y) con el ingreso que nuestro modelo predijo (Eje X).")

    if model is not None:
        y_test_pred = model.predict(X_test)
        plot_data = pd.DataFrame({
            'Ingreso Real (y_test)': y_test,
            'Ingreso Predicho (y_pred)': y_test_pred
        })
        
        chart_pred = alt.Chart(plot_data).mark_circle(size=60).encode(
            x=alt.X('Ingreso Predicho (y_pred)', title='Ingreso Predicho (ARS)', scale=alt.Scale(zero=False)),
            y=alt.Y('Ingreso Real (y_test)', title='Ingreso Real (ARS)', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('Ingreso Real (y_test)', format=',.2f'), alt.Tooltip('Ingreso Predicho (y_pred)', format=',.2f')]
        ).interactive()
        
        line = alt.Chart(pd.DataFrame({'x': [0, 120000], 'y': [0, 120000]})).mark_line(color='red', strokeDash=[5,5]).encode(x='x', y='y')
        
        st.altair_chart(chart_pred + line, use_container_width=True)
        st.markdown(" idealmente, los puntos deberían caer sobre la **línea roja** (predicción perfecta).")
    
    # --- Sección 3: Exploración de Datos ---
    st.header("3. Exploración de los Datos Completos")
    with st.expander("Ver y Filtrar la 'Tabla_Final' completa", expanded=False):
        st.dataframe(df_clean)
        st.markdown(f"Mostrando **{len(df_clean)}** registros limpios.")

else:
    st.error("No se pudo cargar el DataFrame. El resto de la aplicación no puede continuar.")