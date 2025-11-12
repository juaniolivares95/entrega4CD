import streamlit as st
import pandas as pd
import altair as alt 
import joblib 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

# --- Definición de la Estructura de Activos ---
@dataclass
class AppAssets:
    """Contiene todos los datos, gráficos y listas pre-procesadas para la app."""
    df: pd.DataFrame
    chart1: alt.Chart
    chart2: alt.Chart
    chart3: alt.Chart
    niveles_educativos: list
    rangos_etarios: list
    sexos: list
    X_test: pd.DataFrame
    y_test: pd.Series
    FEATURES: list

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Ingresos y Empleo (Gran Mendoza)",
    page_icon="📊",
    layout="wide"
)

# --- Carga de Datos, Modelo y Gráficos ---

@st.cache_data
def load_app_assets() -> AppAssets | None:
    """
    Carga el CSV, limpia los datos, genera los gráficos de Altair y prepara
    los datos para el modelo. Retorna un objeto AppAssets o None si falla.
    """
    try:
        df = pd.read_csv('Tabla_Final.csv')
    except FileNotFoundError:
        st.error("Error: No se encontró 'Tabla_Final.csv'.")
        return None

    df_clean = df.dropna(subset=['IngresoPromedio']).copy()
    
    if df_clean.empty:
        st.error("Error: Los datos están vacíos después de la limpieza.")
        return None
    
    try:
        y_sort_order = ["Primario","Secundario","Terciario no universitario", "Universitario de grado","Posgrado (especialización, maestría o doctorado)"]
        y_axis_definition = alt.Y("NivelEducativo:N", sort=y_sort_order, title="Nivel Educativo")

        # --- Gráfico 1: Pirámide (Mejorado) ---
        pir = (
            df_clean.groupby(["NivelEducativo","Sexo"], as_index=False)
              .agg({"IngresoPromedioUSD":"mean"})
        )
        pir["IngresoPromedioUSD_signed"] = pir.apply(
            lambda x: -x["IngresoPromedioUSD"] if x["Sexo"] == "Mujer" else x["IngresoPromedioUSD"], axis=1
        )
        base = (
            alt.Chart(pir)
            .mark_bar()
            .encode(
                x=alt.X("IngresoPromedioUSD_signed:Q", title="Ingreso Promedio (USD)", axis=alt.Axis(format="$.0f")),
                y=y_axis_definition, 
                color=alt.Color("Sexo:N", title="Sexo", scale=alt.Scale(domain=["Varón","Mujer"], range=["#1f77b4","#ff7f0e"])),
                tooltip=["Sexo","NivelEducativo", alt.Tooltip("IngresoPromedioUSD:Q", format=",.1f", title="Ingreso Promedio (USD)")]
            )
            .properties(width=600, height=350, title="Pirámide educativa de ingresos – Gran Mendoza")
        )
        
        text_varon = alt.Chart(pir.loc[pir['Sexo'] == 'Varón']).mark_text(
            align="left", dx=5, color="white"
        ).encode(
            x=alt.X("IngresoPromedioUSD_signed:Q"),
            y=y_axis_definition, 
            text=alt.Text("IngresoPromedioUSD:Q", format=",.0f")
        )
        text_mujer = alt.Chart(pir.loc[pir['Sexo'] == 'Mujer']).mark_text(
            align="right", dx=-5, color="white"
        ).encode(
            x=alt.X("IngresoPromedioUSD_signed:Q"),
            y=y_axis_definition, 
            text=alt.Text("IngresoPromedioUSD:Q", format=",.0f")
        )
        chart1 = (base + text_varon + text_mujer).resolve_scale(x="shared")

        # --- Gráfico 2: Panel Brushing (Sin cambios) ---
        brush = alt.selection_interval(encodings=["x","y"])
        scatter = (
            alt.Chart(df_clean)
            .mark_circle(size=90, opacity=0.7)
            .encode(
                x=alt.X("HorasTrabajoPromedio:Q", title="Horas semanales"),
                y=alt.Y("IngresoPromedioUSD:Q", title="Ingreso Promedio (USD)"),
                color=alt.condition(brush, "Sexo:N", alt.value("lightgray")),
                tooltip=["Sexo","NivelEducativo","RangoEtario", alt.Tooltip("IngresoPromedioUSD:Q",format=",.1f"), alt.Tooltip("HorasTrabajoPromedio:Q",format=",.1f")]
            )
            .add_params(brush)
            .properties(width=600, height=350, title="Selección libre: Ingreso vs Horas trabajadas")
        )
        bars = (
            alt.Chart(df_clean)
            .mark_bar()
            .encode(
                x=alt.X("mean(IngresoPromedioUSD):Q", title="Ingreso Promedio (USD)"),
                y=alt.Y("NivelEducativo:N", sort=y_sort_order),
                color=alt.Color("Sexo:N", title="Sexo")
            )
            .transform_filter(brush)
            .properties(width=600, height=250, title="Detalle educativo del subconjunto seleccionado")
        )
        chart2 = scatter & bars

        # --- Gráfico 3: Timeline (¡AJUSTADO A CLIC EN LEYENDA!) ---
        timeline_data = (
            df_clean.groupby(["RangoEtario","NivelEducativo"], as_index=False)
              .agg({"IngresoPromedioUSD":"mean"})
        )
        
        # 1. Crear una selección múltiple vinculada a la leyenda
        selection = alt.selection_multi(fields=['NivelEducativo'], bind='legend')

        chart3 = (
            alt.Chart(timeline_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("RangoEtario:N", title="Rango Etario", sort=["15-19","20-24","25-29","30-34","35-39","40-44", "45-49","50-54","55-59","60-64","65+"]),
                y=alt.Y("IngresoPromedioUSD:Q", title="Ingreso Promedio (USD)"),
                color=alt.Color("NivelEducativo:N", title="Nivel Educativo"),
                # 2. La opacidad depende de la selección (1.0 si está seleccionado, 0.2 si no)
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.2)),
                tooltip=["RangoEtario","NivelEducativo",alt.Tooltip("IngresoPromedioUSD:Q",format=",.1f")]
            )
            .properties(width=700, height=350, title="Timeline socioeducativo de ingresos (Gran Mendoza)")
            # 3. Añadir la selección al gráfico
            .add_params(selection)
        )


    except Exception as e:
        st.error(f"Error al crear los gráficos de Altair: {e}")
        chart1, chart2, chart3 = None, None, None
    
    try:
        niveles_educativos = sorted(df_clean['NivelEducativo'].unique().tolist())
        rangos_etarios = sorted(df_clean['RangoEtario'].unique().tolist())
        sexos = sorted(df_clean['Sexo'].unique().tolist())
        
        CATEGORICAL_FEATURES = ['NivelEducativo', 'RangoEtario', 'Sexo']
        NUMERIC_FEATURES = ['HorasTrabajoPromedio', 'TasaActividadPonderada', 'TasaEmpleoPonderada', 'Poblacion']
        FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        TARGET = 'IngresoPromedio'

        X = df_clean[FEATURES]
        y = df_clean[TARGET]
    
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    except Exception as e:
        st.error(f"Error al procesar las columnas del DataFrame: {e}")
        return None

    return AppAssets(
        df=df_clean,
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        niveles_educativos=niveles_educativos,
        rangos_etarios=rangos_etarios,
        sexos=sexos,
        X_test=X_test,
        y_test=y_test,
        FEATURES=FEATURES
    )

@st.cache_resource
def load_model():
    """Carga el modelo .pkl cacheado."""
    try:
        model = joblib.load('modelo_ridge.pkl')
        return model
    except FileNotFoundError:
        st.error("Error Crítico: No se encontró 'modelo_ridge.pkl'.")
        return None
    except Exception as e:
        st.error(f"Error Crítico al cargar 'modelo_ridge.pkl': {e}")
        return None

# --- Cargar todo ---
assets = load_app_assets()
model = load_model()

# --- Cuerpo Principal ---
st.title("📊 4ta Entrega: Visualización e Integración de Modelos")

if assets is not None:
    st.markdown(f"""
    Bienvenido al dashboard del proyecto. Esta aplicación integra los hallazgos de las entregas anteriores.
    - **Modelo Predictivo:** `Ridge Regression` (R² Test: 0.680, RMSE Test: $21,102.53)
    - **Datos:** `Tabla_Final.csv` ({len(assets.df)} segmentos analizados)
    """)

    st.header("1. Visualizaciones Interactivas (Altair)")

    if assets.chart1:
        st.subheader("Pirámide Educativa de Ingresos y Brecha de Género")
        st.altair_chart(assets.chart1, use_container_width=True) 
        st.markdown("""
        Este gráfico compara el ingreso promedio (USD) entre varones y mujeres para cada nivel educativo...
        """) 

    if assets.chart3:
        st.subheader("Evolución del Ingreso Promedio por Edad y Nivel Educativo")
        st.altair_chart(assets.chart3, use_container_width=True)
        st.markdown("""
        Esta visualización muestra la **trayectoria de ingresos** a lo largo de los diferentes rangos etarios.
        **¡Interactivo!** Haz clic en los elementos de la leyenda (ej. "Secundario") para filtrar las líneas.
        """) # ¡Texto actualizado!

    if assets.chart2:
        st.subheader("Panel Interactivo: Ingreso vs. Horas de Trabajo y Nivel Educativo")
        st.altair_chart(assets.chart2, use_container_width=True)
        st.markdown("""
        Este panel doble permite una **exploración interactiva**... **Instrucción:** Use el mouse para **seleccionar un área rectangular**...
        """)

    st.header("2. Evaluación del Modelo de Regresión")
    st.subheader("Valores Reales vs. Valores Predichos (Datos de Test)")
    st.markdown("Esta visualización (creada con el conjunto de *test*) compara el ingreso real (Eje Y) con el ingreso que nuestro modelo predijo (Eje X).")

    if model is not None:
        y_test_pred = model.predict(assets.X_test)
        plot_data = pd.DataFrame({
            'Ingreso Real (y_test)': assets.y_test,
            'Ingreso Predicho (y_pred)': y_test_pred
        })
        
        chart_pred = alt.Chart(plot_data).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X('Ingreso Predicho (y_pred)', title='Ingreso Predicho (ARS)', scale=alt.Scale(zero=False)),
            y=alt.Y('Ingreso Real (y_test)', title='Ingreso Real (ARS)', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('Ingreso Real (y_test)', format=',.2f'), alt.Tooltip('Ingreso Predicho (y_pred)', format=',.2f')]
        ).interactive()
        
        min_val = min(plot_data['Ingreso Real (y_test)'].min(), plot_data['Ingreso Predicho (y_pred)'].min())
        max_val = max(plot_data['Ingreso Real (y_test)'].max(), plot_data['Ingreso Predicho (y_pred)'].max())
        line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})
        line = alt.Chart(line_df).mark_line(color='red', strokeDash=[5,5]).encode(x='x', y='y')
        
        st.altair_chart(chart_pred + line, use_container_width=True)
        st.markdown(" idealmente, los puntos deberían caer sobre la **línea roja** (predicción perfecta).")
    
    st.header("3. Exploración de los Datos Completos")
    with st.expander("Ver y Filtrar la 'Tabla_Final' completa", expanded=False):
        st.dataframe(assets.df)
        st.markdown(f"Mostrando **{len(assets.df)}** registros limpios.")

    # --- Sección de Predicción ---
    st.divider() 
    st.header("4. Probar el Modelo (Ridge Regression)")

    if model is None:
        st.error("El modelo predictivo no pudo cargarse. La función de predicción está deshabilitada.")
    else:
        st.markdown("Ingresa datos de un segmento poblacional para predecir su ingreso promedio.")
        
        col1, col2 = st.columns(2)
        inputs = {}

        with col1:
            st.subheader("Variables Categóricas")
            inputs['NivelEducativo'] = st.selectbox("Nivel Educativo", options=assets.niveles_educativos, key="main_nivel")
            inputs['RangoEtario'] = st.selectbox("Rango Etario", options=assets.rangos_etarios, key="main_rango")
            inputs['Sexo'] = st.selectbox("Sexo", options=assets.sexos, key="main_sexo")

        with col2:
            st.subheader("Variables Numéricas")
            inputs['HorasTrabajoPromedio'] = st.number_input("Horas de Trabajo Promedio", min_value=0.0, max_value=80.0, value=40.0, step=0.1, key="main_horas")
            inputs['TasaActividadPonderada'] = st.slider("Tasa de ActividadPonderada", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="main_actividad")
            inputs['TasaEmpleoPonderada'] = st.slider("Tasa de Empleo Ponderada", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="main_empleo")
            inputs['Poblacion'] = st.number_input("Población del Segmento", min_value=0, max_value=100000, value=1000, step=100, key="main_poblacion")

        if st.button("Predecir Ingreso Promedio", type="primary"):
            input_df = pd.DataFrame([inputs], columns=assets.FEATURES)
            st.markdown("---")
            st.subheader("Datos de Entrada:")
            st.dataframe(input_df)
            
            try:
                prediction = model.predict(input_df)
                st.subheader("Resultado de la Predicción:")
                st.success(f"Ingreso Promedio Estimado: **${prediction[0]:,.2f}**")
            except Exception as e:
                st.error(f"Error al predecir: {e}")

else:
    st.error("No se pudieron cargar los datos ('Tabla_Final.csv'). La aplicación no puede mostrar contenido.")