import streamlit as st
import pandas as pd
import altair as alt 
import joblib 
import numpy as np
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder # <-- ¡NUEVO IMPORTE!
from dataclasses import dataclass
from scipy.stats import percentileofscore

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
    # ... (Esta función no cambia, la omito por brevedad) ...
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

        # --- Gráfico 3: Timeline (Clic en Leyenda) ---
        timeline_data = (
            df_clean.groupby(["RangoEtario","NivelEducativo"], as_index=False)
              .agg({"IngresoPromedioUSD":"mean"})
        )
        
        selection = alt.selection_multi(fields=['NivelEducativo'], bind='legend')

        chart3 = (
            alt.Chart(timeline_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("RangoEtario:N", title="Rango Etario", sort=["15-19","20-24","25-29","30-34","35-39","40-44", "45-49","50-54","55-59","60-64","65+"]),
                y=alt.Y("IngresoPromedioUSD:Q", title="Ingreso Promedio (USD)"),
                color=alt.Color("NivelEducativo:N", title="Nivel Educativo"),
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.2)),
                tooltip=["RangoEtario","NivelEducativo",alt.Tooltip("IngresoPromedioUSD:Q",format=",.1f")]
            )
            .properties(width=700, height=350, title="Timeline socioeducativo de ingresos (Gran Mendoza)")
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

# --- Función LIME ELIMINADA ---
# La crearemos al vuelo

# --- Cargar todo ---
assets = load_app_assets()
model = load_model()

# --- Inicializar Session State ---
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None


# --- Cuerpo Principal ---
st.title("📊 4ta Entrega: Visualización e Integración de Modelos")

if assets is not None:
    st.markdown(f"""
    Bienvenido al dashboard del proyecto. Esta aplicación integra los hallazgos de las entregas anteriores.
    - **Modelo Predictivo:** `Ridge Regression` (R² Test: 0.680, RMSE Test: $21,102.53)
    - **Datos:** `Tabla_Final.csv` ({len(assets.df)} segmentos analizados)
    """)

    # --- Creación de Pestañas ---
    tab1_viz, tab2_model = st.tabs([
        "📈 Visualizaciones Exploratorias", 
        "🤖 Probar el Modelo"
    ])

    # --- Pestaña 1: Contenido de Visualizaciones ---
    with tab1_viz:
        # ... (Esta sección no cambia) ...
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
            """) 

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

    # --- Pestaña 2: Contenido del Predictor (CON LIME CORREGIDO) ---
    with tab2_model:
        st.header("Probar el Modelo (Ridge Regression)")

        if model is None:
            st.error("El modelo predictivo no pudo cargarse. La función de predicción está deshabilitada.")
        else:
            st.markdown("Ingresa datos de un segmento poblacional para predecir su ingreso promedio.")
            
            st.info("Prueba cargando un perfil aleatorio del conjunto de datos de testeo para ver cómo funciona.")
            if st.button("Cargar segmento aleatorio de Test"):
                sample = assets.X_test.sample(1).iloc[0].to_dict()
                st.session_state.sample_data = sample
            
            st.divider()

            col1, col2 = st.columns(2)
            inputs = {}

            with col1:
                st.subheader("Variables Categóricas")
                # ... (Inputs no cambian) ...
                default_nivel_idx = 0
                if st.session_state.sample_data:
                    default_nivel_idx = assets.niveles_educativos.index(st.session_state.sample_data['NivelEducativo'])
                inputs['NivelEducativo'] = st.selectbox(
                    "Nivel Educativo", 
                    options=assets.niveles_educativos, 
                    index=default_nivel_idx,
                    help="Máximo nivel educativo alcanzado por el segmento."
                )

                default_rango_idx = 0
                if st.session_state.sample_data:
                    default_rango_idx = assets.rangos_etarios.index(st.session_state.sample_data['RangoEtario'])
                inputs['RangoEtario'] = st.selectbox(
                    "Rango Etario", 
                    options=assets.rangos_etarios, 
                    index=default_rango_idx,
                    help="Grupo de edad al que pertenece el segmento."
                )

                default_sexo_idx = 0
                if st.session_state.sample_data:
                    default_sexo_idx = assets.sexos.index(st.session_state.sample_data['Sexo'])
                inputs['Sexo'] = st.selectbox(
                    "Sexo", 
                    options=assets.sexos, 
                    index=default_sexo_idx,
                    help="Sexo del segmento."
                )

            with col2:
                st.subheader("Variables Numéricas")
                # ... (Inputs no cambian) ...
                default_horas = 40.0
                if st.session_state.sample_data:
                    default_horas = st.session_state.sample_data['HorasTrabajoPromedio']
                inputs['HorasTrabajoPromedio'] = st.number_input(
                    "Horas de Trabajo Promedio", 
                    min_value=0.0, max_value=80.0, value=default_horas, step=0.1, 
                    help="Promedio de horas semanales trabajadas por el segmento."
                )

                default_actividad = 0.5
                if st.session_state.sample_data:
                    default_actividad = st.session_state.sample_data['TasaActividadPonderada']
                inputs['TasaActividadPonderada'] = st.slider(
                    "Tasa de Actividad Ponderada", 
                    min_value=0.0, max_value=1.0, value=default_actividad, step=0.01, 
                    help="Porcentaje de la población del segmento que está activa (trabaja o busca trabajo)."
                )

                default_empleo = 0.5
                if st.session_state.sample_data:
                    default_empleo = st.session_state.sample_data['TasaEmpleoPonderada']
                inputs['TasaEmpleoPonderada'] = st.slider(
                    "Tasa de Empleo Ponderada", 
                    min_value=0.0, max_value=1.0, value=default_empleo, step=0.01, 
                    help="Porcentaje de la población del segmento que está empleada."
                )

                default_poblacion = 1000
                if st.session_state.sample_data:
                    default_poblacion = int(st.session_state.sample_data['Poblacion'])
                inputs['Poblacion'] = st.number_input(
                    "Población del Segmento", 
                    min_value=0, max_value=100000, value=default_poblacion, step=100, 
                    help="Número de personas en este segmento."
                )

            if st.button("Predecir Ingreso Promedio", type="primary"):
                input_df = pd.DataFrame([inputs], columns=assets.FEATURES)
                st.markdown("---")
                st.subheader("Datos de Entrada:")
                st.dataframe(input_df)
                
                try:
                    # --- 1. PREDICCIÓN ---
                    prediction = model.predict(input_df)
                    prediction_value = prediction[0] 
                    
                    st.subheader("Resultado de la Predicción:")
                    st.success(f"Ingreso Promedio Estimado: **${prediction_value:,.2f}**")
                    
                    # --- 2. PERCENTIL ---
                    st.subheader("Análisis Comparativo")
                    mask = (
                        (assets.df['NivelEducativo'] == inputs['NivelEducativo']) &
                        (assets.df['Sexo'] == inputs['Sexo'])
                    )
                    subset_df = assets.df[mask]

                    if not subset_df.empty:
                        all_ingresos = subset_df['IngresoPromedio'].values
                        p = percentileofscore(all_ingresos, prediction_value)
                        st.info(f"""
                        Este ingreso te ubicaría en el **percentil {p:.0f}** dentro de todos los segmentos con el mismo **Nivel Educativo y Sexo**.
                        """)
                    else:
                        st.warning("No se encontraron datos históricos para este segmento exacto para calcular el percentil.")
                    
                    # --- 3. ¡NUEVA LÓGICA LIME (El "Traductor")! ---
                    st.subheader("Interpretación del Modelo (LIME)")
                    st.markdown("""
                    Este gráfico explica **por qué** el modelo dio ese resultado.
                    - **Barras Verdes:** Variables que **aumentaron** la predicción.
                    - **Barras Rojas:** Variables que **disminuyeron** la predicción.
                    """)
                    
                    # Paso A: Crear el "Mundo Numérico" para LIME
                    
                    # 1. Definimos las columnas categóricas
                    categorical_features_names = ['NivelEducativo', 'RangoEtario', 'Sexo']
                    categorical_features_indices = [assets.FEATURES.index(col) for col in categorical_features_names]
                    
                    # 2. Creamos un "codificador" que convertirá strings a números
                    # (ej. Primario=0, Secundario=1...)
                    # Usamos las listas de 'assets' para asegurar el orden
                    encoder = OrdinalEncoder(categories=[
                        assets.niveles_educativos, 
                        assets.rangos_etarios, 
                        assets.sexos
                    ])
                    
                    # 3. Codificamos los datos de entrenamiento de LIME
                    # Primero, 'fiteamos' el codificador en las columnas categóricas de X_test
                    encoder.fit(assets.X_test[categorical_features_names])
                    
                    # Luego, transformamos X_test a un formato numérico
                    X_test_encoded = assets.X_test.copy()
                    X_test_encoded[categorical_features_names] = encoder.transform(assets.X_test[categorical_features_names])
                    
                    # 4. Creamos el Explicador LIME. Ahora SÍ funcionará.
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_test_encoded.values, # ¡Datos 100% numéricos!
                        feature_names=assets.FEATURES,
                        class_names=['IngresoPromedio'],
                        categorical_features=categorical_features_indices,
                        discretize_continuous=False,
                        mode='regression'
                    )

                    # Paso B: Crear la función "Traductora" (Wrapper)
                    
                    def predict_fn_wrapper(rows_encoded):
                        # Esta función recibe filas NUMÉRICAS de LIME
                        # 1. Convertimos las filas numéricas a un DataFrame
                        df_encoded = pd.DataFrame(rows_encoded, columns=assets.FEATURES)
                        
                        # 2. Creamos una copia para decodificar
                        df_decoded = df_encoded.copy()
                        
                        # 3. Usamos el 'encoder' para "traducir" de vuelta a strings
                        df_decoded[categorical_features_names] = encoder.inverse_transform(
                            df_encoded[categorical_features_names].astype(int)
                        )
                        
                        # 4. Le pasamos el DataFrame con strings al modelo
                        # El modelo ahora está feliz
                        return model.predict(df_decoded)

                    # Paso C: Explicar la predicción actual
                    
                    # 1. Codificamos también la fila de input del usuario
                    input_df_encoded = input_df.copy()
                    input_df_encoded[categorical_features_names] = encoder.transform(input_df[categorical_features_names])
                    input_array_encoded = input_df_encoded.iloc[0].values

                    # 2. Llamamos a LIME
                    exp = explainer.explain_instance(
                        data_row=input_array_encoded, # Le pasamos la fila numérica
                        predict_fn=predict_fn_wrapper, # Le pasamos nuestro "traductor"
                        num_features=len(assets.FEATURES)
                    )

                    # 3. Mostramos el gráfico
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error al predecir o interpretar: {e}")

else:
    st.error("No se pudieron cargar los datos ('Tabla_Final.csv'). La aplicación no puede mostrar contenido.")