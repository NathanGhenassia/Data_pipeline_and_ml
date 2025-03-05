import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn.metrics import accuracy_score, confusion_matrix
from data_pipeline import DataPipeline

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Pipeline y Modelo de IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados con hover en botones
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
        transition: 0.3s;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Barra lateral
st.sidebar.image("https://www.ey.com/content/dam/ey-unified-site/ey-com/en-gr/insights/cybersecurity/images/ey-cybersecurity-eygreece-study-2023.png", use_container_width=True)
st.sidebar.title("🔍 Navegación")
section = st.sidebar.radio("Selecciona una sección", [
    "🏠 Inicio", "🚀 Ejecutar Data Pipeline", "📊 Visualización del Dataset y Gráficos", "🤖 Ejecutar el Modelo"
])

# Sección 1: Inicio
if section == "🏠 Inicio":
    st.title("Aplicación de IA con Streamlit")
    st.write("Bienvenidos a nuestra aplicación web dedicada a la detección de intrusiones cibernéticas. Esta plataforma interactiva se ha desarrollado como parte de un proyecto académico con el objetivo de demostrar la integración y automatización completa de un Data Pipeline junto con un modelo avanzado de inteligencia artificial.")
    if st.expander("👥 Mostrar Participantes del Proyecto"):
        st.markdown("""
        **Participantes del Proyecto:**  
        - 👨‍💻 Nathan Ghenassia (Cientifico de Datos) 
        - 👩‍💻 Maria Fernanda Camacho (Programadora Web)  
        - 👨‍💻 Samuel Salazar (Ingeniero de Datos)  
        """)
    st.image("https://journal.ahima.org/Portals/0/EasyDNNnews/2633/img-Federal-cybersecurity-image-iStock-1420039900.jpg", caption="Esperamos que les guste!", use_container_width=True)

# Crear instancia del pipeline de datos
pipeline = DataPipeline(
    dataset_name="dnkumars/cybersecurity-intrusion-detection-dataset",
    github_user="NathanGhenassia",
    github_repo="Data_pipeline_and_ml",
    github_branch="main"
)

# Función para obtener datasets desde GitHub en formato CSV
def load_dataset_from_github(filename):
    url = f"https://raw.githubusercontent.com/NathanGhenassia/Data_pipeline_and_ml/main/datos/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error(f"No se pudo cargar el dataset {filename} desde GitHub.")
        return None

# Sección 2: Ejecutar Data Pipeline
if section == "🚀 Ejecutar Data Pipeline":
    st.header("🚀 Ejecutar Data Pipeline")
    st.write("Este código implementa un pipeline de datos automatizado que permite la descarga, limpieza, encriptación y carga de conjuntos de datos, integrando herramientas como KaggleApi y GitHub. Durante su ejecución, el sistema genera logs detallados que documentan cada paso del proceso, desde la autenticación en Kaggle y la eliminación de valores NaN, hasta la encriptación de datos sensibles y la subida de archivos procesados a GitHub, lo cual facilita la supervisión y el diagnóstico en caso de errores.")
    st.write("En este sitio web, hemos implementado una función interactiva para demostrar la eficacia y funcionalidad de nuestro Data Pipeline. Al presionar el botón, podrás visualizar los logos que representan cada componente del Data Pipeline en acción.")
    if st.button("Ejecutar Data Pipeline"):
        with st.spinner("Ejecutando Data Pipeline..."):
            logs = pipeline.run_pipeline()
        st.success("✅ Pipeline ejecutado exitosamente!")
        st.text_area("📜 Logs del pipeline", logs, height=300)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*RoFb2sFULMV-gnOy727FoQ.png", use_container_width=True)

# Sección 3: Visualización del Dataset y Gráficos
if section == "📊 Visualización del Dataset y Gráficos":
    st.header("📊 Visualización del Dataset y Gráficos")
    df_original = load_dataset_from_github("dataset_original.csv")
    df_procesado = load_dataset_from_github("dataset_procesado.csv")

    if df_original is not None:
        st.subheader("📌 Dataset Original Encryptado")

        st.dataframe(df_original)
    
    if df_procesado is not None:
        st.subheader("📈 Gráficos del Modelo")
        graphs_path = "outputs"

        # Explicación y visualización del Histograma
        st.subheader("Histograma de distribución de ataques detectados")
        st.write("La visualización muestra la distribución de la variable attack_detected, que es la variable predictoria utilizada para identificar si se detectó un ataque durante una sesión. Existen dos categorías: 0, que indica la ausencia de un ataque, y 1, que representa la detección de un ataque. Según la gráfica de barras, la frecuencia de sesiones sin ataque (categoría 0) es ligeramente mayor, con aproximadamente 4000 casos, en comparación con las sesiones con ataque (categoría 1), que rondan los 3500 casos. Este equilibrio relativo entre ambas clases hace que el dataset sea adecuado para entrenar modelos de detección de intrusiones, permitiendo identificar patrones tanto de sesiones seguras como de aquellas comprometidas.")
        st.image("outputs/predict_variable_distribution.png")
        
        # Explicación y visualización del Gráfico de Pastel
        st.subheader("Distribución de variantes de protocolo")
        st.write("El gráfico de pastel muestra la distribución de variantes de protocolo en el conjunto de datos. Se observa que el protocolo TCP es el más utilizado, representando el 69.5% del total. Le sigue UDP con un 25.2%, mientras que ICMP tiene la menor participación, con solo un 5.3%. Esto indica que la mayoría del tráfico en el conjunto de datos se basa en conexiones orientadas a la transmisión confiable (TCP), mientras que UDP, que es más rápido pero menos confiable, tiene una presencia menor. El protocolo ICMP, utilizado principalmente para diagnósticos de red y mensajes de error, representa la menor proporción del tráfico.")
        st.image("outputs/protocol_pie_chart.png")
        
        # Explicación y visualización del Mapa de Calor de Correlación
        st.subheader("Mapa de calor de correlación entre variables")
        st.write("La matriz de correlación muestra la relación entre diferentes variables del conjunto de datos. Se observa que la variable attack_detected tiene una correlación positiva moderada con failed_logins (0.37) y login_attempts (0.28), lo que indica que un mayor número de intentos de inicio de sesión o fallos en el acceso pueden estar asociados con la detección de un ataque. También existe una correlación positiva entre attack_detected y ip_reputation_score (0.21), lo que sugiere que direcciones IP con mala reputación pueden estar relacionadas con ataques detectados. En contraste, otras variables como protocol_type, session_duration y browser_type tienen correlaciones más débiles con attack_detected, lo que indica que su impacto en la detección de ataques es menor.")
        st.image("outputs/variable_heatmap.png")

# Sección 4: Ejecutar el Modelo
if section == "🤖 Ejecutar el Modelo":
    st.header("🤖 Ejecutar el Modelo con el Dataset Procesado")
    st.write("Aquí podrás ver los resultados del modelo en tiempo real. En la sección de Predicciones, encontrarás las categorías o valores estimados según los datos ingresados. Justo debajo, se muestra la precisión del modelo (accuracy), que indica qué tan bien está funcionando la IA en términos de predicciones correctas. Además, podrás analizar la matriz de confusión, una herramienta visual que permite identificar los aciertos y errores del modelo en cada clase, ayudando a comprender mejor su desempeño.")

    def load_model():
        if os.path.exists("best_model.pkl"):
            return joblib.load("best_model.pkl")
        return None

    def load_scaler():
        if os.path.exists("scaler.pkl"):
            return joblib.load("scaler.pkl")
        return None

    if st.button("Ejecutar Modelo"):
        model = load_model()
        scaler = load_scaler()
        df_procesado = load_dataset_from_github("dataset_procesado.csv")
        
        if model and scaler and df_procesado is not None:
            st.subheader("📌 Predicciones del Modelo")
            X = df_procesado.drop(columns=['attack_detected'])
            y_true = df_procesado['attack_detected']
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            df_procesado["Predicción"] = predictions
            st.dataframe(df_procesado)
            
            accuracy = accuracy_score(y_true, predictions)
            st.subheader("🎯 Precisión del Modelo")
            st.write(f"Precisión: {accuracy:.2%}")
            
            st.subheader("📊 Matriz de Confusión")
            cm = confusion_matrix(y_true, predictions)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["No Ataque", "Ataque"], 
                        yticklabels=["No Ataque", "Ataque"])
            plt.xlabel("Predicción")
            plt.ylabel("Real")
            st.pyplot(plt)
        else:
            st.error("⚠️ No se pudo ejecutar el modelo. Verifica que 'best_model.pkl' y 'scaler.pkl' estén disponibles.")
