# Proyecto ACA 2 - Aprendizaje Automatico (Deep Learning & Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Configuración global para gráficos ejecutivos en entornos oscuros
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "figure.facecolor": "#0d0d26",
    "axes.facecolor": "#0d0d26",
    "grid.color": "#1e1b4b",
    "text.color": "#ffffff",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8"
})

# ==========================================
# 1. CONFIGURACION Y ESTILO MEJORADO 
# ==========================================
st.set_page_config(
    page_title="🧠 Predicción Neuronal de Deserción",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar estados de sesiOn
if 'resultado_mlp' not in st.session_state:
    st.session_state.resultado_mlp = ""
if 'probabilidades_mlp' not in st.session_state:
    st.session_state.probabilidades_mlp = None
    
# Estilos "Cyber Executive" con IluminaciOn Integrada de Alto Contraste
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #050515 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* BARRA LATERAL REDISEÑADA */
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        background-color: #02020a !important;
        border-right: 1px solid #161630 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #6366f1 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.8rem !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label div[role="radiogroup"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: #0a0a1f !important;
        border: 1px solid #161630 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.25s ease-in-out !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"]::before {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radio"] > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        border-color: #6366f1 !important;
        background-color: #121236 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] label {
        background: linear-gradient(90deg, #121236 0%, #1e1b4b 100%) !important;
        border-color: #d946ef !important;
        box-shadow: 0 0 10px rgba(217, 70, 239, 0.15) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] label span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* CÁPSULA PREMIUM PARA EL TÍTULO PRINCIPAL */
    .title-capsule {
        background: linear-gradient(135deg, #0d0d26 0%, #07071c 100%) !important;
        border: 1px solid #1e1b4b !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
        box-shadow: 0 8px 32px 0 rgba(56, 189, 248, 0.15) !important;
        margin-bottom: 2rem !important;
        position: relative;
        overflow: hidden;
    }

    .title-capsule::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #d946ef, #38bdf8);
    }

    .title-capsule h1 {
        margin: 0 0 0.5rem 0 !important;
        font-size: 2.4rem !important;
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.6) !important;
        font-weight: 700 !important;
    }

    .title-capsule p {
        margin: 0 !important;
        font-size: 1.1rem !important;
        color: #38bdf8 !important;
        font-weight: 500 !important;
    }

    /* ESTILOS DEL CONTENIDO DEL DASHBOARD */
    h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
    }
    
    .main p, .main span {
        color: #cbd5e1 !important;
    }
    
    .card, .metric-card {
        background-color: #0d0d26 !important;
        padding: 1.75rem;
        border-radius: 8px !important;
        box-shadow: 0 4px 30px rgba(217, 70, 239, 0.08) !important;
        border: 1px solid #1e1b4b !important;
        border-top: 3px solid #d946ef !important; 
        margin-bottom: 1.5rem;
    }
    
    .card p, .card ul, .card li, .metric-card * {
        color: #f1f5f9 !important;
        font-size: 1rem !important;
    }
    
    .card strong {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1 0%, #d946ef 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 0 15px rgba(217, 70, 239, 0.5) !important;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #c084fc 100%) !important;
        box-shadow: 0 0 25px rgba(217, 70, 239, 0.8) !important;
    }
    
    [data-testid="stTabs"] button {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #d946ef !important;
        border-bottom: 2px solid #d946ef !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA DE NAVEGACIÓN
# ==========================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h2 style='color: white; margin: 0;'>🧠 Deep Learning</h2>
</div>
""", unsafe_allow_html=True)

seccion = st.sidebar.radio("Menú Principal", [
    "🏠 Inicio",
    "📊 Análisis y SQL",
    "🧠 Red Neuronal (MLP)",
    "🔮 Predicción en Tiempo Real",
    "📄 Conclusiones"
])

st.sidebar.markdown("""
<div style='padding: 1rem; text-align: center; color: #93c5fd; font-size: 0.85rem;'>
    <p>ACA N° 2 - Redes Neuronales</p>
    <p>© 2026 - CUN Especialización IA</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. GESTIÓN SQL Y PIPELINE DE PREPROCESAMIENTO
# ==========================================
@st.cache_data
def init_and_load_sql():
    db_name = 'desercion_estudiantes.db'
    conn = sqlite3.connect(db_name)
    
    # Verificar si la tabla ya existe en SQLite; si no, cargarla desde data.csv
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='estudiantes';")
    if not cursor.fetchone():
        df_csv = pd.read_csv('data.csv', sep=';')
        df_csv.columns = df_csv.columns.str.strip()
        df_csv.to_sql('estudiantes', conn, if_exists='replace', index=False)
    
     # Consultar los datos desde SQL (Paso 1 del ACA 2)
    df_sql = pd.read_sql_query("SELECT * FROM estudiantes", conn)
    conn.close()
    return df_sql

df_raw = init_and_load_sql()
df = df_raw.copy()

# Guardar codificadores por columna para usarlos en la predicción en tiempo real
encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Target':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Codificar variable objetivo
df['Target'] = df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

# Separar características y objetivo
X = df.drop('Target', axis=1)
y = df['Target']

# División de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado de características (Obligatorio para Redes Neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. ENTRENAMIENTO DE RED NEURONAL (MLP)
# ==========================================
@st.cache_resource
def entrenar_red_neuronal(X_tr, y_tr):
    # Perceptrón Multicapa (MLP) de 3 capas ocultas: 128 -> 64 -> 32 neuronas
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    mlp.fit(X_tr, y_tr)
    return mlp

model_mlp = entrenar_red_neuronal(X_train_scaled, y_train)
preds_mlp = model_mlp.predict(X_test_scaled)

acc_mlp = accuracy_score(y_test, preds_mlp)
f1_mlp = f1_score(y_test, preds_mlp, average='weighted')

def crear_pdf(datos, res_mlp, probs):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0.12, 0.23, 0.55)
    c.drawCentredString(width/2, height - 50, "Reporte Neuronal - Deserción Escolar (ACA 2)")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    c.line(50, height - 70, 560, height - 70)
    
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.15, 0.26, 0.55)
    c.drawString(50, height - 100, "📝 Perfil del Estudiante Evaluado")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    y_pos = height - 130
    for clave, valor in datos.items():
        if y_pos < 80:
            c.showPage()
            y_pos = height - 50
        c.drawString(50, y_pos, f"{clave}: {valor}")
        y_pos -= 20
    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(0.15, 0.26, 0.55)
    c.drawString(50, y_pos - 20, "🧠 Diagnóstico de la Red Neuronal (MLP)")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    y_pos -= 50
    c.drawString(50, y_pos, f"Predicción Principal: {res_mlp}")
    if probs is not None:
        y_pos -= 25
        c.setFont("Helvetica", 11)
        c.drawString(50, y_pos, f"Probabilidades -> Deserción: {probs[0]:.1%} | Matriculado: {probs[1]:.1%} | Graduado: {probs[2]:.1%}")

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(width/2, 50, "🎓 Generado por el Sistema de Aprendizaje Profundo - CUN")
    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
# SECCIÓN: INICIO
# ==========================================
if seccion == "🏠 Inicio":
    st.markdown("""
    <div class='title-capsule'>
        <h1>🧠 Predicción de Deserción Escolar con Deep Learning</h1>
        <p>Arquitectura Perceptrón Multicapa (MLP) + Gestión de Datos en SQLite</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>📋 Ficha Técnica del Proyecto (ACA N° 2)</h3>
        <ul style='font-size: 1rem; line-height: 1.8;'>
            <li><strong>Asignatura:</strong> Aprendizaje Automático (Especialización en IA - CUN)</li>
            <li><strong>Pilar Temático:</strong> Proyecto 2 - Educación (Analítica de Trayectorias Formativas)</li>
            <li><strong>Base de Datos:</strong> Almacenamiento y consulta estructurada en <code>SQLite</code></li>
            <li><strong>Arquitectura Neuronal:</strong> Perceptrón Multicapa Profundo (3 capas ocultas: 128-64-32 neuronas, activación ReLU, optimizador Adam)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>📖 Propósito de la Solución</h3>
        <p style='font-size: 1rem; line-height: 1.7;'>
            Esta aplicación evoluciona el modelado estadístico hacia el <strong>Aprendizaje Profundo (Deep Learning)</strong>. 
            A través de un flujo automatizado que extrae registros desde una base de datos SQL, estandariza las variables 
            mediante <code>StandardScaler</code> y entrena una red neuronal multicapa, el sistema detecta patrones no lineales 
            complejos para anticipar el riesgo de deserción académica.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #38bdf8 !important;'>
            <h4 style='margin:0; color: #38bdf8 !important; font-size: 0.85rem; text-transform: uppercase;'>🗄️ Registros SQL</h4>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0.3rem 0;'>{len(df)}</p>
            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>Consultados desde SQLite</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #6366f1 !important;'>
            <h4 style='margin:0; color: #6366f1 !important; font-size: 0.85rem; text-transform: uppercase;'>🧬 Variables Entrada</h4>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0.3rem 0;'>{X.shape[1]}</p>
            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>Features normalizadas</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #d946ef !important;'>
            <h4 style='margin:0; color: #d946ef !important; font-size: 0.85rem; text-transform: uppercase;'>🎯 Exactitud MLP</h4>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0.3rem 0;'>{acc_mlp:.1%}</p>
            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>En conjunto de prueba</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #22c55e !important;'>
            <h4 style='margin:0; color: #22c55e !important; font-size: 0.85rem; text-transform: uppercase;'>⚖️ F1-Score</h4>
            <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0.3rem 0;'>{f1_mlp:.1%}</p>
            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>Promedio ponderado</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# SECCIÓN: ANÁLISIS DE DATOS Y SQL
# ==========================================
elif seccion == "📊 Análisis y SQL":
    st.title("📊 Exploración de Datos y Gestión SQL")
    
    tab1, tab2, tab3 = st.tabs(["🗄️ Consulta SQLite", "📈 Distribución", "🔗 Correlación"])
    
    with tab1:
        st.subheader("Consulta Directa a Base de Datos Relacional (SQLite)")
        st.markdown("El pipeline ingesta el dataset en `desercion_estudiantes.db` permitiendo consultas SQL eficientes:")
        query_ejemplo = st.text_input("Ejecutar consulta SQL sobre la tabla 'estudiantes':", 
                                      value="SELECT Target, COUNT(*) as Total, ROUND(AVG(`Admission grade`), 2) as Promedio_Admision FROM estudiantes GROUP BY Target;")
        try:
            conn = sqlite3.connect('desercion_estudiantes.db')
            df_query = pd.read_sql_query(query_ejemplo, conn)
            conn.close()
            st.dataframe(df_query, use_container_width=True)
        except Exception as e:
            st.error(f"Error en la consulta SQL: {e}")
            
        st.markdown("**Vista previa de registros almacenados:**")
        st.dataframe(df_raw.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Distribución de Estudiantes por Categoría")
        fig, ax = plt.subplots(figsize=(10, 5))
        colores = ['#ef4444', '#f59e0b', '#22c55e']
        df_raw['Target'].value_counts().plot(kind='bar', color=colores, ax=ax, edgecolor='white', linewidth=1.5)
        ax.set_title('Cantidad de Estudiantes por Estado Académico', fontsize=14, pad=15, fontweight='bold', color='#38bdf8')
        ax.set_xlabel('Estado', fontsize=11)
        ax.set_ylabel('Número de Estudiantes', fontsize=11)
        for i, v in enumerate(df_raw['Target'].value_counts()):
            ax.text(i, v + 15, str(v), ha='center', fontweight='bold', fontsize=11, color='white')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Matriz de Correlación de Variables")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='magma', center=0, linewidths=0.3, ax=ax)
        ax.set_title('Mapa de Calor de Correlaciones', fontsize=14, pad=15, fontweight='bold', color='#38bdf8')
        st.pyplot(fig)

# ==========================================
# SECCIÓN: RED NEURONAL (MLP) Y CURVAS DE PÉRDIDA
# ==========================================
elif seccion == "🧠 Red Neuronal (MLP)":
    st.title("🧠 Arquitectura y Evaluación de la Red Neuronal")
    
    # Tarjeta resumen de arquitectura
    st.markdown(f"""
    <div class='card'>
        <h3>🏗️ Topología de la Red Neuronal Artificial (MLPClassifier)</h3>
        <p><strong>Capa de Entrada:</strong> {X.shape[1]} neuronas (variables estandarizadas con Z-score) &rarr; 
        <strong>Capas Ocultas:</strong> 3 capas densas (128 &rarr; 64 &rarr; 32 neuronas) con activación <code>ReLU</code> &rarr; 
        <strong>Capa de Salida:</strong> 3 neuronas con activación <code>Softmax</code> (Deserción, Matriculado, Graduado).</p>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("🎯 Exactitud (Accuracy)", f"{acc_mlp:.2%}")
    col_m2.metric("⚖️ F1-Score Ponderado", f"{f1_mlp:.2%}")
    col_m3.metric("🔄 Épocas Entrenadas", f"{model_mlp.n_iter_} épocas")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Curva de Pérdida (Training Loss)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(model_mlp.loss_curve_, color='#d946ef', linewidth=2.5, label='Pérdida de Entrenamiento (Log-Loss)')
        ax.set_title('Evolución de la Función de Pérdida por Época', fontsize=13, fontweight='bold', color='#38bdf8')
        ax.set_xlabel('Épocas (Iteraciones)', fontsize=11)
        ax.set_ylabel('Pérdida (Loss)', fontsize=11)
        ax.legend(facecolor='#0d0d26', edgecolor='#6366f1')
        ax.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig)
        
        if hasattr(model_mlp, 'validation_scores_') and model_mlp.validation_scores_:
            st.subheader("📈 Curva de Exactitud en Validación")
            fig_val, ax_val = plt.subplots(figsize=(8, 5))
            ax_val.plot(model_mlp.validation_scores_, color='#38bdf8', linewidth=2.5, label='Exactitud de Validación')
            ax_val.set_title('Evolución de Exactitud en Validación Interna', fontsize=13, fontweight='bold', color='#38bdf8')
            ax_val.set_xlabel('Épocas (Iteraciones)', fontsize=11)
            ax_val.set_ylabel('Accuracy Validación', fontsize=11)
            ax_val.legend(facecolor='#0d0d26', edgecolor='#6366f1')
            ax_val.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig_val)

    with col2:
        st.subheader("🔲 Matriz de Confusión")
        cm_mlp = confusion_matrix(y_test, preds_mlp)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 5))
        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Purples', ax=ax_cm,
                    xticklabels=['Deserción', 'Matriculado', 'Graduado'],
                    yticklabels=['Deserción', 'Matriculado', 'Graduado'],
                    cbar=False, linewidths=1, linecolor='#050515')
        ax_cm.set_ylabel('Clase Real', fontweight=600)
        ax_cm.set_xlabel('Clase Predicha por la Red', fontweight=600)
        ax_cm.set_title('Matriz de Confusión - Conjunto de Prueba', fontsize=13, fontweight='bold', color='#38bdf8')
        st.pyplot(fig_cm)
        
        st.subheader("📋 Reporte Detallado de Clasificación")
        st.code(classification_report(
            y_test, preds_mlp, 
            target_names=['🔴 Deserción', '🟡 Matriculado', '🟢 Graduado'], 
            zero_division=0
        ))

# ==========================================
# SECCIÓN: PREDICCIÓN EN TIEMPO REAL
# ==========================================
elif seccion == "🔮 Predicción en Tiempo Real":
    st.title("🔮 Inferencia Neuronal en Tiempo Real")
    
    mapa = {0: "🔴 Deserción", 1: "🟡 Matriculado", 2: "🟢 Graduado"}
    
    st.markdown("### 📝 Ingrese los Datos del Estudiante")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("🎂 Edad al matricularse", 
                              min_value=int(df_raw['Age at enrollment'].min()),
                              max_value=int(df_raw['Age at enrollment'].max()),
                              value=int(df_raw['Age at enrollment'].median()))
        gender_options = sorted(df_raw['Gender'].unique().tolist())
        gender = st.selectbox("👤 Género", options=gender_options)
        admission_grade = st.number_input("📝 Nota de Admisión", 
                                         min_value=0.0, max_value=200.0,
                                         value=float(df_raw['Admission grade'].median()))
    
    with col2:
        col1_sem = st.number_input("📚 Unidades matriculadas (1er Sem)", min_value=0, 
                                   value=int(df_raw['Curricular units 1st sem (enrolled)'].median()))
        app1_sem = st.number_input("✅ Unidades aprobadas (1er Sem)", min_value=0, 
                                   value=int(df_raw['Curricular units 1st sem (approved)'].median()))
        grade1_sem = st.number_input("⭐ Nota promedio (1er Sem)", min_value=0.0, max_value=20.0,
                                     value=float(df_raw['Curricular units 1st sem (grade)'].median()))
    
    with col3:
        col2_sem = st.number_input("📚 Unidades matriculadas (2do Sem)", min_value=0, 
                                   value=int(df_raw['Curricular units 2nd sem (enrolled)'].median()))
        app2_sem = st.number_input("✅ Unidades aprobadas (2do Sem)", min_value=0, 
                                   value=int(df_raw['Curricular units 2nd sem (approved)'].median()))
        grade2_sem = st.number_input("⭐ Nota promedio (2do Sem)", min_value=0.0, max_value=20.0,
                                     value=float(df_raw['Curricular units 2nd sem (grade)'].median()))
    
    col4, col5 = st.columns(2)
    with col4:
        debtor_options = sorted(df_raw['Debtor'].unique().tolist())
        debtor = st.selectbox("💰 ¿Es Deudor?", options=debtor_options)
    with col5:
        tuition_options = sorted(df_raw['Tuition fees up to date'].unique().tolist())
        tuition_up_to_date = st.selectbox("📄 ¿Matrícula al día?", options=tuition_options)
    
    input_data = {
        'Age at enrollment': age,
        'Gender': gender,
        'Admission grade': admission_grade,
        'Curricular units 1st sem (enrolled)': col1_sem,
        'Curricular units 1st sem (approved)': app1_sem,
        'Curricular units 1st sem (grade)': grade1_sem,
        'Curricular units 2nd sem (enrolled)': col2_sem,
        'Curricular units 2nd sem (approved)': app2_sem,
        'Curricular units 2nd sem (grade)': grade2_sem,
        'Debtor': debtor,
        'Tuition fees up to date': tuition_up_to_date
    }
    
    # Construir fila completa usando la mediana de las variables restantes para no distorsionar la red
    full_input = X.median().to_frame().T
    for k, v in input_data.items():
        if k in encoders:
            full_input[k] = encoders[k].transform([v])[0]
        else:
            full_input[k] = v
            
    # Aplicar el mismo StandardScaler del entrenamiento
    full_input_scaled = scaler.transform(full_input[X.columns])
    
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predecir = st.button("🧠 Ejecutar Inferencia Neuronal", type="primary", use_container_width=True)
    with col_btn2:
        limpiar = st.button("🔄 Limpiar Resultado", use_container_width=True)
        
    if limpiar:
        st.session_state.resultado_mlp = ""
        st.session_state.probabilidades_mlp = None
        st.rerun()
        
    if predecir:
        pred = model_mlp.predict(full_input_scaled)[0]
        probs = model_mlp.predict_proba(full_input_scaled)[0]
        st.session_state.resultado_mlp = mapa[pred]
        st.session_state.probabilidades_mlp = probs
        
    if st.session_state.resultado_mlp != "":
        st.markdown("### 📋 Resultado de la Red Neuronal")
        st.info(f"**Diagnóstico Predicho:** {st.session_state.resultado_mlp}")
        
        probs = st.session_state.probabilidades_mlp
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("🔴 Prob. Deserción", f"{probs[0]:.1%}")
        p_col2.metric("🟡 Prob. Matriculado", f"{probs[1]:.1%}")
        p_col3.metric("🟢 Prob. Graduado", f"{probs[2]:.1%}")
        
        pdf_buffer = crear_pdf(input_data, st.session_state.resultado_mlp, probs)
        st.download_button(
            label="📄 Descargar Reporte Neuronal PDF",
            data=pdf_buffer,
            file_name="reporte_neuronal_aca2.pdf",
            mime="application/pdf",
            use_container_width=True
        )
# ==========================================
# SECCIÓN: CONCLUSIONES
# ==========================================
elif seccion == "📄 Conclusiones":
    st.title("📄 Conclusiones y Síntesis del Proyecto")
    
    st.markdown(f"""
    <div class='card'>
        <h3>🔍 Resumen de Resultados (ACA N° 2)</h3>
        <p style='font-size: 1rem; line-height: 1.7;'>
            La integración de una base de datos relacional (<strong>SQLite</strong>) junto con una arquitectura de 
            <strong>Perceptrón Multicapa Profundo (MLP)</strong> permitió alcanzar una exactitud de <strong>{acc_mlp:.2%}</strong> 
            y un F1-Score ponderado de <strong>{f1_mlp:.2%}</strong> en la clasificación de trayectorias estudiantiles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>📌 Hallazgos Técnicos y Académicos</h3>
        <div style='display: flex; flex-direction: column; gap: 0.8rem; margin-top: 1rem;'>
            <div style='background: #050515; padding: 0.75rem 1rem; border-radius: 6px; border-left: 3px solid #6366f1;'>
                <strong style='color: #38bdf8;'>⚙️ Impacto del Preprocesamiento:</strong> La estandarización mediante <code>StandardScaler</code> resultó indispensable para estabilizar el descenso de gradiente (Adam) y lograr una convergencia suave en la curva de pérdida.
            </div>
            <div style='background: #050515; padding: 0.75rem 1rem; border-radius: 6px; border-left: 3px solid #d946ef;'>
                <strong style='color: #d946ef;'>🧠 Capacidad No Lineal:</strong> Las tres capas ocultas (128-64-32) capturaron interacciones complejas entre el desempeño del primer y segundo semestre y la situación financiera del estudiante.
            </div>
            <div style='background: #050515; padding: 0.75rem 1rem; border-radius: 6px; border-left: 3px solid #22c55e;'>
                <strong style='color: #22c55e;'>🗄️ Escalabilidad SQL:</strong> El almacenamiento en SQLite desacopla la ingesta de datos planos y permite realizar consultas analíticas en tiempo real desde la interfaz web.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    



