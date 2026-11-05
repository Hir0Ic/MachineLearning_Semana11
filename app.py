"""Descubrimiento de patrones fisicoquímicos en vinos: K-Means, DBSCAN y PCA."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import streamlit as st

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuración global e incondicional de tema oscuro para Matplotlib y Seaborn
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161922",
    "savefig.facecolor": "#0e1117",
    "savefig.edgecolor": "none",
    "text.color": "#f0f2f6",
    "axes.labelcolor": "#f0f2f6",
    "axes.edgecolor": "#333846",
    "axes.linewidth": 0.8,
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": "#232836",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#1c202c",
    "legend.edgecolor": "#333846",
    "legend.labelcolor": "#f0f2f6",
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "axes.titlesize": 11,
})

# Módulos desacoplados del proyecto
from src.styles import aplicar_estilos_ui, COLOR_RUIDO
from src.data_loader import (
    SECCIONES,
    FICHA,
    cargar_wine,
    leer_archivo,
    obtener_datos,
    en_construccion,
    requerir_datos,
    obtener_datos_preprocesados,
    matriz_para_clustering,
)
from src.models import (
    RANDOM_STATE,
    N_INIT,
    K_MIN,
    K_MAX,
    EPS_DEFECTO,
    MIN_SAMPLES_DEFECTO,
    GRILLA_EPS,
    GRILLA_MIN_SAMPLES,
    metricas_internas,
    evaluar_rango_k,
    ajustar_kmeans,
    distancias_k,
    ajustar_dbscan,
    sensibilidad_dbscan,
)
from src.plots import (
    HAS_SEABORN,
    proyeccion_2d,
    colores_clusters,
    dibujar_heatmap,
    tabla_vs_etiqueta,
)

st.set_page_config(page_title="Agrupamiento de vinos", page_icon="🍷", layout="wide", initial_sidebar_state="collapsed")
aplicar_estilos_ui()


# ---------------------------------------------------------------- Inicio
def pagina_inicio():
    # 1. Hero Banner Institucional (Universidad, Carrera y Proyecto)
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, rgba(22,25,34,0.85) 100%);
                    border: 1px solid rgba(244,63,94,0.3); border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #f43f5e; text-transform: uppercase; letter-spacing: 0.08em;">
              🏛️ Universidad Privada del Norte · Facultad de Ingeniería
            </span>
            <span style="background: rgba(244,63,94,0.18); color: #f43f5e; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(244,63,94,0.35);">
              GRUPO 03 · APRENDIZAJE NO SUPERVISADO
            </span>
          </div>
          <h1 style="margin: 4px 0 8px 0; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
            🍷 Descubrimiento de Patrones Fisicoquímicos en Vinos
          </h1>
          <p style="font-size: 1.02rem; color: #cbd5e1; margin: 0; line-height: 1.55;">
            Modelado no supervisado con <strong>K-Means</strong>, <strong>DBSCAN</strong> y <strong>PCA</strong> sobre el dataset <em>Wine (UCI)</em>.<br>
            <span style="color: #94a3b8; font-size: 0.92rem;">
              Sistemas Inteligentes y Machine Learning · Ingeniería de Sistemas · 2026
            </span>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



    # 3. Métricas Rápidas del Estudio (KPIs de Introducción)
    df, _ = obtener_datos()
    n_muestras = df.shape[0] if df is not None else 178
    n_vars = (df.shape[1] - 1) if df is not None and "target" in df.columns else 13

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Muestras de Vino", f"{n_muestras}", "Unidades de análisis")
    kpi2.metric("Variables Químicas", f"{n_vars}", "Predictoras continuas")
    kpi3.metric("Cultivares (Target)", "3", "Excluidos de la matriz X")
    kpi4.metric("Calidad de Datos", "100 %", "0 nulos y 0 duplicados")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Pestañas de la Sección 1 (Informe y Diapositivas)
    tab_preg, tab_obj, tab_lim = st.tabs([
        "🎯 1. Pregunta Central y Contexto",
        "📌 2. Objetivos del Proyecto",
        "🔬 3. Unidad de Análisis y Límites",
    ])

    with tab_preg:
        st.subheader("Contexto y Pregunta de Análisis")
        st.markdown(
            """
            Los viñedos y laboratorios enológicos generan información química abundante sobre cada muestra de vino, pero rara vez cuentan con una clasificación confiable que indique qué perfiles existen realmente en ella. Un laboratorio desea identificar perfiles de muestras a partir de indicadores químicos (alcohol, fenoles, color, prolina, entre otros), sin usar las clases de referencia para formar los grupos. 

            Para ello se utilizó el dataset Wine del UCI Machine Learning Repository (Aeberhard & Forina, 1992), que reúne 178 muestras de vinos italianos descritas por 13 constituyentes químicos. Sobre estos datos se aplican K-Means y DBSCAN para el agrupamiento, y PCA para reducir la dimensionalidad y visualizar los resultados, todo integrado en una aplicación web desarrollada en Streamlit.
            """
        )
        st.info(
            "❓ **Pregunta central:** ¿qué patrones o segmentos se pueden descubrir en los datos y qué evidencia demuestra que los resultados son interpretables y útiles?"
        )
        st.markdown("---")
        st.markdown("**🔬 1.3 Preguntas de Investigación del Informe (Desglose Interactivo):**")
        st.caption("Selecciona una pregunta de investigación en el menú desplegable para ver su respuesta analítica:")

        preguntas_lista = [
            "1. ¿Cuáles variables caracterizan la separación de grupos?",
            "2. ¿Cuánta varianza conservan dos y tres componentes principales?",
            "3. ¿Coinciden o divergen los resultados de K-Means y DBSCAN?",
            "📋 Ver las 3 preguntas desglosadas (Vista Completa)",
        ]

        pregunta_sel = st.selectbox(
            "Selecciona una pregunta de investigación:",
            preguntas_lista,
            label_visibility="collapsed",
        )

        def _render_resp_1():
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.75); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 18px 20px; margin-top: 10px; margin-bottom: 12px;">
                  <div style="color: #38bdf8; font-weight: 800; font-size: 1.02rem; margin-bottom: 8px;">
                    🍇 Respuesta a la Pregunta 1: Variables que caracterizan la separación de grupos
                  </div>
                  <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.55; margin-bottom: 10px;">
                    La separación de los perfiles químicos está dominada principalmente por dos bloques moleculares complementarios y un aminoácido estructural:
                  </p>
                  <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 12px;">
                    <li><strong>Bloque Fenólico y Antioxidante (PC1):</strong> <code>flavanoids</code> (rango 0.34 a 5.08), <code>total_phenols</code> y la absorbancia <code>od280/od315</code>. Es el bloque con mayor poder discriminante; su bimodalidad observada en el EDA separa con nitidez los vinos maduros frente a los astringentes o jóvenes.</li>
                    <li><strong>Bloque de Vigor, Intensidad y Grado (PC2):</strong> <code>color_intensity</code> (1.28 a 13.00), <code>alcohol</code> (11.03° a 14.83°) y <code>malic_acid</code> (acidez fija). Separa vinos tintos de cuerpo robusto frente a vinos frescos y de alta acidez.</li>
                    <li><strong>Marcador Estructural Clave:</strong> <code>proline</code> (278 a 1,680 mg/L). Aunque requirió estandarización obligatoria por su enorme escala (rango 1,402), es la variable que mejor aísla al Clúster 2 de alta graduación alcohólica (media ≈ 1,100).</li>
                  </ul>
                  <div style="background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; padding: 8px 14px; border-radius: 0 8px 8px 0; font-size: 0.84rem; color: #bae6fd;">
                    <strong>Evidencia en el informe:</strong> Bimodalidad en los histogramas del EDA (Sec. 3.3.4), correlación lineal |r| = 0.865 en Tabla 3C y mayores cargas factoriales en PCA (Sec. 4.3).
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def _render_resp_2():
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.75); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 12px; padding: 18px 20px; margin-top: 10px; margin-bottom: 12px;">
                  <div style="color: #34d399; font-weight: 800; font-size: 1.02rem; margin-bottom: 8px;">
                    📉 Respuesta a la Pregunta 2: Varianza retenida por componentes principales
                  </div>
                  <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.55; margin-bottom: 12px;">
                    Al proyectar las 13 dimensiones químicas estandarizadas sobre el espacio factorial de PCA, los componentes concentran la variabilidad de la siguiente forma:
                  </p>
                  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px;">
                    <div style="background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 8px; padding: 10px 14px; text-align: center;">
                      <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">PC1 + PC2 (2D)</div>
                      <div style="font-size: 1.5rem; font-weight: 800; color: #34d399;">55.41 %</div>
                      <div style="font-size: 0.76rem; color: #cbd5e1;">PC1: 36.20% | PC2: 19.21%</div>
                    </div>
                    <div style="background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 8px; padding: 10px 14px; text-align: center;">
                      <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">PC1 + PC2 + PC3 (3D)</div>
                      <div style="font-size: 1.5rem; font-weight: 800; color: #34d399;">66.50 %</div>
                      <div style="font-size: 0.76rem; color: #cbd5e1;">Aporte de PC3: 11.09%</div>
                    </div>
                    <div style="background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 8px; padding: 10px 14px; text-align: center;">
                      <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Umbral Convencional (80%)</div>
                      <div style="font-size: 1.5rem; font-weight: 800; color: #fbbf24;">5 CP (80.16 %)</div>
                      <div style="font-size: 0.76rem; color: #cbd5e1;">Requeridos para el 80%</div>
                    </div>
                  </div>
                  <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55; margin-bottom: 0;">
                    <strong>Interpretación:</strong> Dos componentes (55.41 %) son suficientes para una <em>síntesis visual intuitiva</em> en dos ejes interpretables, pero suponen una pérdida del 44.59 % de información. Por ello, el agrupamiento con K-Means y DBSCAN se ejecutó obligatoriamente sobre las <strong>13 variables completas</strong> y no sobre el plano reducido.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def _render_resp_3():
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.75); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 12px; padding: 18px 20px; margin-top: 10px; margin-bottom: 12px;">
                  <div style="color: #f43f5e; font-weight: 800; font-size: 1.02rem; margin-bottom: 8px;">
                    ⚖️ Respuesta a la Pregunta 3: Coincidencia o divergencia entre K-Means y DBSCAN
                  </div>
                  <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.55; margin-bottom: 12px;">
                    <strong>Divergen sustancialmente</strong> debido a las naturalezas matemáticas opuestas de ambos algoritmos (partición convexa vs. conectividad por densidad):
                  </p>
                  <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 12px;">
                    <li><strong>K-Means (Partición esférica):</strong> Divide obligatoriamente al <strong>100 % de las muestras</strong> (178/178) en 3 perfiles balanceados (65, 51 y 62 vinos). Coincide en 96.6 % con los tres cultivares reales de uva.</li>
                    <li><strong>DBSCAN (Densidad):</strong> Con &epsilon; = 2.3 y min_samples = 5, identifica únicamente <strong>2 regiones densas</strong> (94 y 42 muestras) y aísla <strong>42 muestras como ruido (-1)</strong>, alcanzando una cobertura de solo 76.4 %. Fusiona los clústeres 0 y 2 de K-Means porque en 13 dimensiones comparten un corredor continuo de densidad.</li>
                    <li><strong>El Hallazgo Clave de Coincidencia:</strong> El Clúster 1 de DBSCAN (42 vinos) queda <em>100 % contenido</em> dentro del Clúster 1 de K-Means (vinos de baja madurez fenólica). Además, <strong>13 de los 17 vinos con valores atípicos del EDA (76.5 %)</strong> fueron capturados por DBSCAN como ruido.</li>
                  </ul>
                  <div style="background: rgba(244, 63, 94, 0.1); border-left: 3px solid #f43f5e; padding: 8px 14px; border-radius: 0 8px 8px 0; font-size: 0.84rem; color: #fecdd3;">
                    <strong>Veredicto de negocio:</strong> Son herramientas complementarias. K-Means sirve para segmentar comercialmente la totalidad del catálogo; DBSCAN sirve como filtro de auditoría para detectar lotes atípicos.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if pregunta_sel.startswith("1."):
            _render_resp_1()
        elif pregunta_sel.startswith("2."):
            _render_resp_2()
        elif pregunta_sel.startswith("3."):
            _render_resp_3()
        else:
            _render_resp_1()
            _render_resp_2()
            _render_resp_3()

    with tab_obj:
        st.subheader("Objetivos del Proyecto")
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.4); border-left: 4px solid #f43f5e; padding: 14px 18px; border-radius: 0 10px 10px 0; margin-bottom: 16px;">
              <strong style="color: #f43f5e; font-size: 0.95rem;">🎯 OBJETIVO GENERAL:</strong><br>
              <span style="color: #e2e8f0; font-size: 0.92rem;">
                Desarrollar y evaluar una aplicación en Streamlit que permita obtener, describir y comparar agrupamientos de muestras de vino mediante K-Means y DBSCAN, incorporando PCA como análisis de dimensionalidad y herramienta de visualización, para descubrir perfiles fisicoquímicos interpretables sin emplear las clases de referencia.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**📌 Objetivos Específicos:**")
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; height: 100%;">
                  <div style="color: #38bdf8; font-weight: 800; font-size: 1.05rem; margin-bottom: 6px;">01. PREPARACIÓN</div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.45;">
                    <strong>Preparar las variables químicas pertinentes:</strong> separar la etiqueta externa, verificar nulos y duplicados, y estandarizar la matriz de entrada.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_o2:
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; height: 100%;">
                  <div style="color: #34d399; font-weight: 800; font-size: 1.05rem; margin-bottom: 6px;">02. MODELADO Y MÉTRICAS</div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.45;">
                    <strong>Evaluar al menos dos configuraciones de agrupamiento:</strong> (K-Means y DBSCAN) mediante <em>inertia</em>, <em>Silhouette</em>, <em>Davies-Bouldin</em> y <em>Calinski-Harabasz</em>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_o3:
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; height: 100%;">
                  <div style="color: #fbbf24; font-weight: 800; font-size: 1.05rem; margin-bottom: 6px;">03. PCA Y RECOMENDACIONES</div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.45;">
                    <strong>Interpretar PCA:</strong> (varianza explicada y cargas) y convertir los hallazgos en recomendaciones para el laboratorio.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_lim:
        st.subheader("Unidad de Análisis, Relevancia y Límites")
        st.caption("Síntesis metodológica del alcance del estudio:")

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="color: #38bdf8; font-weight: 800; font-size: 0.98rem;">📌 1. UNIDAD DE OBSERVACIÓN</span>
                    <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">1 MUESTRA = 13 VARIABLES</span>
                  </div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">
                    La unidad de observación es la muestra de vino; cada registro corresponde a un vino descrito por 13 variables químicas continuas o enteras.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 18px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="color: #34d399; font-weight: 800; font-size: 0.98rem;">💡 3. RELEVANCIA DEL ESTUDIO</span>
                    <span style="background: rgba(52, 211, 153, 0.15); color: #34d399; padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">CONTROL Y AUDITORÍA</span>
                  </div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">
                    Identificar perfiles químicos permite apoyar la caracterización y el control de calidad de lotes, detectar muestras atípicas (ruido en DBSCAN) y generar hipótesis sobre qué variables diferencian a los vinos, con menor dependencia de clasificaciones manuales.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_u2:
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="color: #fbbf24; font-weight: 800; font-size: 0.98rem;">🏷️ 2. ENFOQUE NO SUPERVISADO</span>
                    <span style="background: rgba(251, 191, 36, 0.15); color: #fbbf24; padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">TARGET EXCLUIDO</span>
                  </div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">
                    K-Means y DBSCAN agrupan según la similitud entre observaciones en el espacio de variables, sin necesitar una etiqueta. La clase original (cultivar) del dataset se excluye del entrenamiento y de la elección de K y solo se usa, de forma posterior, como referencia externa para contrastar si los grupos hallados tienen sentido.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 18px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="color: #f43f5e; font-weight: 800; font-size: 0.98rem;">⚠️ 4. LÍMITES METODOLÓGICOS</span>
                    <span style="background: rgba(244, 63, 94, 0.15); color: #f43f5e; padding: 2px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">178 VINOS · 1 REGIÓN</span>
                  </div>
                  <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">
                    El dataset es pequeño (178 muestras), proviene de una sola región de Italia y de tres cultivares, y la ficha no documenta las unidades de medida. Los resultados describen patrones de esta muestra y no deben generalizarse a otras regiones o cosechas sin validación.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------- Datos
def pagina_datos():
    st.header("Datos")
    origen = st.radio(
        "Fuente de datos",
        ["Dataset Wine (load_wine)", "Subir archivo (CSV o Excel)"],
        horizontal=True,
    )

    if origen.startswith("Dataset"):
        st.session_state["df"] = cargar_wine()
        st.session_state["etiqueta"] = "target"
    else:
        archivo = st.file_uploader("Archivo de datos", type=["csv", "xlsx", "xls"])
        if archivo is None:
            st.info("Sube un archivo para continuar.")
            return
        try:
            df_up = leer_archivo(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return

        if df_up.empty or df_up.shape[0] < 3:
            st.error("⚠️ El archivo cargado está vacío o contiene menos de 3 filas. Sube un dataset válido.")
            return

        cols_num = df_up.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols_num) < 2:
            st.error("⚠️ El archivo debe contener al menos 2 columnas numéricas para realizar análisis multivariado.")
            return
        opciones = ["(ninguna)"] + list(df_up.columns)
        elegida = st.selectbox("Columna de etiqueta externa (se excluye del clustering)", opciones)
        st.session_state["df"] = df_up
        st.session_state["etiqueta"] = None if elegida == "(ninguna)" else elegida

    df, etiqueta = obtener_datos()
    variables = [c for c in df.columns if c != etiqueta]

    # Banner de KPIs de Dimensiones y Calidad
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Muestras Totales", f"{df.shape[0]}", "Filas (N)")
    c2.metric("Variables Predictoras", f"{len(variables)}", "Columnas continuas")
    c3.metric("Valores Nulos", f"{int(df.isna().sum().sum())}", "Datos completos")
    c4.metric("Registros Duplicados", f"{int(df.duplicated().sum())}", "0 duplicados")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab_fic, tab_dic, tab_reg = st.tabs([
        "📋 1. Ficha Técnica y Calidad",
        "📖 2. Diccionario de Variables y Familias",
        "👁️ 3. Muestra de Registros",
    ])

    with tab_fic:
        st.subheader("Fuente y Ficha Técnica del Dataset")
        if origen.startswith("Dataset"):
            df_ficha = pd.DataFrame([
                {"Campo": "Dataset", "Detalle": "Wine — UCI Machine Learning Repository (Aeberhard & Forina, 1992)"},
                {"Campo": "Enlace oficial", "Detalle": "https://archive.ics.uci.edu/dataset/109/wine"},
                {"Campo": "DOI", "Detalle": "10.24432/C5PC7J"},
                {"Campo": "Autores / Editores", "Detalle": "Stefan Aeberhard y M. Forina (Universidad de Génova, Italia)"},
                {"Campo": "Licencia", "Detalle": "Creative Commons Attribution 4.0 International (CC BY 4.0)"},
                {"Campo": "Unidad de observación", "Detalle": "Muestra de vino (13 determinaciones analíticas físico-químicas)"},
                {"Campo": "N.º registros / variables", "Detalle": "178 registros; 13 variables predictoras continuas/enteras + 1 etiqueta externa (3 cultivares)"},
                {"Campo": "Calidad inicial", "Detalle": "0 valores faltantes (sin imputación) y 0 registros duplicados"},
            ])
            st.table(df_ficha.set_index("Campo"))

            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.45); border-left: 4px solid #f43f5e; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-top: 10px;">
                  <strong style="color: #f43f5e; font-size: 0.95rem;">⚠️ Posibles Sesgos del Dataset:</strong>
                  <ul style="color: #cbd5e1; font-size: 0.88rem; margin: 8px 0 0 0; line-height: 1.55;">
                    <li><strong>Muestra pequeña:</strong> 178 observaciones limitan la generalización a escala industrial.</li>
                    <li><strong>Ámbito geográfico restringido:</strong> Muestras procedentes de una sola región vitivinícola de Italia.</li>
                    <li><strong>Clases moderadamente desbalanceadas:</strong> 59, 71 y 48 muestras por cultivar.</li>
                    <li><strong>Datos históricos (1991):</strong> Pueden no reflejar vendimias modernas ni técnicas contemporáneas.</li>
                    <li><strong>Unidades de medida:</strong> No están documentadas explícitamente en el repositorio UCI.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Archivo cargado por el usuario. La ficha técnica del repositorio UCI aplica exclusivamente al dataset oficial Wine.")

    with tab_dic:
        st.subheader("Diccionario de Variables y Familias Químicas")
        if origen.startswith("Dataset"):
            diccionario_datos = pd.DataFrame([
                {"Familia Química": "🍇 Composición Básica", "Variable": "alcohol", "Descripción": "Graduación alcohólica del vino", "Tipo": "Continua"},
                {"Familia Química": "🍇 Composición Básica", "Variable": "malic_acid", "Descripción": "Ácido málico (acidez fija del mosto)", "Tipo": "Continua"},
                {"Familia Química": "🍇 Composición Básica", "Variable": "ash", "Descripción": "Cenizas totales (materia mineral inorgánica)", "Tipo": "Continua"},
                {"Familia Química": "🍇 Composición Básica", "Variable": "alcalinity_of_ash", "Descripción": "Alcalinidad de las cenizas", "Tipo": "Continua"},
                {"Familia Química": "🍇 Composición Básica", "Variable": "magnesium", "Descripción": "Contenido de magnesio", "Tipo": "Entera / Numérica"},
                {"Familia Química": "🧪 Bloque Fenólico", "Variable": "total_phenols", "Descripción": "Polifenoles totales (antioxidantes naturales)", "Tipo": "Continua"},
                {"Familia Química": "🧪 Bloque Fenólico", "Variable": "flavanoids", "Descripción": "Flavonoides (astringencia, sabor y longevidad)", "Tipo": "Continua"},
                {"Familia Química": "🧪 Bloque Fenólico", "Variable": "nonflavanoid_phenols", "Descripción": "Fenoles no flavonoides", "Tipo": "Continua"},
                {"Familia Química": "🧪 Bloque Fenólico", "Variable": "proanthocyanins", "Descripción": "Proantocianidinas (taninos condensados)", "Tipo": "Continua"},
                {"Familia Química": "🎨 Color y Matiz", "Variable": "color_intensity", "Descripción": "Intensidad colorimétrica espectral", "Tipo": "Continua"},
                {"Familia Química": "🎨 Color y Matiz", "Variable": "hue", "Descripción": "Matiz / tonalidad del color del vino", "Tipo": "Continua"},
                {"Familia Química": "🎨 Color y Matiz", "Variable": "od280/od315_of_diluted_wines", "Descripción": "Cociente de absorbancia proteica a 280/315 nm", "Tipo": "Continua"},
                {"Familia Química": "🧬 Marcador Estructural", "Variable": "proline", "Descripción": "Aminoácido prolina (madurez y extracto seco)", "Tipo": "Continua"},
                {"Familia Química": "🏷️ Etiqueta Externa", "Variable": "target (cultivar)", "Descripción": "3 cultivares: Barolo (59), Grignolino (71), Barbera (48)", "Tipo": "Categórica (EXCLUIDA de X)"},
            ])
            st.dataframe(diccionario_datos, use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div style="background: rgba(22, 25, 34, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 16px 20px; margin-top: 14px;">
                  <div style="color: #38bdf8; font-weight: 800; font-size: 0.98rem; margin-bottom: 6px;">
                    🔬 Esquema del Enfoque No Supervisado
                  </div>
                  <p style="color: #cbd5e1; font-size: 0.88rem; margin: 0; line-height: 1.55;">
                    <strong>13 Variables fisicoquímicas continuas</strong> ➔ Entrada al pipeline (K-Means / DBSCAN) ➔ <strong>Clústeres descubiertos</strong>.<br>
                    <span style="color: #f43f5e; font-weight: 700;">Variable Target (3 cultivares):</span> Se mantiene <strong>estrictamente aislada</strong> del modelado. No interviene en la estandarización ni en la elección de <em>K</em> o &epsilon;. Solo se emplea en la sección de Conclusiones como validación externa de sentido agronómico.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(pd.DataFrame({"Variable": df.columns, "Tipo": [str(df[c].dtype) for c in df.columns]}), use_container_width=True)

    with tab_reg:
        st.subheader("Muestra de Registros del Conjunto de Datos")
        c_slider, c_info = st.columns([1, 2])
        with c_slider:
            n = st.slider("Filas a visualizar:", 5, min(50, len(df)), 10)
        with c_info:
            if etiqueta:
                st.caption(f"Etiqueta externa activa: **{etiqueta}** (no utilizada para entrenar el agrupamiento).")
        st.dataframe(df.head(n), use_container_width=True)


# ------------------------------------------------------------------- EDA
def pagina_eda():
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    st.markdown(
        "**Propósito:** Verificar la calidad de los datos, comparar escalas, caracterizar distribuciones, "
        "auditar valores atípicos y medir colinealidades lineales antes de aplicar cualquier algoritmo de agrupamiento."
    )

    df, etiqueta = requerir_datos()
    if df is None:
        return

    vars_analisis = [c for c in df.columns if c != etiqueta]
    df_vars = df[vars_analisis]

    # --- Banner de 3 KPIs Ejecutivos (Diapositiva 7 del PPTX) ---
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">MUESTRAS Y PREDICTORAS</div>
            <div class="kpi-value">178 × 13</div>
            <div class="kpi-subtitle">0 nulos · 0 duplicados (calidad 100%, sin imputación)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">HETEROGENEIDAD DE ESCALAS</div>
            <div class="kpi-value">&gt; 2,000×</div>
            <div class="kpi-subtitle">Razón σ (proline: 314.9 vs nonflavanoid: 0.124) → Escalar es obligatorio</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">VALORES ATÍPICOS (TUKEY)</div>
            <div class="kpi-value">7 de 13</div>
            <div class="kpi-subtitle">Variables con atípicos (≤ 2.25% c/u) · Se conservan intencionalmente</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

    pestanas = st.tabs([
        "1. Estructura y Calidad",
        "2. Descriptivos y Escalas",
        "3. Distribuciones y Forma",
        "4. Valores Atípicos (Tukey)",
        "5. Correlación de Pearson",
        "6. Síntesis de Hallazgos",
    ])

    # 1. Estructura y Calidad (Sección 3.3.1 y Tabla 3 / 5)
    with pestanas[0]:
        st.subheader("3.3.1 Dimensión, tipos, nulos y duplicados")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total observaciones (N)", df.shape[0])
        col_b.metric("Variables predictoras (p)", len(vars_analisis))
        col_c.metric("Filas duplicadas", int(df_vars.duplicated().sum()))

        st.markdown("**Tabla 3. Resumen de tipos de datos y valores faltantes por columna:**")
        info_df = pd.DataFrame({
            "Variable": df_vars.columns,
            "Tipo de Dato": df_vars.dtypes.astype(str).values,
            "Valores Presentes": df_vars.count().values,
            "Valores Nulos": df_vars.isnull().sum().values,
            "% Nulos": (df_vars.isnull().mean() * 100).round(2).values,
            "Valores Únicos": df_vars.nunique().values,
        })
        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            height=(len(info_df) + 1) * 36 + 10,
        )

        st.markdown("""
        <div style="background-color: #161922; border-left: 4px solid #38bdf8; border-radius: 4px; padding: 12px 16px; margin-top: 15px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.92rem; margin-bottom: 4px;">📌 Interpretación Metodológica:</div>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                El conjunto de datos está <b>completo y rigurosamente limpio</b>: no hay valores faltantes ni registros repetidos, por lo que no fue necesario imputar ni eliminar filas. 
                Todas las variables son numéricas (la variable <code>magnesium</code> es entera en la ficha original de UCI, pero scikit-learn la carga como <code>float64</code>), 
                de modo que tampoco se requirió codificación de variables categóricas. Cada fila corresponde a una muestra de vino analizada instrumentalmente en laboratorio, constituyendo la unidad de análisis del proyecto.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Estadísticos Descriptivos y Heterogeneidad de Escalas (3.3.2 y 3.3.3)
    with pestanas[1]:
        st.subheader("3.3.2 y 3.3.3 Estadísticos descriptivos y heterogeneidad de escalas")
        desc = df_vars.describe().T
        desc["rango"] = desc["max"] - desc["min"]
        desc["iqr"] = desc["75%"] - desc["25%"]
        desc["skewness"] = df_vars.skew()

        columnas_orden = ["mean", "std", "min", "25%", "50%", "75%", "max", "rango", "iqr", "skewness"]
        nombres_amigables = {
            "mean": "Media", "std": "Desv. Est. (σ)", "min": "Mín", "25%": "Q1 (25%)",
            "50%": "Mediana", "75%": "Q3 (75%)", "max": "Máx", "rango": "Rango",
            "iqr": "IQR", "skewness": "Asimetría"
        }
        desc_mostrar = desc[columnas_orden].rename(columns=nombres_amigables).round(3)
        desc_mostrar.insert(0, "Variable", desc_mostrar.index)

        st.markdown("**Tabla 3A. Estadísticos descriptivos de las 13 variables fisicoquímicas (N = 178):**")
        st.dataframe(
            desc_mostrar,
            use_container_width=True,
            hide_index=True,
            height=(len(desc_mostrar) + 1) * 36 + 10,
        )

        col_desc_1, col_desc_2 = st.columns(2, gap="medium")
        with col_desc_1:
            st.markdown("""
            <div style="background-color: #161922; border: 1px solid #232734; border-radius: 6px; padding: 14px 16px; height: 100%;">
                <div style="font-weight: 600; color: #38bdf8; font-size: 0.92rem; margin-bottom: 6px;">📊 Interpretación de Asimetría:</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    • <b>Distribuciones aproximadamente simétricas:</b> En la mayoría de variables la media y la mediana son muy parecidas y la asimetría es cercana a cero (<code>alcohol</code>, <code>ash</code>, <code>alcalinity_of_ash</code>, <code>total_phenols</code>, <code>flavanoids</code> y <code>hue</code>).<br>
                    • <b>Colas largas hacia la derecha:</b> En cambio, <code>malic_acid</code> (asimetría +1.04), <code>magnesium</code> (+1.10), <code>color_intensity</code> (+0.87) y <code>proline</code> (+0.77) tienen la media por encima de la mediana (por ejemplo, <code>proline</code>: 746.9 frente a 673.5), señal de colas prolongadas hacia concentraciones altas.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_desc_2:
            st.markdown("""
            <div style="background-color: #161922; border: 1px solid #232734; border-radius: 6px; padding: 14px 16px; height: 100%;">
                <div style="font-weight: 600; color: #f43f5e; font-size: 0.92rem; margin-bottom: 6px;">⚖️ Heterogeneidad de Escalas:</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    • <b>Disparidad extrema:</b> <code>proline</code> va de 278 a 1,680 (rango 1,402; σ = 314.9), mientras que <code>nonflavanoid_phenols</code> varía solo entre 0.13 y 0.66 (rango 0.53; σ = 0.124).<br>
                    • <b>Consecuencia algorítmica:</b> K-Means, DBSCAN y PCA operan sobre distancias euclídeas o matrices de covarianza. Si se usaran los datos sin transformar, <code>proline</code> y <code>magnesium</code> dominarían el cálculo y las variables fenólicas casi no influirían. <b>Por eso el escalado es obligatorio y no opcional.</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 3. Distribuciones (3.3.4 y Figura 1)
    with pestanas[2]:
        st.subheader("3.3.4 Distribuciones fisicoquímicas y comportamientos enológicos")

        st.markdown("""
        <div style="background-color: #161922; border-left: 4px solid #a855f7; border-radius: 4px; padding: 12px 16px; margin-bottom: 16px;">
            <div style="font-weight: 600; color: #c084fc; font-size: 0.92rem; margin-bottom: 4px;">🔬 Los 3 Comportamientos Enológicos Identificados:</div>
            <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">
                <b>1. Forma acampanada y unimodal:</b> <code>alcohol</code>, <code>ash</code>, <code>alcalinity_of_ash</code> y <code>hue</code> presentan asimetría cercana a cero.<br>
                <b>2. Bimodalidad (indicio inicial de clústeres):</b> <code>flavanoids</code>, <code>total_phenols</code> y <code>od280/od315</code> sugieren dos concentraciones de valores (baja y alta madurez fenólica). Es un indicio visual de que las muestras podrían separarse en grupos agronómicos diferenciados.<br>
                <b>3. Cola larga hacia valores altos:</b> <code>malic_acid</code>, <code>magnesium</code>, <code>color_intensity</code> y <code>proline</code> (asimetría entre +0.77 y +1.10). En las tres primeras se debe a unas pocas muestras extremas; en <code>proline</code> la cola es gradual y no genera atípicos bajo Tukey.
            </div>
        </div>
        """, unsafe_allow_html=True)

        modo_vista = st.radio("Modo de visualización:", ["Variable individual a detalle", "Ver todas en cuadrícula (Figura 1 del Informe)"], horizontal=True)

        if modo_vista == "Variable individual a detalle":
            var_sel = st.selectbox("Selecciona la variable a inspeccionar:", vars_analisis)
            media = df_vars[var_sel].mean()
            mediana = df_vars[var_sel].median()
            desv = df_vars[var_sel].std()
            asimetria = df_vars[var_sel].skew()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Media", f"{media:.2f}")
            col_m2.metric("Mediana", f"{mediana:.2f}")
            col_m3.metric("Desv. Estándar (σ)", f"{desv:.2f}")
            col_m4.metric("Coef. Asimetría", f"{asimetria:+.2f}")

            fig, ax = plt.subplots(figsize=(8.5, 4.4))
            if HAS_SEABORN:
                sns.histplot(df_vars[var_sel], kde=True, color="#f43f5e", ax=ax, bins=20, edgecolor="#161922", alpha=0.75)
            else:
                ax.hist(df_vars[var_sel], bins=20, color="#f43f5e", edgecolor="#161922", alpha=0.75)

            # Anotar valores enteros de frecuencia sobre las barras
            if ax.patches:
                max_h = max([p.get_height() for p in ax.patches])
                ax.set_ylim(0, max_h * 1.16)
                for p in ax.patches:
                    h = p.get_height()
                    if h > 0:
                        ax.annotate(
                            f"{int(h)}",
                            (p.get_x() + p.get_width() / 2.0, h),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            fontweight="bold",
                            color="#fca5a5",
                            xytext=(0, 2),
                            textcoords="offset points",
                        )

            ax.axvline(media, color="#38bdf8", linestyle="--", linewidth=1.8, label=f"Media ({media:.2f})")
            ax.axvline(mediana, color="#34d399", linestyle="-.", linewidth=1.8, label=f"Mediana ({mediana:.2f})")
            ax.set_title(f"Distribución y Densidad (KDE) de {var_sel}", fontsize=11.5, fontweight="bold")
            ax.set_xlabel(var_sel)
            ax.set_ylabel("Frecuencia (conteo de muestras)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Figura 1. Histogramas y curvas de densidad (KDE) de las 13 variables fisicoquímicas")
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
            axes_flat = axes.flatten()
            for i, col in enumerate(vars_analisis):
                if HAS_SEABORN:
                    sns.histplot(df_vars[col], kde=True, ax=axes_flat[i], color="#f43f5e", bins=15, edgecolor="#161922", alpha=0.75)
                else:
                    axes_flat[i].hist(df_vars[col], bins=15, color="#f43f5e", edgecolor="#161922", alpha=0.75)
                axes_flat[i].set_title(col, fontsize=10, fontweight="bold")
                axes_flat[i].set_xlabel("")
                axes_flat[i].set_ylabel("")
            for j in range(len(vars_analisis), len(axes_flat)):
                fig.delaxes(axes_flat[j])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # 4. Valores Atípicos (3.3.5, Tabla 3B y Slide 7)
    with pestanas[3]:
        st.subheader("3.3.5 Detección de valores atípicos (Criterio de Tukey: 1.5 × IQR)")

        outliers_dict = {}
        for col in vars_analisis:
            q25 = df_vars[col].quantile(0.25)
            q75 = df_vars[col].quantile(0.75)
            iqr = q75 - q25
            lim_inf = q25 - 1.5 * iqr
            lim_sup = q75 + 1.5 * iqr
            conteo = int(((df_vars[col] < lim_inf) | (df_vars[col] > lim_sup)).sum())
            outliers_dict[col] = {
                "Límite Inferior": round(lim_inf, 3),
                "Límite Superior": round(lim_sup, 3),
                "N.º de atípicos": conteo,
                "%": round((conteo / len(df_vars)) * 100, 2),
            }
        df_outliers = pd.DataFrame(outliers_dict).T.sort_values(by="N.º de atípicos", ascending=False)
        df_outliers_mostrar = df_outliers.reset_index().rename(columns={"index": "Variable"})

        # Gráfico Horizontal de Atípicos (Slide 7 del PPTX)
        col_rank, col_exp = st.columns([1.1, 1.0], gap="large")
        with col_rank:
            st.markdown("**Ranking de Variables con Atípicos:**")
            vars_con_out = [c for c in df_outliers.index if df_outliers.loc[c, "N.º de atípicos"] > 0]
            conteo_con_out = [int(df_outliers.loc[c, "N.º de atípicos"]) for c in vars_con_out]
            vars_con_out.reverse()
            conteo_con_out.reverse()

            fig_out, ax_out = plt.subplots(figsize=(6.2, 3.2))
            barras = ax_out.barh(vars_con_out, conteo_con_out, color="#f43f5e", edgecolor="#ffffff", height=0.6, alpha=0.85)
            ax_out.set_xlim(0, 5)
            ax_out.set_xlabel("Número de observaciones atípicas")
            for bar in barras:
                w = bar.get_width()
                pct = (w / len(df_vars)) * 100
                ax_out.text(w + 0.12, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({pct:.2f}%)", va="center", fontsize=8.5, color="#e2e8f0", fontweight="bold")
            ax_out.grid(axis="x", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_out)
            plt.close(fig_out)

        with col_exp:
            st.markdown("""
            <div style="background-color: #161922; border-left: 4px solid #f43f5e; border-radius: 4px; padding: 12px 16px; margin-top: 15px;">
                <div style="font-weight: 600; color: #f43f5e; font-size: 0.92rem; margin-bottom: 4px;">📌 Interpretación de Atípicos:</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    • <b>Presencia acotada:</b> 7 de las 13 variables presentan atípicos según la regla de Tukey <code>[Q1 − 1.5·IQR, Q3 + 1.5·IQR]</code>, y en ninguna superan el <b>2.25 % de las muestras</b> (máximo 4 casos).<br>
                    • <b>Las 6 variables restantes:</b> <code>alcohol</code>, <code>total_phenols</code>, <code>flavanoids</code>, <code>nonflavanoid_phenols</code>, <code>od280/od315</code> y <code>proline</code> registran <b>0 atípicos</b>.<br>
                    • <b>Decisión de conservación:</b> Son valores extremos pero químicamente verosímiles (vinos con acidez o color muy intensos), no errores de medición; por eso <b>se conservan en el dataset</b>. Esta decisión es crucial para <b>DBSCAN</b>, que auditará si estas muestras constituyen ruido (-1).
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        modo_box = st.radio(
            "Modo de visualización de boxplots:",
            ["Variable individual a detalle", "Ver todas en cuadrícula (Figura 2 del Informe)"],
            horizontal=True,
            key="modo_box_sel",
        )

        col_config_outliers = {
            "Variable": st.column_config.TextColumn("Variable", width="medium"),
            "Límite Inferior": st.column_config.NumberColumn("Lím. Inferior", format="%.3f"),
            "Límite Superior": st.column_config.NumberColumn("Lím. Superior", format="%.3f"),
            "N.º de atípicos": st.column_config.NumberColumn("N.º Atípicos", format="%d"),
            "%": st.column_config.NumberColumn("% de Muestras", format="%.2f%%"),
        }

        if modo_box == "Variable individual a detalle":
            col_t1, col_t2 = st.columns([1.3, 1.1], gap="large")
            with col_t1:
                st.markdown("**Tabla 3B. Valores atípicos por variable según la regla de Tukey:**")
                st.dataframe(
                    df_outliers_mostrar,
                    use_container_width=True,
                    hide_index=True,
                    height=(len(df_outliers_mostrar) + 1) * 36 + 10,
                    column_config=col_config_outliers,
                )
            with col_t2:
                st.markdown("**Inspección visual del boxplot:**")
                var_box = st.selectbox("Selecciona variable para boxplot:", vars_analisis, key="box_sel")

                q25_v = df_vars[var_box].quantile(0.25)
                q75_v = df_vars[var_box].quantile(0.75)
                n_out_v = outliers_dict[var_box]["N.º de atípicos"]

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Q1 (25%)", f"{q25_v:.2f}")
                col_b2.metric("Mediana", f"{df_vars[var_box].median():.2f}")
                col_b3.metric("Atípicos", f"{n_out_v}")

                fig, ax = plt.subplots(figsize=(6.8, 3.8))
                if HAS_SEABORN:
                    sns.boxplot(
                        x=df_vars[var_box],
                        color="#f43f5e",
                        ax=ax,
                        flierprops={"marker": "o", "color": "#f87171", "markersize": 6.5, "markeredgecolor": "#ffffff"},
                        boxprops={"edgecolor": "#ffffff", "linewidth": 1},
                        whiskerprops={"color": "#e2e8f0", "linewidth": 1},
                        capprops={"color": "#e2e8f0", "linewidth": 1},
                        medianprops={"color": "#fbbf24", "linewidth": 2.2},
                    )
                else:
                    ax.boxplot(
                        df_vars[var_box],
                        vert=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#f43f5e", edgecolor="#ffffff"),
                        whiskerprops=dict(color="#e2e8f0"),
                        capprops=dict(color="#e2e8f0"),
                        medianprops=dict(color="#fbbf24", linewidth=2.2),
                        flierprops=dict(marker="o", markerfacecolor="#f87171", markersize=6.5, markeredgecolor="#ffffff"),
                    )

                # Anotar estadísticos clave en el boxplot individual
                med_val = float(df_vars[var_box].median())
                lim_i = outliers_dict[var_box]["Límite Inferior"]
                lim_s = outliers_dict[var_box]["Límite Superior"]

                ax.annotate(
                    f"Med: {med_val:.2f}",
                    xy=(med_val, 0),
                    xytext=(0, 28),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.2,
                    fontweight="bold",
                    color="#fbbf24",
                    arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#161922", edgecolor="#fbbf24", alpha=0.9)
                )

                ax.annotate(f"Q1: {q25_v:.2f}", xy=(q25_v, 0), xytext=(0, -25), textcoords="offset points", ha="center", fontsize=7.8, color="#cbd5e1")
                ax.annotate(f"Q3: {q75_v:.2f}", xy=(q75_v, 0), xytext=(0, -25), textcoords="offset points", ha="center", fontsize=7.8, color="#cbd5e1")

                # Anotar valores atípicos individuales
                out_vals = df_vars.loc[(df_vars[var_box] < lim_i) | (df_vars[var_box] > lim_s), var_box].values
                for o_v in out_vals:
                    ax.annotate(
                        f"{o_v:.2f}",
                        xy=(o_v, 0),
                        xytext=(0, 16),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.8,
                        fontweight="bold",
                        color="#f87171"
                    )

                ax.set_ylim(-0.55, 0.55)
                ax.set_title(f"Diagrama de Caja de {var_box}", fontsize=11, fontweight="bold")
                ax.set_xlabel(var_box)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.markdown("**Tabla 3B. Valores atípicos por variable según la regla de Tukey:**")
            st.dataframe(
                df_outliers_mostrar,
                use_container_width=True,
                hide_index=True,
                height=(len(df_outliers_mostrar) + 1) * 36 + 10,
                column_config=col_config_outliers,
            )
            st.markdown("---")
            st.caption("Figura 2. Diagramas de caja de las 13 variables fisicoquímicas")
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
            axes_flat = axes.flatten()
            for i, col in enumerate(vars_analisis):
                ax = axes_flat[i]
                if HAS_SEABORN:
                    sns.boxplot(
                        x=df_vars[col],
                        color="#f43f5e",
                        ax=ax,
                        flierprops={"marker": "o", "color": "#f87171", "markersize": 5, "markeredgecolor": "#ffffff"},
                        boxprops={"edgecolor": "#ffffff", "linewidth": 0.8},
                        whiskerprops={"color": "#e2e8f0", "linewidth": 0.8},
                        capprops={"color": "#e2e8f0", "linewidth": 0.8},
                        medianprops={"color": "#fbbf24", "linewidth": 1.8},
                    )
                else:
                    ax.boxplot(
                        df_vars[col],
                        vert=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#f43f5e", edgecolor="#ffffff"),
                        whiskerprops=dict(color="#e2e8f0"),
                        capprops=dict(color="#e2e8f0"),
                        medianprops=dict(color="#fbbf24", linewidth=1.8),
                        flierprops=dict(marker="o", markerfacecolor="#f87171", markersize=5, markeredgecolor="#ffffff"),
                    )
                n_out = outliers_dict[col]["N.º de atípicos"]
                titulo_color = "#f43f5e" if n_out > 0 else "#94a3b8"
                ax.set_title(f"{col} ({n_out} atípicos)", fontsize=10, fontweight="bold", color=titulo_color)
                ax.set_xlabel("")
            for j in range(len(vars_analisis), len(axes_flat)):
                fig.delaxes(axes_flat[j])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # 5. Correlación de Pearson (3.3.6, Tabla 3C y Figura 3)
    with pestanas[4]:
        st.subheader("3.3.6 Correlación lineal de Pearson y relaciones multivariadas")
        corr = df_vars.corr()

        vista_matriz = st.radio(
            "Modo de visualización de la matriz:",
            ["Matriz Completa (13×13, como en el informe)", "Triángulo Inferior (sin redundancia)"],
            horizontal=True,
            key="vista_corr_matriz",
        )

        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        mask = np.triu(np.ones_like(corr, dtype=bool)) if vista_matriz.startswith("Triángulo") else None

        if HAS_SEABORN:
            sns.heatmap(
                corr,
                mask=mask,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                center=0.0,
                annot=True,
                fmt=".2f",
                linewidths=0.6,
                linecolor="#161922",
                ax=ax,
                cbar_kws={"label": "Coeficiente de Correlación de Pearson (r)", "shrink": 0.82},
                annot_kws={"size": 8, "weight": "bold"},
            )
        else:
            cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            cbar = fig.colorbar(cax, ax=ax, shrink=0.82)
            cbar.set_label("Coeficiente de Correlación de Pearson (r)")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            for r in range(len(corr.columns)):
                for c in range(len(corr.columns)):
                    if mask is not None and mask[r, c]:
                        continue
                    val = corr.iloc[r, c]
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.6 else "white", fontsize=7.5, weight="bold")

        ax.set_title("Figura 3. Matriz de Correlación Lineal de Pearson (13 Variables Químicas)", fontsize=11.5, fontweight="bold", pad=12)
        st.pyplot(fig)
        plt.close(fig)

        # Tabla 3C exacta del Informe (Top 10 pares con |r|)
        st.markdown("**Tabla 3C. Los 10 pares de variables con mayor correlación lineal (|r|):**")
        st.caption("En total, 14 pares superan |r| > 0.50; aquí se muestran los 10 de mayor magnitud.")

        tabla_3c_datos = [
            {"N.º": 1, "Par de Variables": "total_phenols — flavanoids", "r": "+0.865", "Lectura Enológica": "Positiva muy fuerte (núcleo fenólico)"},
            {"N.º": 2, "Par de Variables": "flavanoids — od280/od315_of_diluted_wines", "r": "+0.787", "Lectura Enológica": "Positiva fuerte (absorbancia y flavonoides)"},
            {"N.º": 3, "Par de Variables": "total_phenols — od280/od315_of_diluted_wines", "r": "+0.700", "Lectura Enológica": "Positiva fuerte (indicador proteico/fenólico)"},
            {"N.º": 4, "Par de Variables": "flavanoids — proanthocyanins", "r": "+0.653", "Lectura Enológica": "Positiva moderada-fuerte (taninos condensados)"},
            {"N.º": 5, "Par de Variables": "alcohol — proline", "r": "+0.644", "Lectura Enológica": "Positiva moderada (madurez de uva y vigor)"},
            {"N.º": 6, "Par de Variables": "total_phenols — proanthocyanins", "r": "+0.612", "Lectura Enológica": "Positiva moderada (polifenoles totales)"},
            {"N.º": 7, "Par de Variables": "hue — od280/od315_of_diluted_wines", "r": "+0.565", "Lectura Enológica": "Positiva moderada (matiz y absorbancia)"},
            {"N.º": 8, "Par de Variables": "malic_acid — hue", "r": "−0.561", "Lectura Enológica": "Negativa moderada (acidez málica frente al matiz)"},
            {"N.º": 9, "Par de Variables": "alcohol — color_intensity", "r": "+0.546", "Lectura Enológica": "Positiva moderada (grado alcohólico y pigmentación)"},
            {"N.º": 10, "Par de Variables": "flavanoids — hue", "r": "+0.543", "Lectura Enológica": "Positiva moderada (flavonoides y tonalidad cromática)"},
        ]
        st.dataframe(pd.DataFrame(tabla_3c_datos), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background-color: #161922; border-left: 4px solid #38bdf8; border-radius: 4px; padding: 12px 16px; margin-top: 14px; margin-bottom: 18px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.92rem; margin-bottom: 4px;">📌 Interpretación y Consecuencias Metodológicas:</div>
            <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                Las variables fenólicas (<code>total_phenols</code>, <code>flavanoids</code>, <code>od280/od315</code> y <code>proanthocyanins</code>) forman un <b>bloque fuertemente correlacionado</b>. Esto tiene dos consecuencias críticas:<br>
                <b>(a) Redundancia de información:</b> Respalda plenamente reducir la dimensionalidad mediante <b>PCA</b> para sintetizar estas variables en componentes ortogonales.<br>
                <b>(b) Peso en distancias euclídeas:</b> Al calcular distancias euclídeas sin transformar, este bloque aporta más peso conjunto que variables independientes como <code>ash</code>.<br>
                <i>Nota importante:</i> Las correlaciones medidas describen asociación lineal estadística entre estas 178 muestras y no implican una relación de causalidad.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Gráficos de dispersión de relaciones destacadas")
        st.caption("Dispersión bivariada con línea de tendencia lineal ajustada.")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_d1, ax_d1 = plt.subplots(figsize=(6, 4))
            x1 = df_vars["total_phenols"]
            y1 = df_vars["flavanoids"]
            ax_d1.scatter(x1, y1, color="#f43f5e", edgecolors="#ffffff", linewidth=0.5, alpha=0.8, s=50)
            m1, b1 = np.polyfit(x1, y1, 1)
            x_vals1 = np.linspace(x1.min(), x1.max(), 100)
            ax_d1.plot(x_vals1, m1 * x_vals1 + b1, color="#38bdf8", linestyle="--", linewidth=2.2, label=f"Tendencia (r = +0.865)")
            ax_d1.set_title("total_phenols vs. flavanoids", fontsize=11, fontweight="bold")
            ax_d1.set_xlabel("total_phenols [g/L]")
            ax_d1.set_ylabel("flavanoids [g/L]")
            ax_d1.legend(loc="upper left", fontsize=8)
            ax_d1.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_d1)
            plt.close(fig_d1)

        with col_d2:
            fig_d2, ax_d2 = plt.subplots(figsize=(6, 4))
            x2 = df_vars["alcohol"]
            y2 = df_vars["proline"]
            ax_d2.scatter(x2, y2, color="#34d399", edgecolors="#ffffff", linewidth=0.5, alpha=0.8, s=50)
            m2, b2 = np.polyfit(x2, y2, 1)
            x_vals2 = np.linspace(x2.min(), x2.max(), 100)
            ax_d2.plot(x_vals2, m2 * x_vals2 + b2, color="#fbbf24", linestyle="--", linewidth=2.2, label=f"Tendencia (r = +0.644)")
            ax_d2.set_title("alcohol vs. proline", fontsize=11, fontweight="bold")
            ax_d2.set_xlabel("alcohol [% vol]")
            ax_d2.set_ylabel("proline [mg/L]")
            ax_d2.legend(loc="upper left", fontsize=8)
            ax_d2.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_d2)
            plt.close(fig_d2)

    # 6. Hallazgos Clave (Sección 3.3.7 del Informe)
    with pestanas[5]:
        st.subheader("3.3.7 Síntesis de Hallazgos del EDA (Conclusiones Formales)")
        st.markdown(
            "Los resultados exploratorios sustentan las decisiones metodológicas de las fases subsiguientes de preprocesamiento y modelado:"
        )

        col_h1, col_h2 = st.columns(2, gap="medium")
        with col_h1:
            st.markdown("""
            <div style="background-color: #161922; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 4px;">1. Calidad Íntegra de Datos</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                    178 filas, 13 variables fisicoquímicas numéricas, <b>0 valores nulos y 0 duplicados</b>. No se requiere imputación ni codificación categórica.
                </div>
            </div>
            <div style="background-color: #161922; border-left: 4px solid #f43f5e; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #f43f5e; font-size: 0.95rem; margin-bottom: 4px;">2. Heterogeneidad Drástica de Escala</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                    Las variables difieren en varios órdenes de magnitud (σ de <code>proline</code> = 314.9 frente a 0.124 en <code>nonflavanoid_phenols</code>, razón <b>> 2,000×</b>). El <b>escalado es indispensable</b> para evitar que dos variables monopolicen las distancias.
                </div>
            </div>
            <div style="background-color: #161922; border-left: 4px solid #a855f7; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #c084fc; font-size: 0.95rem; margin-bottom: 4px;">3. Indicio Inicial de Estructura de Clústeres</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                    <code>flavanoids</code>, <code>total_phenols</code> y <code>od280/od315</code> sugieren <b>dos concentraciones bien diferenciadas de valores</b> (baja y alta madurez fenólica), anticipando la viabilidad de separar las muestras en grupos enológicos.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_h2:
            st.markdown("""
            <div style="background-color: #161922; border-left: 4px solid #fbbf24; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #fbbf24; font-size: 0.95rem; margin-bottom: 4px;">4. Conservación Consciente de Atípicos</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                    Presentes en 7 variables, siempre por debajo del <b>2.25 %</b> (máx. 4 muestras por variable). Son vinos con perfiles intensos y biológicamente válidos; <b>se conservan en X</b> para que <b>DBSCAN</b> los audite como posible ruido.
                </div>
            </div>
            <div style="background-color: #161922; border-left: 4px solid #34d399; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #34d399; font-size: 0.95rem; margin-bottom: 4px;">5. Redundancia Colineal Justifica PCA</div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                    El bloque fenólico está fuertemente intercorrelacionado (<code>total_phenols</code> – <code>flavanoids</code>: r = +0.865). Esto respalda el uso de <b>PCA</b> para comprimir la dimensionalidad y proyectar la estructura geométrica en 2D y 3D.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------- Preprocesamiento
def pagina_preprocesamiento():
    st.header("⚙️ Preprocesamiento y Preparación de Datos")
    df, etiqueta = requerir_datos()
    if df is None:
        return

    st.markdown(
        """
        En esta etapa se prepara la **matriz de características $X$** para los algoritmos no supervisados
        (K-Means, DBSCAN y PCA). Se aplica la regla metodológica fundamental: **excluir cualquier etiqueta previa**
        y normalizar las escalas de las variables.
        """
    )

    # Paso 1: Exclusión de la etiqueta
    st.subheader("Paso 1: Aislamiento de la matriz X (Aprendizaje No Supervisado)")
    vars_x = [c for c in df.columns if c != etiqueta]
    X_raw = df[vars_x].copy()

    c1, c2 = st.columns(2)
    c1.metric("Dimensiones de X", f"{X_raw.shape[0]} filas × {X_raw.shape[1]} variables")
    c2.metric("Etiqueta excluida", f"'{etiqueta}' (no interviene en agrupamiento)" if etiqueta else "Ninguna")

    with st.expander("Ver variables predictoras que conforman X"):
        st.write(list(vars_x))

    # Paso 2: Comparación de decisiones de preparación
    st.subheader("Paso 2: Comparación de dos decisiones de preparación de datos")
    st.markdown(
        """
        Dado que el EDA evidenció tanto **escalas dispares** como **presencia de valores atípicos**, se comparan
        dos estrategias fundamentales de estandarización:
        
        * **Decisión A: StandardScaler (Z-Score)**
          $$z = \frac{x - \mu}{\sigma}$$
          Centra los datos con media $\mu = 0$ y desviación estándar $\sigma = 1$. Es el estándar óptimo para
          algoritmos basados en distancias euclídeas y varianza (K-Means y PCA).
          
        * **Decisión B: RobustScaler (Mediana e IQR)**
          $$z = \frac{x - \text{mediana}}{\text{IQR}}$$
          Centra usando la mediana y escala con el rango intercuartílico ($Q_3 - Q_1$). Es inmune a la influencia
          de los outliers identificados en el EDA.
        """
    )

    scaler_std = StandardScaler()
    X_std = pd.DataFrame(scaler_std.fit_transform(X_raw), columns=vars_x)

    scaler_rob = RobustScaler()
    X_rob = pd.DataFrame(scaler_rob.fit_transform(X_raw), columns=vars_x)

    tab_vis, tab_met = st.tabs(["Comparativa Visual de Distribución", "Comparativa Estadística"])

    with tab_vis:
        st.markdown("**Comparación de Boxplots: Original vs StandardScaler vs RobustScaler**")
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 11), sharex=False)
        
        flierprops = dict(marker="o", markerfacecolor="none", markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=5.5)

        if HAS_SEABORN:
            sns.boxplot(data=X_raw, ax=axes[0], orient="h", color="#38bdf8", linecolor="#ffffff", linewidth=1.2, flierprops=flierprops)
            sns.boxplot(data=X_std, ax=axes[1], orient="h", color="#34d399", linecolor="#ffffff", linewidth=1.2, flierprops=flierprops)
            sns.boxplot(data=X_rob, ax=axes[2], orient="h", color="#f43f5e", linecolor="#ffffff", linewidth=1.2, flierprops=flierprops)
        else:
            props_box_raw = dict(facecolor="#38bdf8", edgecolor="#ffffff", linewidth=1.2)
            props_box_std = dict(facecolor="#34d399", edgecolor="#ffffff", linewidth=1.2)
            props_box_rob = dict(facecolor="#f43f5e", edgecolor="#ffffff", linewidth=1.2)
            props_line = dict(color="#ffffff", linewidth=1.2)

            axes[0].boxplot([X_raw[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True,
                            boxprops=props_box_raw, whiskerprops=props_line, capprops=props_line,
                            medianprops=props_line, flierprops=flierprops)
            axes[1].boxplot([X_std[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True,
                            boxprops=props_box_std, whiskerprops=props_line, capprops=props_line,
                            medianprops=props_line, flierprops=flierprops)
            axes[2].boxplot([X_rob[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True,
                            boxprops=props_box_rob, whiskerprops=props_line, capprops=props_line,
                            medianprops=props_line, flierprops=flierprops)

        for ax_i in axes:
            ax_i.grid(True, axis="x", linestyle="--", alpha=0.35, color="#333846")

        axes[0].set_title("1. Datos Originales (Rangos dispares: Proline domina la escala)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Escala original")
        axes[1].set_title("2. Decisión A: StandardScaler (Media = 0, Varianza = 1)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Desviaciones estándar (Z-score)")
        axes[2].set_title("3. Decisión B: RobustScaler (Centrado en Mediana = 0, Escala IQR)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Unidades IQR")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_met:
        st.markdown("**Resumen comparativo de propiedades:**")
        comp_df = pd.DataFrame({
            "Métrica": ["Media promedio", "Desv. Est. promedio", "Sensibilidad a outliers", "Apto para K-Means/PCA", "Apto para DBSCAN"],
            "Datos Originales": ["Muy heterogénea (1.93 a 746)", "Muy heterogénea (0.12 a 314)", "N/A", "❌ No (Sesgo por escala)", "❌ No (Distancias distorsionadas)"],
            "Decisión A: StandardScaler": ["0.000 (Exacto)", "1.000 (Exacto)", "Moderada / Alta", "✅ Óptimo (Supuestos cumplidos)", "✅ Adecuado"],
            "Decisión B: RobustScaler": ["Varía según asimetría", "Varía según IQR", "Baja (Robusto)", "⚠️ Menor compatibilidad con PCA", "✅ Muy adecuado para densidades"],
        })
        st.table(comp_df)

    # Paso 3: Selección de la matriz final
    st.subheader("Paso 3: Matriz X final para los modelos")
    decision_elegida = st.radio(
        "Selecciona la transformación a guardar en memoria para K-Means, DBSCAN y PCA:",
        [
            "Decisión A: StandardScaler (Recomendada para K-Means y PCA)",
            "Decisión B: RobustScaler (Alternativa robusta a outliers)",
        ],
        index=0,
    )

    if decision_elegida.startswith("Decisión A"):
        X_final = X_std
        scaler_final = scaler_std
        nombre_scaler = "StandardScaler"
    else:
        X_final = X_rob
        scaler_final = scaler_rob
        nombre_scaler = "RobustScaler"

    st.session_state["X_final"] = X_final
    st.session_state["X_raw"] = X_raw
    st.session_state["scaler_obj"] = scaler_final
    st.session_state["scaler_nombre"] = nombre_scaler
    st.session_state["vars_x"] = vars_x

    st.success(f"✅ Matriz final establecida con **{nombre_scaler}** ({X_final.shape[0]} observaciones, {X_final.shape[1]} variables) guardada en `st.session_state`.")

    with st.expander("Vista previa de los primeros 5 registros de la matriz X estandarizada"):
        st.dataframe(X_final.head(), use_container_width=True)

    # Paso 4: Parámetros reproducibles
    st.subheader("Paso 4: Parámetros reproducibles (3.5)")
    st.markdown(
        """
        Para garantizar la reproducibilidad exacta en cualquier entorno, se definen los siguientes hiperparámetros
        y configuraciones de ejecución:
        """
    )
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Semilla aleatoria (SEED)", "42")
    col_p2.metric("Escalador activo", nombre_scaler)
    col_p3.metric("Variables continuas fijadas", len(vars_x))

    st.code(
        f"""# Configuración de preprocesamiento reproducible
import numpy as np
import pandas as pd
from sklearn.preprocessing import {nombre_scaler}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Separación de matriz de características
features = {list(vars_x)}
X = df[features].copy()

# 2. Instanciación y ajuste del transformador
scaler = {nombre_scaler}()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
""",
        language="python",
    )


def pagina_kmeans():
    st.header("🎯 Agrupamiento K-Means")
    X_final, X_raw, vars_x, scaler = matriz_para_clustering()
    if X_final is None:
        return
    X = X_final.to_numpy()
    n = X.shape[0]
    k_max = min(K_MAX, n - 1)
    if k_max < K_MIN:
        st.error(f"Con {n} registros no es posible evaluar K entre {K_MIN} y {K_MAX}.")
        return

    st.markdown(
        f"K-Means con inicialización **k-means++**, `n_init={N_INIT}` y `random_state={RANDOM_STATE}`. "
        f"Se evalúa K = {K_MIN} a {k_max} con inertia (método del codo), Silhouette, Davies-Bouldin "
        "y Calinski-Harabasz."
    )

    tabla_k = evaluar_rango_k(X, K_MIN, k_max)
    k_sugerido = int(tabla_k.loc[tabla_k["Silhouette"].idxmax(), "K"]) if "Silhouette" in tabla_k else K_MIN

    tab_sel, tab_mod, tab_perf, tab_vis = st.tabs([
        "1. Selección de K", "2. Modelo y métricas", "3. Perfil de clústeres", "4. Visualización",
    ])

    # 1. Selección de K
    with tab_sel:
        st.subheader("Curva del codo y métricas internas por K")
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        series = [
            ("Inertia", "Inercia (Método del Codo)\n[Menor es mejor ↓]", "#f43f5e"),
            ("Silhouette", "Coeficiente Silhouette\n[Mayor es mejor ↑]", "#34d399"),
            ("Davies-Bouldin", "Índice Davies-Bouldin\n[Menor es mejor ↓]", "#60a5fa"),
            ("Calinski-Harabasz", "Índice Calinski-Harabasz\n[Mayor es mejor ↑]", "#fbbf24"),
        ]
        for ax, (col, titulo, color) in zip(axes.flatten(), series):
            ax.plot(tabla_k["K"], tabla_k[col], marker="o", color=color, linewidth=2.4, markersize=6.5, label=col)
            ax.fill_between(tabla_k["K"], tabla_k[col], alpha=0.15, color=color)
            ax.axvline(k_sugerido, color="#f87171", linestyle="--", alpha=0.85, linewidth=1.5, label=f"K óptimo = {k_sugerido}")
            val_opt = tabla_k.loc[tabla_k["K"] == k_sugerido, col].values[0]
            ax.plot(k_sugerido, val_opt, marker="o", markersize=11, color=color, markeredgecolor="#ffffff", markeredgewidth=2.2)

            # Etiquetar cada punto con su valor exacto como en el informe
            y_vals = tabla_k[col].values
            y_min, y_max = y_vals.min(), y_vals.max()
            margin = (y_max - y_min) * 0.20 if y_max != y_min else 1.0
            ax.set_ylim(y_min - margin * 0.25, y_max + margin * 0.55)

            for x_val, y_val in zip(tabla_k["K"], y_vals):
                if col == "Inertia":
                    txt = f"{int(round(y_val))}"
                elif col == "Calinski-Harabasz":
                    txt = f"{y_val:.1f}"
                else:
                    txt = f"{y_val:.3f}"
                es_optimo = (x_val == k_sugerido)
                peso = "bold" if es_optimo else "normal"
                color_txt = "#ffffff" if es_optimo else "#cbd5e1"
                ax.annotate(
                    txt,
                    xy=(x_val, y_val),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight=peso,
                    color=color_txt,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#161922", edgecolor=color if es_optimo else "none", alpha=0.85)
                )

            ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=8)
            ax.set_xlabel("Número de clústeres (K)", fontsize=9)
            ax.set_xticks(tabla_k["K"])
            ax.grid(True, linestyle="--", alpha=0.35, color="#2c3345")
            ax.legend(fontsize=8, loc="best", framealpha=0.7)
        plt.tight_layout(pad=2.2)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('''
        <div style="background-color: #161922; border-left: 4px solid #f43f5e; border-radius: 4px; padding: 12px 16px; margin-top: 10px; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #f43f5e; font-size: 0.92rem; margin-bottom: 4px;">📌 Interpretación de la Inercia y el Codo:</div>
            <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">
                La <b>inertia cae 380.8 unidades al pasar de K = 2 a K = 3</b> (de 1658.8 a 1277.9), pero solo 102.5 al pasar a K = 4 (1175.4); 
                desde ahí cada clúster adicional reduce apenas entre 44 y 66 unidades. <b>El codo se ubica con claridad en K = 3</b>.<br>
                Para el laboratorio esto significa que <b>tres perfiles químicos explican la mayor parte de la heterogeneidad de las muestras</b>; 
                agregar más grupos apenas reduce la dispersión interna y fragmenta perfiles sin aportar diferenciación enológica nueva.
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.dataframe(
            tabla_k.set_index("K").style.format({
                "Inertia": "{:.1f}", "Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}", "Calinski-Harabasz": "{:.1f}",
            }).highlight_max(subset=["Silhouette", "Calinski-Harabasz"], props="background-color: #1b5e20; color: #ffffff; font-weight: bold;")
              .highlight_min(subset=["Davies-Bouldin"], props="background-color: #1b5e20; color: #ffffff; font-weight: bold;"),
            width="stretch",
            height=(len(tabla_k) + 1) * 36 + 10,
        )
        mejores = {
            "Silhouette": int(tabla_k.loc[tabla_k["Silhouette"].idxmax(), "K"]),
            "Davies-Bouldin": int(tabla_k.loc[tabla_k["Davies-Bouldin"].idxmin(), "K"]),
            "Calinski-Harabasz": int(tabla_k.loc[tabla_k["Calinski-Harabasz"].idxmax(), "K"]),
        }
        coinciden = len(set(mejores.values())) == 1
        st.info(
            f"**K sugerido = {k_sugerido}** (máxima Silhouette). Mejor K por métrica: "
            + ", ".join(f"{m} → {k}" for m, k in mejores.items())
            + (". Las tres métricas coinciden." if coinciden else ". Las métricas no coinciden: revisar el codo y la interpretabilidad.")
        )

    # 2. Modelo
    with tab_mod:
        st.subheader("Ajuste del modelo")
        k = st.slider("Número de clústeres K", K_MIN, k_max, min(k_sugerido, k_max), key="k_kmeans",
                      help="Cada cambio vuelve a entrenar K-Means y recalcula las métricas.")
        if k >= n:
            st.error(f"⚠️ El número de clústeres K ({k}) no puede ser mayor o igual al número de muestras ({n}).")
            return
        labels, centros, inertia, n_iter = ajustar_kmeans(X, k)
        met, motivo = metricas_internas(X, labels)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inertia", f"{inertia:.1f}")
        c2.metric("Silhouette", f"{met['Silhouette']:.3f}" if met else "No calculable")
        c3.metric("Davies-Bouldin", f"{met['Davies-Bouldin']:.3f}" if met else "No calculable")
        c4.metric("Calinski-Harabasz", f"{met['Calinski-Harabasz']:.1f}" if met else "No calculable")
        if motivo:
            st.warning(f"Métricas no calculables: {motivo}.")
        st.caption(f"Convergió en {n_iter} iteraciones. Cobertura: 100 % de los puntos (K-Means no produce ruido).")

        tamanos = pd.Series(labels).value_counts().sort_index()

        # Construir columnas de la tabla con la 4ta columna de características químicas (Tabla 6 del Informe)
        filas_tabla = []
        nombres_k3 = {
            0: "0 · Ligeros, poco color y baja prolina",
            1: "1 · Bajos en fenoles, color intenso y más ácidos",
            2: "2 · Alto alcohol, prolina y fenoles",
        }
        caract_k3 = {
            0: "alcohol 12.25 (12.25) · color_intensity 2.97 (2.90) · proline 510 (495) · magnesium 92.7 (88.0) · hue 1.06 (1.04) · flavanoids 2.05 (2.03)",
            1: "flavanoids 0.82 (0.70) · od280/od315 1.70 (1.68) · hue 0.69 (0.67) · color_intensity 7.23 (7.30) · malic_acid 3.31 (3.27) · alcohol 13.13 (13.11)",
            2: "proline 1100 (1072.5) · alcohol 13.68 (13.74) · flavanoids 3.00 (2.98) · total_phenols 2.85 (2.82) · magnesium 108.0 (105.5) · alcalinity_of_ash 17.46 (17.00)",
        }

        # Medias y medianas por clúster en unidades originales
        medias_grp = X_raw.groupby(labels).mean()
        medianas_grp = X_raw.groupby(labels).median()
        global_mean = X_raw.mean()
        global_std = X_raw.std()

        for g in tamanos.index:
            tam_g = int(tamanos[g])
            pct_g = (tam_g / n) * 100
            if k == 3 and g in nombres_k3:
                nombre_g = nombres_k3[g]
                caract_g = caract_k3[g]
            else:
                nombre_g = f"Clúster {g}"
                # Calcular dinámicamente las 5 variables con mayor desviación para cualquier K
                z_diff = ((medias_grp.loc[g] - global_mean) / global_std).abs().sort_values(ascending=False)
                top_vars = z_diff.head(5).index
                caract_g = " · ".join([f"{v} {medias_grp.loc[g, v]:.2f} ({medianas_grp.loc[g, v]:.2f})" for v in top_vars])

            filas_tabla.append({
                "Clúster": nombre_g,
                "Tamaño": tam_g,
                "%": f"{pct_g:.1f} %",
                "Características químicas: media (mediana) en unidades originales": caract_g,
            })

        tam_df = pd.DataFrame(filas_tabla)

        titulo_tabla = "**Tabla 6. Perfil y tamaño de los clústeres de K-Means (K = 3):**" if k == 3 else f"**Perfil y tamaño de los clústeres de K-Means (K = {k}):**"
        st.markdown(titulo_tabla)

        col_t, col_g = st.columns([1.75, 1.0], gap="large")
        with col_t:
            st.dataframe(
                tam_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Clúster": st.column_config.TextColumn("Clúster", width="medium"),
                    "Tamaño": st.column_config.NumberColumn("Tamaño", format="%d", width="small"),
                    "%": st.column_config.TextColumn("%", width="small"),
                    "Características químicas: media (mediana) en unidades originales": st.column_config.TextColumn(
                        "Características químicas: media (mediana) en unidades originales",
                        width="large"
                    ),
                },
                height=(len(tam_df) + 1) * 42 + 20
            )
        with col_g:
            fig, ax = plt.subplots(figsize=(5.5, 3.1))
            etiquetas_barras = [f"C{g}" for g in tamanos.index]
            bars = ax.bar(etiquetas_barras, tamanos.values, color=colores_clusters(k), edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
            ax.set_ylim(0, max(tamanos.values) * 1.25)
            for bar, val in zip(bars, tamanos.values):
                pct = (val / n) * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#ffffff")
            ax.set_xlabel("Clúster")
            ax.set_ylabel("N.º de muestras")
            ax.set_title("Tamaño y Proporción (K-Means)", fontsize=10.5, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.session_state["labels_kmeans"] = labels
        st.session_state["kmeans_params"] = {"K": k, "n_init": N_INIT, "random_state": RANDOM_STATE, "init": "k-means++"}
        st.session_state["kmeans_metricas"] = {"Inertia": inertia, **(met or {})}
        st.session_state["kmeans_centros"] = centros

    # 3. Perfil
    with tab_perf:
        st.subheader(f"Perfil químico de los {k} clústeres")
        centros_orig = pd.DataFrame(scaler.inverse_transform(centros), columns=vars_x)
        centros_orig.index.name = "Clúster"
        st.markdown("**Centroides en unidades originales** (media del clúster):")
        st.dataframe(centros_orig.T.round(2), width="stretch")

        medianas = X_raw.groupby(labels).median()
        medianas.index.name = "Clúster"
        with st.expander("Medianas por clúster (unidades originales)"):
            st.dataframe(medianas.T.round(2), width="stretch")

        st.markdown("**Centroides estandarizados** (cuántas desviaciones se alejan de la media global):")
        centros_z = pd.DataFrame(centros, columns=vars_x)
        centros_z.index = [f"C{i}" for i in range(k)]
        fig, ax = plt.subplots(figsize=(12, 0.6 * k + 2.5))
        dibujar_heatmap(ax, centros_z)
        ax.set_title("Centroides en escala estandarizada", fontsize=11, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        rasgos = []
        for i in range(k):
            fila = centros_z.iloc[i]
            altas = fila.nlargest(2)
            bajas = fila.nsmallest(2)
            rasgos.append(
                f"- **Clúster {i}** ({int((labels == i).sum())} muestras): alto en "
                + ", ".join(f"`{v}` ({z:+.2f})" for v, z in altas.items())
                + "; bajo en " + ", ".join(f"`{v}` ({z:+.2f})" for v, z in bajas.items()) + "."
            )
        st.markdown("**Rasgos que distinguen a cada clúster:**\n" + "\n".join(rasgos))

    # 4. Visualización
    with tab_vis:
        st.subheader("Clústeres en proyección 2D")
        coords, var_exp, pca = proyeccion_2d(X)
        centros_2d = pca.transform(centros)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for i, color in enumerate(colores_clusters(k)):
            m = labels == i
            ax.scatter(coords[m, 0], coords[m, 1], color=color, edgecolors="#ffffff", linewidth=0.5,
                       alpha=0.85, s=50, label=f"Clúster {i} (n={m.sum()})")
        ax.scatter(centros_2d[:, 0], centros_2d[:, 1], marker="X", s=220, color="#ffffff", edgecolors="#f43f5e", linewidth=1.6, label="Centroides")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} varianza)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} varianza)")
        ax.set_title(f"K-Means (K = {k}) proyectado en PC1-PC2", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "La proyección PCA se usa solo para dibujar; K-Means se entrenó con todas las variables. "
            "El análisis completo de PCA está en la sección **PCA**."
        )
        tabla_vs_etiqueta(labels, "Clúster K-Means")


# ------------------------------------------------------------------- DBSCAN
def pagina_dbscan():
    st.header("🔬 Agrupamiento DBSCAN")
    X_final, X_raw, vars_x, scaler = matriz_para_clustering()
    if X_final is None:
        return
    X = X_final.to_numpy()
    n = X.shape[0]

    st.markdown(
        "DBSCAN agrupa por densidad con distancia **euclídea**: un punto es núcleo si tiene al menos "
        "`min_samples` puntos (incluido él mismo) a distancia ≤ `eps`. Los puntos no alcanzables "
        "desde un núcleo se marcan como **ruido (-1)**."
    )

    c_eps, c_ms = st.columns(2)
    eps = c_eps.slider("eps (radio de vecindad)", 0.10, 6.00, EPS_DEFECTO, 0.05, key="eps_dbscan",
                       help="Cada cambio vuelve a ejecutar DBSCAN.")
    min_samples = c_ms.slider("min_samples", 2, min(20, n), min(MIN_SAMPLES_DEFECTO, n), key="ms_dbscan")

    labels = ajustar_dbscan(X, eps, min_samples)
    n_ruido = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if n_ruido else 0)
    met, motivo = metricas_internas(X, labels)

    st.session_state["labels_dbscan"] = labels
    st.session_state["dbscan_params"] = {"eps": eps, "min_samples": min_samples, "metric": "euclidean"}
    st.session_state["dbscan_metricas"] = {
        "N.º clústeres": n_clusters, "N.º ruido": n_ruido, "% ruido": 100 * n_ruido / n, **(met or {}),
    }

    tab_kd, tab_mod, tab_sens, tab_vis = st.tabs([
        "1. Gráfico k-distance", "2. Modelo y métricas", "3. Sensibilidad", "4. Visualización y perfil",
    ])

    # 1. k-distance
    with tab_kd:
        st.subheader("Gráfico k-distance para elegir eps")
        kdist = distancias_k(X, min_samples)
        pct_nucleo = 100 * (kdist <= eps).mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(np.arange(1, n + 1), kdist, color="#38bdf8", linewidth=2.4, label="Distancia k-vecino")
        ax.fill_between(np.arange(1, n + 1), kdist, alpha=0.15, color="#38bdf8")
        ax.axhline(eps, color="#f43f5e", linestyle="--", linewidth=1.8, label=f"eps actual = {eps:.2f}")
        
        # Marcar punto de cruce con anotación numérica
        idx_cruce = int(np.searchsorted(kdist, eps))
        if 0 <= idx_cruce < n:
            val_cruce = kdist[idx_cruce]
            ax.plot(idx_cruce + 1, val_cruce, marker="o", markersize=10, color="#f43f5e", markeredgecolor="#ffffff", markeredgewidth=1.8)
            ax.annotate(
                f"Punto corte: muestra #{idx_cruce + 1} (d = {val_cruce:.2f})\n{pct_nucleo:.1f}% puntos núcleo (≤ eps)",
                xy=(idx_cruce + 1, val_cruce),
                xytext=(-50, 22),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#161922", edgecolor="#f43f5e", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=1.3)
            )

        ax.set_xlabel("Puntos ordenados por distancia")
        ax.set_ylabel(f"Distancia al {min_samples - 1}.º vecino más cercano")
        ax.set_title(f"k-distance (min_samples = {min_samples})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")
        st.pyplot(fig)
        plt.close(fig)
        p50, p75, p90 = np.percentile(kdist, [50, 75, 90])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Percentil 50", f"{p50:.2f}")
        c2.metric("Percentil 75", f"{p75:.2f}")
        c3.metric("Percentil 90", f"{p90:.2f}")
        c4.metric("Puntos núcleo con este eps", f"{pct_nucleo:.1f} %")
        st.caption(
            "El eps adecuado suele ubicarse donde la curva empieza a subir con fuerza (codo). "
            "Los puntos por encima de la línea roja no son núcleo: serán borde o ruido."
        )

    # 2. Modelo
    with tab_mod:
        st.subheader(f"Resultado con eps = {eps:.2f} y min_samples = {min_samples}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N.º clústeres (sin -1)", n_clusters)
        c2.metric("N.º puntos de ruido", n_ruido)
        c3.metric("% ruido", f"{100 * n_ruido / n:.1f} %")
        c4.metric("Cobertura", f"{100 * (n - n_ruido) / n:.1f} %")

        c5, c6, c7 = st.columns(3)
        c5.metric("Silhouette (sin ruido)", f"{met['Silhouette']:.3f}" if met else "No calculable")
        c6.metric("Davies-Bouldin (sin ruido)", f"{met['Davies-Bouldin']:.3f}" if met else "No calculable")
        c7.metric("Calinski-Harabasz (sin ruido)", f"{met['Calinski-Harabasz']:.1f}" if met else "No calculable")
        st.caption("Las métricas se calculan solo sobre los puntos asignados a un clúster (se excluye -1).")

        if n_clusters == 0:
            st.warning("Todos los puntos quedaron como ruido: eps es demasiado pequeño o min_samples demasiado alto.")
        elif n_clusters == 1:
            st.warning("Solo se formó un clúster: eps es demasiado grande para separar regiones densas; "
                       f"métricas no calculables ({motivo}).")
        elif motivo:
            st.warning(f"Métricas no calculables: {motivo}.")
        if 100 * n_ruido / n > 30:
            st.info("Más del 30 % de los puntos es ruido: conviene aumentar eps o reducir min_samples.")

        tamanos = pd.Series(labels).value_counts().sort_index()
        tam_df = pd.DataFrame({
            "Grupo": ["Ruido (-1)" if g == -1 else f"Clúster {g}" for g in tamanos.index],
            "Tamaño": tamanos.values,
            "%": (100 * tamanos.values / n).round(1),
        })
        col_tdb, col_gdb = st.columns([1, 2])
        col_tdb.dataframe(tam_df, hide_index=True, width="stretch")

        fig, ax = plt.subplots(figsize=(6, 3.2))
        colores_barras_db = ["#f87171" if g == -1 else colores_clusters(n_clusters)[g] for g in tamanos.index]
        etiquetas_x = ["Ruido (-1)" if g == -1 else f"C{g}" for g in tamanos.index]
        bars_db = ax.bar(etiquetas_x, tamanos.values, color=colores_barras_db, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
        ax.set_ylim(0, max(tamanos.values) * 1.22)
        for bar, val in zip(bars_db, tamanos.values):
            pct = (val / n) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#ffffff")
        ax.set_xlabel("Grupo / Clúster")
        ax.set_ylabel("N.º de muestras")
        ax.set_title("Tamaño y Proporción de Grupos DBSCAN", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        col_gdb.pyplot(fig)
        plt.close(fig)

    # 3. Sensibilidad
    with tab_sens:
        st.subheader("Sensibilidad a eps y min_samples")
        sens = sensibilidad_dbscan(X, tuple(GRILLA_EPS), tuple(GRILLA_MIN_SAMPLES))
        col_a, col_b = st.columns(2)
        for col, valor, titulo, cmap, fmt in [
            (col_a, "N.º clústeres", "N.º de clústeres (sin -1)", "Blues", "d"),
            (col_b, "% ruido", "% de ruido", "Reds", ".1f"),
        ]:
            pivote = sens.pivot(index="min_samples", columns="eps", values=valor)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            dibujar_heatmap(ax, pivote, cmap=cmap, fmt=fmt, centro=None)
            ax.set_title(titulo, fontsize=11, fontweight="bold")
            plt.tight_layout()
            col.pyplot(fig)
            plt.close(fig)
        st.dataframe(sens, hide_index=True, width="stretch")
        st.caption(
            "Pequeños cambios de eps alteran mucho el resultado: con eps bajo casi todo es ruido y con eps "
            "alto todas las muestras se funden en un solo clúster."
        )

    # 4. Visualización
    with tab_vis:
        st.subheader("Puntos DBSCAN con ruido marcado (proyección 2D)")
        coords, var_exp, _ = proyeccion_2d(X)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ruido = labels == -1
        ax.scatter(coords[ruido, 0], coords[ruido, 1], marker="x", color="#f87171", s=50, linewidth=1.5,
                   label=f"Ruido -1 (n={ruido.sum()})")
        for i, color in enumerate(colores_clusters(n_clusters)):
            m = labels == i
            ax.scatter(coords[m, 0], coords[m, 1], color=color, edgecolors="#ffffff", linewidth=0.5,
                       alpha=0.85, s=50, label=f"Clúster {i} (n={m.sum()})")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} varianza)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} varianza)")
        ax.set_title(f"DBSCAN (eps = {eps:.2f}, min_samples = {min_samples})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("La proyección PCA se usa solo para dibujar; DBSCAN se ejecutó con todas las variables.")

        st.markdown("**Media de cada grupo en unidades originales:**")
        perfil = X_raw.groupby(labels).mean().round(2)
        perfil.index = ["Ruido (-1)" if g == -1 else f"Clúster {g}" for g in perfil.index]
        st.dataframe(perfil.T, width="stretch")
        if n_ruido:
            z_max_ruido = np.abs(X[ruido]).max(axis=1).mean()
            z_max_resto = np.abs(X[~ruido]).max(axis=1).mean() if (~ruido).any() else float("nan")
            st.caption(
                f"Valor |z| máximo promedio: ruido {z_max_ruido:.2f} vs. resto {z_max_resto:.2f}. "
                "Un valor mayor indica que el ruido reúne muestras con valores extremos en alguna variable."
            )
        tabla_vs_etiqueta(labels, "Clúster DBSCAN")

# ------------------------------------------------------------------- PCA
def pagina_pca():
    st.header("📉 Análisis de Componentes Principales (PCA)")
    X_final, X_raw, vars_x, scaler = matriz_para_clustering()
    if X_final is None:
        return
    X = X_final.to_numpy()

    st.markdown(
        "El PCA transforma las 13 variables correlacionadas en componentes no correlacionados "
        "ordenados por la varianza que explican. Se usa para evaluar la redundancia y visualizar "
        "los clústeres en 2D."
    )

    # Ajustar PCA con todas las componentes para el Scree Plot
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X)
    varianza_exp = pca_full.explained_variance_ratio_
    varianza_acum = np.cumsum(varianza_exp)

    tab_var, tab_cargas, tab_biplot, tab_vis = st.tabs(["1. Varianza Explicada", "2. Cargas (Loadings)", "3. Biplot (Vectores)", "4. Visualización PC1-PC2"])

    # 1. Varianza Explicada (Scree Plot)
    with tab_var:
        st.subheader("Scree Plot y Varianza Acumulada")
        n_comp = st.slider("Número de componentes a mostrar en el gráfico", 2, 13, 10)
        
        fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
        v_ind_pct = varianza_exp[:n_comp] * 100
        v_acum_pct = varianza_acum[:n_comp] * 100

        bars = ax1.bar(range(1, n_comp + 1), v_ind_pct, alpha=0.75, color="#f43f5e", edgecolor="#ffffff", linewidth=0.5, label="Varianza Individual (%)")
        ax1.set_ylim(0, max(v_ind_pct) * 1.25)
        for bar, val in zip(bars, v_ind_pct):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#fca5a5")
        
        ax1.set_xlabel("Componente Principal", fontsize=9.5)
        ax1.set_ylabel("Proporción de Varianza Explicada (%)", color="#f43f5e", fontsize=9.5)
        ax1.tick_params(axis='y', labelcolor="#f43f5e")
        ax1.set_xticks(range(1, n_comp + 1))
        ax1.set_xticklabels([f"PC{i}" for i in range(1, n_comp + 1)], fontsize=8.5)
        
        ax2 = ax1.twinx()
        ax2.plot(range(1, n_comp + 1), v_acum_pct, marker="o", color="#38bdf8", linewidth=2.4, markersize=6.5, label="Varianza Acumulada (%)")
        ax2.set_ylim(0, 110)
        ax2.axhline(80.0, color="#fbbf24", linestyle="--", alpha=0.75, linewidth=1.4, label="Umbral 80% (5 CPs)")
        
        for x_i, y_i in zip(range(1, n_comp + 1), v_acum_pct):
            es_clave = (x_i in [2, 3, 5])
            ax2.annotate(
                f"{y_i:.1f}%",
                xy=(x_i, y_i),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold" if es_clave else "normal",
                color="#ffffff" if es_clave else "#38bdf8",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#161922", edgecolor="#38bdf8" if es_clave else "none", alpha=0.8)
            )

        ax2.set_ylabel("Varianza Acumulada (%)", color="#38bdf8", fontsize=9.5)
        ax2.tick_params(axis='y', labelcolor="#38bdf8")
        fig.suptitle("Figura 10. Scree Plot: Varianza Explicada Individual y Acumulada", fontweight="bold", color="#f0f2f6", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"**PC1** explica el **{varianza_exp[0]:.2%}** de la varianza total. "
            f"**PC1 y PC2** juntos explican el **{varianza_acum[1]:.2%}**. "
            f"Para retener al menos el 80% de la información se necesitan **{np.argmax(varianza_acum >= 0.80) + 1}** componentes."
        )

    # 2. Cargas (Loadings)
    with tab_cargas:
        st.subheader("Contribución de las variables a los componentes")
        st.markdown("Las cargas (loadings) indican cuánto aporta cada variable original a cada componente. Valores altos en valor absoluto indican mayor influencia.")
        
        n_comp_cargas = st.slider("Componentes a visualizar", 2, 5, 2, key="slider_cargas")
        cargas_df = pd.DataFrame(
            pca_full.components_[:n_comp_cargas].T,
            columns=[f"PC{i+1}" for i in range(n_comp_cargas)],
            index=vars_x
        )
        
        fig, ax = plt.subplots(figsize=(8, 5))
        dibujar_heatmap(ax, cargas_df, cmap="vlag", fmt=".2f", centro=0)
        ax.set_title("Cargas de las variables en los Componentes Principales", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            "**Interpretación:** Variables como `flavanoids`, `total_phenols` y `od280/od315` "
            "tienen cargas altas en PC1, lo que sugiere que este componente separa los vinos "
            "principalmente por su **perfil fenólico**."
        )

    # 3. Biplot (Vectores de Cargas + Muestras)
    with tab_biplot:
        st.subheader("Biplot de Componentes Principales (PC1 vs PC2)")
        st.markdown(
            "El **Biplot** proyecta simultáneamente las observaciones (muestras de vino) y las variables "
            "originales como vectores direccionados. La longitud y dirección de cada vector indican su peso y correlación con PC1 y PC2."
        )
        scale_arrows = st.slider("Factor de escala para vectores de carga", 2.0, 7.0, 4.8, step=0.2, key="slider_biplot_scale")
        
        coords = pca_full.transform(X)[:, :2]
        components = pca_full.components_[:2]
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(coords[:, 0], coords[:, 1], c="#94a3b8", alpha=0.65, edgecolors="#ffffff", linewidths=0.4, s=45, label="Muestras de vino")
        
        for i, col in enumerate(vars_x):
            x_arr = components[0, i] * scale_arrows
            y_arr = components[1, i] * scale_arrows
            ax.annotate("", xy=(x_arr, y_arr), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=1.8, mutation_scale=14))
            ax.text(x_arr * 1.05, y_arr * 1.05, col, fontsize=8, fontweight="bold", color="#f1f5f9",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e2230", edgecolor="#3b4256", alpha=0.9))
                    
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(f"PC1 ({varianza_exp[0]:.1%} varianza)")
        ax.set_ylabel(f"PC2 ({varianza_exp[1]:.1%} varianza)")
        ax.set_title("Biplot: Muestras y Vectores de Variables Fisicoquímicas", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left")
        st.pyplot(fig)
        plt.close(fig)
        
        st.caption("Interpretación: Variables como flavanoids y total_phenols apuntan hacia la derecha (PC1), mientras que color_intensity y alcohol apuntan hacia arriba (PC2).")

    # 4. Visualización PC1-PC2
    with tab_vis:
        st.subheader("Proyección de los datos en PC1 vs PC2")
        coords = pca_full.transform(X)[:, :2]
        
        # Verificar si hay clústeres en session_state
        labels_km = st.session_state.get("labels_kmeans", None)
        labels_db = st.session_state.get("labels_dbscan", None)
        
        color_por = st.radio("Colorear puntos por:", ["Ninguno", "K-Means", "DBSCAN"], horizontal=True)
        
        color_array = None
        if color_por == "K-Means" and labels_km is not None:
            color_array = labels_km
        elif color_por == "DBSCAN" and labels_db is not None:
            color_array = labels_db
            # Reemplazar -1 por un color específico para ruido
            color_array = np.where(color_array == -1, 99, color_array) 
        elif color_por != "Ninguno":
            st.warning(f"Primero debes ejecutar el modelo {color_por} en su sección correspondiente.")

        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        if color_por == "K-Means" and labels_km is not None:
            k_val = len(np.unique(labels_km))
            paleta = colores_clusters(k_val)
            for i in range(k_val):
                mask = (labels_km == i)
                n_count = int(mask.sum())
                ax.scatter(coords[mask, 0], coords[mask, 1], color=paleta[i], edgecolors="#ffffff",
                           linewidth=0.5, alpha=0.85, s=50, label=f"Clúster {i} (n={n_count})")
            
            # Obtener y proyectar centroides a 2D en PC1-PC2
            centros_raw = st.session_state.get("kmeans_centros", None)
            if centros_raw is None or len(centros_raw) != k_val:
                centros_raw = np.array([X[labels_km == i].mean(axis=0) for i in range(k_val)])
            centros_2d = pca_full.transform(centros_raw)[:, :2]
            ax.scatter(centros_2d[:, 0], centros_2d[:, 1], marker="X", s=220, color="#ffffff",
                       edgecolors="#f43f5e", linewidth=1.8, label="Centroides", zorder=5)
            for i_c, (cx, cy) in enumerate(centros_2d):
                ax.annotate(f"C{i_c}", (cx, cy), xytext=(7, 7), textcoords="offset points",
                            fontsize=9, fontweight="bold", color="#ffffff",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#161922", edgecolor="#f43f5e", alpha=0.9))
            ax.legend(loc="best", fontsize=8.5, framealpha=0.75)

        elif color_por == "DBSCAN" and labels_db is not None:
            ruido = (labels_db == -1)
            n_ruido = int(ruido.sum())
            if n_ruido > 0:
                ax.scatter(coords[ruido, 0], coords[ruido, 1], marker="x", color="#f87171", s=50,
                           linewidth=1.5, label=f"Ruido -1 (n={n_ruido})")
            unique_clusters = sorted([c for c in set(labels_db) if c != -1])
            paleta = colores_clusters(len(unique_clusters)) if unique_clusters else []
            for idx_c, c_id in enumerate(unique_clusters):
                mask = (labels_db == c_id)
                n_count = int(mask.sum())
                ax.scatter(coords[mask, 0], coords[mask, 1], color=paleta[idx_c], edgecolors="#ffffff",
                           linewidth=0.5, alpha=0.85, s=50, label=f"Clúster {c_id} (n={n_count})")
            ax.legend(loc="best", fontsize=8.5, framealpha=0.75)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="#38bdf8", edgecolors="#ffffff", linewidth=0.5, alpha=0.75, s=50, label="Muestras de vino (N=178)")
            ax.legend(loc="best", fontsize=8.5, framealpha=0.75)

        ax.set_xlabel(f"PC1 ({varianza_exp[0]:.1%} varianza)")
        ax.set_ylabel(f"PC2 ({varianza_exp[1]:.1%} varianza)")
        ax.set_title("Muestras de vino proyectadas en el plano PC1-PC2", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)
        
        st.caption("Nota: PCA no agrupa, solo transforma el espacio. Si los puntos de un mismo clúster aparecen juntos, significa que las variables originales que los separan están bien representadas en estos dos componentes.")


# -------------------------------------------------------------- Comparación
def pagina_comparacion():
    st.header("⚖️ Comparación Metodológica: K-Means vs. DBSCAN")
    
    labels_km = st.session_state.get("labels_kmeans", None)
    labels_db = st.session_state.get("labels_dbscan", None)
    
    if labels_km is None or labels_db is None:
        st.warning("⚠️ Debes ejecutar primero los modelos en las secciones **K-Means** y **DBSCAN** para poder compararlos.")
        return
        
    X_final, X_raw, vars_x, scaler = matriz_para_clustering()
    if X_final is None or X_raw is None:
        return
    X = X_final.to_numpy()
    
    # Calcular métricas K-Means
    met_km, _ = metricas_internas(X, labels_km)
    n_clusters_km = len(np.unique(labels_km))
    
    # Calcular métricas DBSCAN
    met_db, motivo_db = metricas_internas(X, labels_db)
    n_ruido_db = int((labels_db == -1).sum())
    n_clusters_db = len(set(labels_db)) - (1 if n_ruido_db > 0 else 0)
    cobertura_db = 100 * (len(labels_db) - n_ruido_db) / len(labels_db)
    
    st.markdown(
        "Esta sección contrasta los resultados obtenidos por ambos algoritmos sobre la misma "
        "matriz X estandarizada. **Importante:** Las métricas de DBSCAN se calculan excluyendo los puntos de ruido (-1)."
    )
    
    # Tabla Comparativa
    st.subheader("Tabla Comparativa de Métricas")
    data = {
        "Indicador": [
            "N.º de grupos", 
            "Proporción de ruido", 
            "Cobertura de puntos", 
            "Inertia", 
            "Silhouette", 
            "Davies-Bouldin", 
            "Calinski-Harabasz"
        ],
        "K-Means": [
            n_clusters_km,
            "0% (No aplica)",
            "100%",
            f"{st.session_state['kmeans_metricas']['Inertia']:.1f}",
            f"{met_km['Silhouette']:.3f}" if met_km else "No calculable",
            f"{met_km['Davies-Bouldin']:.3f}" if met_km else "No calculable",
            f"{met_km['Calinski-Harabasz']:.1f}" if met_km else "No calculable"
        ],
        "DBSCAN": [
            n_clusters_db,
            f"{100 * n_ruido_db / len(labels_db):.1f}% ({n_ruido_db} muestras)",
            f"{cobertura_db:.1f}%",
            "No aplica",
            f"{met_db['Silhouette']:.3f}" if met_db else "No calculable",
            f"{met_db['Davies-Bouldin']:.3f}" if met_db else "No calculable",
            f"{met_db['Calinski-Harabasz']:.1f}" if met_db else "No calculable"
        ]
    }
    st.table(pd.DataFrame(data))

    # Visualización comparativa de Silhouette
    st.subheader("Comparación Visual de la Calidad del Agrupamiento")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribución de Clústeres (K-Means)**")
        tam_km = pd.Series(labels_km).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars_km = ax.bar(tam_km.index.astype(str), tam_km.values, color=colores_clusters(n_clusters_km), edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
        ax.set_ylim(0, max(tam_km.values) * 1.22)
        for bar, val in zip(bars_km, tam_km.values):
            pct = (val / len(labels_km)) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#ffffff")
        ax.set_xlabel("Clúster")
        ax.set_ylabel("N.º de muestras")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.markdown("**Distribución de Clústeres (DBSCAN)**")
        tam_db = pd.Series(labels_db).value_counts().sort_index()
        labels_x = ["Ruido (-1)" if i == -1 else f"C{i}" for i in tam_db.index]
        colors_db = ["#f87171" if i == -1 else colores_clusters(n_clusters_db)[i] for i in tam_db.index]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars_db = ax.bar(labels_x, tam_db.values, color=colors_db, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
        ax.set_ylim(0, max(tam_db.values) * 1.22)
        for bar, val in zip(bars_db, tam_db.values):
            pct = (val / len(labels_db)) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#ffffff")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("N.º de muestras")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Análisis Crítico
    st.subheader("Análisis Crítico y Conclusiones de la Comparación")
    st.markdown(
        f"""
        * **Cobertura vs. Ruido:** K-Means asigna todas las 178 muestras a un grupo, asumiendo que todas pertenecen a un perfil definido. DBSCAN identificó un **{100 * n_ruido_db / len(labels_db):.1f}%** de ruido ({n_ruido_db} muestras), las cuales coinciden en gran medida con los valores atípicos detectados en el EDA. Para el laboratorio, estas muestras de ruido son candidatas a **revisión individual o análisis de calidad**.
        * **Calidad de los Grupos (Silhouette):** DBSCAN obtiene una Silhouette ligeramente superior ({met_db['Silhouette']:.3f} vs {met_km['Silhouette']:.3f}), lo que indica que los grupos que *sí* encontró están muy bien definidos. Sin embargo, esta métrica se calcula excluyendo el 23.6% de los datos, por lo que no es directamente comparable con el 100% de K-Means.
        * **Utilidad de Negocio:** K-Means es más útil para **segmentar todos los lotes** de vino en 3 perfiles comerciales (ya que todos caen en un grupo). DBSCAN es más útil como **herramienta de control de calidad** para detectar anomalías (ruido) y encontrar los dos perfiles más extremos (vinos muy fenólicos vs. vinos muy ácidos).
        * **Sensibilidad:** K-Means (con K=3) es estable y converge rápido. DBSCAN es extremadamente sensible a `eps`; un cambio mínimo (de 2.3 a 2.5) fusiona todo en un solo clúster.
        """
    )
    
    # Descarga de etiquetas
    st.subheader("📥 Exportación de Etiquetas Asignadas")
    st.markdown(
        "Descarga el conjunto de datos con los identificadores originales de muestra y las etiquetas "
        "asignadas por **K-Means** y **DBSCAN** (los puntos marcados como `-1` corresponden al ruido detectado)."
    )
    
    df_descarga = X_raw.copy()
    df_descarga.insert(0, "ID_Muestra", range(1, len(df_descarga) + 1))
    df_descarga["Cluster_KMeans"] = labels_km
    df_descarga["Cluster_DBSCAN"] = labels_db
    
    st.dataframe(df_descarga.head(10), use_container_width=True)
    
    csv_datos = df_descarga.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar asignación de clústeres (CSV)",
        data=csv_datos,
        file_name="vinos_agrupados_kmeans_dbscan.csv",
        mime="text/csv",
        help="Archivo CSV con 178 filas, variables fisicoquímicas y clústeres de K-Means y DBSCAN."
    )



# ---------------------------------------------------------------- Conclusiones
def pagina_conclusiones():
    st.header("🏁 Conclusiones, Limitaciones y Recomendaciones")
    st.caption("Cierre y entrega integral del proyecto · Sistemas Inteligentes y Machine Learning · Grupo 03")

    st.markdown(
        """
        En este proyecto se abordó el descubrimiento no supervisado de patrones fisicoquímicos en muestras de vino,
        evaluando el desempeño de **K-Means**, **DBSCAN** y **PCA** sobre 13 variables cuantitativas estandarizadas.
        A continuación se consolidan los hallazgos, las 5 conclusiones cuantitativas, las 3 limitaciones técnicas
        y las 3 recomendaciones estratégicas para la toma de decisiones enológicas e industriales.
        """
    )

    pestanas = st.tabs([
        "1. Síntesis y Perfiles",
        "2. Las 5 Conclusiones",
        "3. Limitaciones (3)",
        "4. Recomendaciones (3)",
        "5. Ficha Ejecutiva",
    ])

    # 1. Síntesis y Perfiles
    with pestanas[0]:
        st.subheader("Resumen Comparativo de Modelos y Perfiles Enológicos")
        c1, c2, c3 = st.columns(3)
        c1.metric("K-Means Óptimo", "K = 3", "Silhouette = 0.285 (100% Cobertura)")
        c2.metric("DBSCAN Óptimo", "eps = 2.30, min_samples = 5", "2 grupos + 23.6% Ruido (Silhouette = 0.331*)")
        c3.metric("PCA Reducción", "3 Componentes", "66.5% Varianza Acumulada")
        st.caption("*La métrica Silhouette de DBSCAN se calcula únicamente sobre las muestras densas asignadas, excluyendo el ruido.")

        st.markdown("### Perfiles Químicos Descubiertos en los Vinos:")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(
                """
                **🍇 Perfil A: Alta Estructura y Graduación (Clúster 0)**
                * **Graduación alcohólica:** Elevada (~13.7 % vol).
                * **Compuestos fenólicos:** Ricos en prolina (~1,115) y flavonoides (~2.98 g/L).
                * **Color:** Alta intensidad cromática (~5.5).
                * **Interpretación comercial:** Vinos con alta graduación y contenido fenólico elevado.
                """
            )
        with col_p2:
            st.markdown(
                """
                **🥂 Perfil B: Vinos Ligeros y Ácidos (Clúster 1)**
                * **Graduación alcohólica:** Moderada a baja (~12.2 % vol).
                * **Acidez y cenizas:** Alta alcalinidad de cenizas (~21.4) y prolina baja (~519).
                * **Color:** Matices claros (hue alto ~1.06).
                * **Interpretación comercial:** Vinos con menor graduación alcohólica y mayor alcalinidad de cenizas.
                """
            )
        with col_p3:
            st.markdown(
                """
                **🍷 Perfil C: Vinos Astringentes / Oxidativos (Clúster 2)**
                * **Graduación alcohólica:** Media-alta (~13.1 % vol).
                * **Compuestos fenólicos:** Muy baja concentración de flavonoides (~0.78 g/L) y baja relación od280/od315 (~1.68).
                * **Acidez:** Alta concentración de ácido málico (~3.3 g/L).
                * **Interpretación comercial:** Vinos con bajo contenido de flavonoides y acidez málica elevada.
                """
            )

    # 2. Las 5 Conclusiones
    with pestanas[1]:
        st.subheader("Las 5 Conclusiones Centrales del Proyecto")
        st.markdown(
            r"""
            1. **Descubrimiento de una estructura natural tripartita en los vinos:**
               El análisis del método del codo (Inertia = 1,277.9) y el coeficiente de Silhouette (máximo global en $K = 3$ con 0.285) demostraron cuantitativamente que las 13 variables fisicoquímicas contienen información suficiente para segregar las 178 muestras en **tres perfiles químicos diferenciados**, mostrando un solapamiento inter-clúster moderado (Silhouette = 0.285) donde 172 de las 178 muestras (96.6%) corresponden al cultivar dominante de cada clúster.

            2. **Efectividad de la reducción dimensional mediante PCA:**
               La descomposición en componentes principales reveló una alta redundancia en el espacio original: los dos primeros componentes explican el **55.4%** de la varianza total (PC1 con 36.2% dominado por flavonoides y fenoles totales; PC2 con 19.2% dominado por color y alcohol), mientras que con **3 componentes se alcanza el 66.5%**, permitiendo una representación visual fidedigna sin pérdida crítica de información.

            3. **Aislamiento eficaz de anomalías fisicoquímicas con DBSCAN:**
               DBSCAN demostró ser una potente herramienta de auditoría de calidad al clasificar **42 muestras (23.6%) como ruido (-1)**. El análisis estadístico evidenció que estas muestras poseen un valor $|z|$ extremo superior al promedio del dataset, reuniendo vinos con acidez málica desmedida o concentraciones inusuales de magnesio correspondientes a los atípicos detectados en el EDA.

            4. **Impacto determinante de la estandarización multivariada:**
               El preprocesamiento mediante `StandardScaler` ($\mu = 0, \sigma = 1$) resultó mandatorio. Variables como `proline` (con rango original de 1,402) habrían dominado desproporcionadamente la suma de distancias euclídeas de no haber sido escaladas. La normalización otorgó un peso equitativo a marcadores traza como los flavonoides, garantizando una segmentación basada en la identidad química global del vino.

            5. **Sinergia metodológica para la industria vitivinícola:**
               Ningún algoritmo reemplaza al otro: **K-Means** es el método óptimo para la **segmentación comercial total** (cobertura del 100% de los lotes para catálogo de venta), mientras que **DBSCAN** es insustituible en el **control de calidad de laboratorio** (para descartar lotes defectuosos o vendimias anómalas antes del envasado).
            """
        )

    # 3. Limitaciones
    with pestanas[2]:
        st.subheader("Limitaciones Técnicas del Estudio (3)")
        st.markdown(
            """
            * ⚠️ **Limitación 1: Supuestos de esfericidad e isometría en K-Means:**
              K-Means impone particiones convexas y esféricas en el espacio multidimensional. Aquellas muestras ubicadas en los límites difusos entre cultivares son forzadas a un centroide, ignorando la incertidumbre de frontera.

            * ⚠️ **Limitación 2: Sensibilidad de DBSCAN ante densidades variables en alta dimensión:**
              En un espacio de 13 dimensiones, el fenómeno de la "maldición de la dimensionalidad" dispersa las distancias relativas. Por ello, una variación leve en el parámetro `eps` (por ejemplo, de 2.30 a 2.50) provoca que los clústeres se fusionen masivamente o que el ruido aumente drásticamente, exigiendo una calibración empírica minuciosa del gráfico k-distance.

            * ⚠️ **Limitación 3: Tamaño y representatividad geográfica de la muestra:**
              El conjunto de datos consta de 178 registros correspondientes a tres variedades cultivadas en una región específica de Italia en 1992. Los umbrales numéricos obtenidos no pueden extrapolarse directamente a vinos de climas tropicales o del nuevo mundo sin un reentrenamiento con lotes contemporáneos locales.
            """
        )

    # 4. Recomendaciones
    with pestanas[3]:
        st.subheader("Recomendaciones Metodológicas y Prácticas (3)")
        st.markdown(
            """
            * 💡 **Recomendación 1: Implementación de un pipeline de dos fases en planta enológica:**
              Se recomienda adoptar un flujo operativo en cascada: primero, someter los lotes entrantes a **DBSCAN** para identificar y separar anomalías y defectos; segundo, clasificar los lotes aprobados mediante **K-Means** ($K=3$) para asignar su línea comercial (Crianza, Joven, Varietal).

            * 💡 **Recomendación 2: Ejecución de clustering sobre el espacio latente de PCA:**
              Para optimizar la velocidad y estabilidad de algoritmos basados en densidad, se sugiere proyectar previamente las 13 variables sobre las primeras 3 componentes principales (66.5% varianza) antes de aplicar DBSCAN, eliminando el ruido residual de las variables poco informativas.

            * 💡 **Recomendación 3: Incorporación de variables organolépticas y climáticas:**
              Complementar los parámetros de laboratorio con variables sensoriales obtenidas por paneles de cata profesionales (astringencia percibida, persistencia aromática) y metadatos de añada (horas de sol, régimen de lluvias) para enriquecer la interpretabilidad comercial de los clústeres.
            """
        )

    # 5. Ficha Ejecutiva
    with pestanas[4]:
        st.subheader("Ficha de Entrega y Resumen Ejecutivo")
        ficha_texto = """# INFORME FINAL DE PROYECTO
Curso: Sistemas Inteligentes y Machine Learning
Grupo: 03 · UPN
Tema: Descubrimiento de patrones fisicoquímicos en vinos (K-Means, DBSCAN y PCA)

1. MODELO K-MEANS:
   - Configuración óptima: K = 3 (K-Means++, n_init=10, random_state=42)
   - Métricas: Inertia = 1277.9 | Silhouette = 0.285 | Davies-Bouldin = 1.389 | Calinski-Harabasz = 70.9
   - Cobertura: 100% de las 178 muestras asignadas a 3 perfiles comerciales.

2. MODELO DBSCAN:
   - Configuración óptima: eps = 2.30, min_samples = 5 (distancia euclídea)
   - Resultados: 2 clústeres densos + 42 muestras de ruido (23.6%)
   - Silhouette (sin ruido): 0.331 | Davies-Bouldin: 1.182 | Calinski-Harabasz = 70.5

3. MODELO PCA:
   - Dimensiones reducidas: 13 variables -> 3 Componentes Principales
   - Varianza explicada: PC1 (36.2%) + PC2 (19.2%) + PC3 (11.1%) = 66.5% acumulada
"""
        st.code(ficha_texto, language="markdown")
        st.download_button(
            label="📥 Descargar Ficha Ejecutiva de Conclusiones (.txt)",
            data=ficha_texto,
            file_name="resumen_ejecutivo_conclusiones.txt",
            mime="text/plain",
        )


def renderizar_encabezado_stepper(seccion_actual: str):
    """Renderiza el Stepper del Pipeline en la parte superior con barra compacta de 9 tarjetas en fila (Opción B)."""
    idx = SECCIONES.index(seccion_actual) if seccion_actual in SECCIONES else 0
    total = len(SECCIONES)
    pct = int(((idx + 1) / total) * 100)

    st.markdown(
        f"""
        <div style="margin-bottom: 8px; padding: 8px 14px; background: rgba(244, 63, 94, 0.05); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #f43f5e;">
                    🧪 Pipeline de Machine Learning · Fase {idx + 1} de {total}: <span style="color: #ffffff;">{seccion_actual}</span>
                </span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #f0f2f6;">
                    {pct}% Completado
                </span>
            </div>
            <div style="width: 100%; height: 5px; background: rgba(255, 255, 255, 0.08); border-radius: 10px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #f43f5e 0%, #be123c 50%, #38bdf8 100%); border-radius: 10px; transition: width 0.35s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pasos = [
        ("Inicio", "01", "🏠", "Inicio"),
        ("Datos", "02", "📁", "Datos"),
        ("EDA", "03", "📊", "EDA"),
        ("Preprocesamiento", "04", "⚖️", "Escala"),
        ("K-Means", "05", "🔴", "K-Means"),
        ("DBSCAN", "06", "🔵", "DBSCAN"),
        ("PCA", "07", "📉", "PCA"),
        ("Comparación", "08", "⚖️", "Comparar"),
        ("Conclusiones", "09", "📑", "Conclusión"),
    ]

    cols = st.columns(9, gap="small")
    for col, (key, num, icon, label) in zip(cols, pasos):
        with col:
            p_idx = SECCIONES.index(key) if key in SECCIONES else 0
            is_active = (key == seccion_actual)
            is_done = (p_idx < idx)

            if is_active:
                btn_text = f"● {num} {icon} {label}"
                btn_type = "primary"
            elif is_done:
                btn_text = f"✓ {num} {icon} {label}"
                btn_type = "secondary"
            else:
                btn_text = f"{num} {icon} {label}"
                btn_type = "secondary"

            if st.button(btn_text, key=f"bar_step_{key}", use_container_width=True, type=btn_type):
                st.session_state["seccion_activa"] = key
                st.rerun()


def renderizar_botones_flotantes_laterales(seccion_actual: str):
    """Renderiza botones de navegación flotantes discretos (< y >) en los laterales."""
    idx = SECCIONES.index(seccion_actual) if seccion_actual in SECCIONES else 0
    total = len(SECCIONES)

    col_fl, col_fr = st.columns([1, 1])

    with col_fl:
        if idx > 0:
            anterior = SECCIONES[idx - 1]
            if st.button("❮", key="btn_flotante_prev", help=f"Anterior: {anterior}"):
                st.session_state["seccion_activa"] = anterior
                st.rerun()

    with col_fr:
        if idx < total - 1:
            siguiente = SECCIONES[idx + 1]
            if st.button("❯", key="btn_flotante_next", help=f"Siguiente: {siguiente}"):
                st.session_state["seccion_activa"] = siguiente
                st.rerun()
        else:
            if st.button("↺", key="btn_flotante_next", help="Reiniciar al Inicio"):
                st.session_state["seccion_activa"] = "Inicio"
                st.rerun()




# ------------------------------------------------------------------ Main
PAGINAS = {
    "Inicio": pagina_inicio,
    "Datos": pagina_datos,
    "EDA": pagina_eda,
    "Preprocesamiento": pagina_preprocesamiento,
    "K-Means": pagina_kmeans,
    "DBSCAN": pagina_dbscan,
    "PCA": pagina_pca,             
    "Comparación": pagina_comparacion, 
    "Conclusiones": pagina_conclusiones,
}

if "seccion_activa" not in st.session_state:
    st.session_state["seccion_activa"] = "Inicio"

# Barra lateral sincronizada automáticamente (cumple la rúbrica docente)
st.sidebar.title("🍷 Navegación")
idx_sidebar = SECCIONES.index(st.session_state["seccion_activa"]) if st.session_state["seccion_activa"] in SECCIONES else 0
seccion_sidebar = st.sidebar.radio(
    "Sección",
    SECCIONES,
    index=idx_sidebar,
    label_visibility="collapsed",
)
if seccion_sidebar != st.session_state["seccion_activa"]:
    st.session_state["seccion_activa"] = seccion_sidebar
    st.rerun()

# 1. Stepper del Pipeline en la parte superior (Cuadrícula 3x3 por etapas)
renderizar_encabezado_stepper(st.session_state["seccion_activa"])
st.write("")

# 2. Botones de navegación flotantes laterales (a mitad de pantalla)
renderizar_botones_flotantes_laterales(st.session_state["seccion_activa"])

# 3. Renderizar la sección activa (sin alterar ningún gráfico ni tabla)
PAGINAS.get(st.session_state["seccion_activa"], lambda: en_construccion(st.session_state["seccion_activa"]))()

# Pie de página sutil
st.markdown(
    """
    <div style="text-align: center; margin-top: 45px; padding: 18px 0; color: #64748b; font-size: 0.76rem; border-top: 1px solid rgba(255, 255, 255, 0.06);">
        🍷 Descubrimiento de Patrones Fisicoquímicos en Vinos · Sistemas Inteligentes y Machine Learning · Grupo 03
    </div>
    """,
    unsafe_allow_html=True,
)

