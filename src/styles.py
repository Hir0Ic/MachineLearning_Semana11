"""
Estilos visuales, temas y componentes de interfaz para la aplicación Streamlit.
"""
import streamlit as st
import matplotlib.pyplot as plt

COLOR_RUIDO = "#9e9e9e"


def aplicar_tema_graficos_dark():
    """Configura Matplotlib y Seaborn con un tema oscuro elegante y nativo para Streamlit."""
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


def aplicar_estilos_ui():
    """Aplica diseño moderno, enológico y de alta gama a la interfaz Streamlit."""
    aplicar_tema_graficos_dark()
    
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }
        
        h1 {
            color: #f43f5e !important;
        }

        /* Tarjetas de métricas modernas (KPIs) */
        [data-testid="stMetric"] {
            background: rgba(22, 25, 34, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        [data-testid="stMetric"]:hover {
            border-color: rgba(244, 63, 94, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(244, 63, 94, 0.15);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            color: #f43f5e !important;
        }

        /* Pestañas (Tabs) refinadas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            padding: 0 16px;
            font-weight: 600;
            font-size: 0.88rem;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #f43f5e;
            background: rgba(244, 63, 94, 0.08);
        }

        .stTabs [aria-selected="true"] {
            color: #f43f5e !important;
            background: rgba(244, 63, 94, 0.12) !important;
            border-bottom: 2px solid #f43f5e !important;
        }

        /* Botones primarios y de descarga */
        .stButton > button, .stDownloadButton > button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #f43f5e !important;
            color: white !important;
            border-color: #f43f5e !important;
            box-shadow: 0 4px 14px rgba(244, 63, 94, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        /* Cajas de alerta */
        .stAlert {
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }

        /* Barra lateral (Sidebar) */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Tablas y DataFrames */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Contenedores de gráficos integrados como tarjetas */
        [data-testid="stImage"] > img {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            background-color: #161922;
        }

        /* Botones flotantes laterales discretos y semi-invisibles (< y >) */
        div.st-key-btn_flotante_prev {
            position: fixed !important;
            left: 8px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 999999 !important;
            width: 38px !important;
            height: 38px !important;
        }

        div.st-key-btn_flotante_prev button {
            width: 38px !important;
            height: 38px !important;
            min-width: 38px !important;
            max-width: 38px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(22, 25, 34, 0.25) !important;
            backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: rgba(255, 255, 255, 0.45) !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            opacity: 0.35 !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        div.st-key-btn_flotante_prev button:hover {
            opacity: 1 !important;
            background: rgba(244, 63, 94, 0.85) !important;
            color: #ffffff !important;
            border-color: #f43f5e !important;
            transform: scale(1.15) !important;
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.6) !important;
        }

        div.st-key-btn_flotante_next {
            position: fixed !important;
            right: 10px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 999999 !important;
            width: 38px !important;
            height: 38px !important;
        }

        div.st-key-btn_flotante_next button {
            width: 38px !important;
            height: 38px !important;
            min-width: 38px !important;
            max-width: 38px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(22, 25, 34, 0.25) !important;
            backdrop-filter: blur(6px) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: rgba(255, 255, 255, 0.45) !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            opacity: 0.35 !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        div.st-key-btn_flotante_next button:hover {
            opacity: 1 !important;
            background: rgba(244, 63, 94, 0.85) !important;
            color: #ffffff !important;
            border-color: #f43f5e !important;
            transform: scale(1.15) !important;
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.6) !important;
        }

        /* Barra de 9 tarjetas en fila del Stepper (Opción B) */
        [data-testid="column"] div[class*="st-key-bar_step_"] button {
            padding: 8px 3px !important;
            border-radius: 9px !important;
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
