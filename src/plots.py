"""
Módulo de visualización gráfica (Matplotlib y Seaborn).
Implementa proyecciones 2D, diagramas de calor (heatmaps), paletas de colores y tablas de contraste.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@st.cache_data
def proyeccion_2d(X: np.ndarray, random_state: int = 42):
    """PCA a 2 componentes SOLO para visualizar (el agrupamiento usa todas las variables)."""
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X), pca.explained_variance_ratio_, pca


def colores_clusters(n: int):
    """Paleta de colores categórica para los clústeres."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def dibujar_heatmap(ax, datos: pd.DataFrame, cmap="vlag", fmt=".2f", centro=0.0):
    """Dibuja un mapa de calor usando Seaborn si está disponible, o imshow de Matplotlib como respaldo."""
    if HAS_SEABORN:
        sns.heatmap(
            datos,
            cmap=cmap,
            center=centro,
            annot=True,
            fmt=fmt,
            linewidths=0.6,
            linecolor="#161922",
            ax=ax,
            annot_kws={"size": 8, "weight": "bold"},
            cbar_kws={"shrink": 0.8},
        )
    else:
        im = ax.imshow(datos.values, cmap="coolwarm", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(datos.shape[1]))
        ax.set_xticklabels(datos.columns, rotation=0)
        ax.set_yticks(range(datos.shape[0]))
        ax.set_yticklabels(datos.index)
        for r in range(datos.shape[0]):
            for c in range(datos.shape[1]):
                ax.text(c, r, format(datos.iat[r, c], fmt), ha="center", va="center", fontsize=8, color="#ffffff")


def tabla_vs_etiqueta(labels: np.ndarray, nombre: str, df: pd.DataFrame = None, etiqueta: str = None):
    """Tabla cruzada clúster vs. etiqueta externa (solo referencia posterior)."""
    if df is None:
        df = st.session_state.get("df")
    if etiqueta is None:
        etiqueta = st.session_state.get("etiqueta")

    if not etiqueta or df is None or etiqueta not in df.columns or len(df) != len(labels):
        return
    with st.expander("Contraste posterior con la etiqueta externa (no se usó para agrupar)"):
        st.caption(
            "La etiqueta original no interviene en el entrenamiento ni en la elección de parámetros; "
            "esta tabla solo sirve para verificar si los grupos tienen sentido."
        )
        cruce = pd.crosstab(pd.Series(labels, name=nombre), df[etiqueta].reset_index(drop=True).rename(etiqueta))
        st.dataframe(cruce, width="stretch")
