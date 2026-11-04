"""
Módulo de algoritmos de Machine Learning no supervisado y métricas de validación.
Implementa K-Means, DBSCAN y cálculo de métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import streamlit as st

RANDOM_STATE = 42
N_INIT = 10
K_MIN, K_MAX = 2, 10
EPS_DEFECTO, MIN_SAMPLES_DEFECTO = 2.3, 5
GRILLA_EPS = [1.8, 2.0, 2.2, 2.3, 2.5, 2.8, 3.0]
GRILLA_MIN_SAMPLES = [3, 4, 5, 7, 10]


def metricas_internas(X: np.ndarray, labels: np.ndarray):
    """Silhouette, Davies-Bouldin y Calinski-Harabasz sobre los puntos con etiqueta != -1.

    Devuelve (dict, None) o (None, motivo) cuando las métricas no son calculables.
    """
    mascara = labels != -1
    Xv, lv = X[mascara], labels[mascara]
    n_grupos = len(np.unique(lv))
    if n_grupos < 2:
        return None, "se necesitan al menos 2 clústeres (sin contar el ruido)"
    if n_grupos >= len(lv):
        return None, "cada punto forma su propio clúster"
    return {
        "Silhouette": float(silhouette_score(Xv, lv)),
        "Davies-Bouldin": float(davies_bouldin_score(Xv, lv)),
        "Calinski-Harabasz": float(calinski_harabasz_score(Xv, lv)),
    }, None


@st.cache_data
def evaluar_rango_k(X: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    """Ajusta K-Means para cada K del rango y devuelve inertia y métricas internas."""
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        met, _ = metricas_internas(X, km.labels_)
        filas.append({"K": k, "Inertia": km.inertia_, **(met or {})})
    return pd.DataFrame(filas)


@st.cache_data
def ajustar_kmeans(X: np.ndarray, k: int):
    """Ajusta K-Means con k clusters y devuelve labels, centroides, inercia e iteraciones."""
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    return km.labels_, km.cluster_centers_, float(km.inertia_), int(km.n_iter_)


@st.cache_data
def distancias_k(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Distancia de cada punto a su vecino n.º min_samples (contando el propio punto), ordenada.

    Un punto es núcleo en DBSCAN si esta distancia es <= eps.
    """
    vecinos = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = vecinos.kneighbors(X)
    return np.sort(dist[:, -1])


@st.cache_data
def ajustar_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Ajusta DBSCAN con distancia euclídea."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)


@st.cache_data
def sensibilidad_dbscan(X: np.ndarray, eps_lista: tuple, ms_lista: tuple) -> pd.DataFrame:
    """Evalúa la sensibilidad de DBSCAN para una grilla de parámetros eps y min_samples."""
    filas = []
    for ms in ms_lista:
        for eps in eps_lista:
            labels = ajustar_dbscan(X, eps, ms)
            n_ruido = int((labels == -1).sum())
            met, _ = metricas_internas(X, labels)
            filas.append({
                "eps": eps,
                "min_samples": ms,
                "N.º clústeres": len(set(labels)) - (1 if n_ruido else 0),
                "N.º ruido": n_ruido,
                "% ruido": round(100 * n_ruido / len(labels), 1),
                "Silhouette (sin ruido)": f"{met['Silhouette']:.3f}" if met else "No calculable",
            })
    return pd.DataFrame(filas)

