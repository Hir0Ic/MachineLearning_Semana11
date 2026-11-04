"""
Módulo de carga de datos, lectura de archivos y preparación de la matriz X.
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import streamlit as st

SECCIONES = [
    "Inicio",
    "Datos",
    "EDA",
    "Preprocesamiento",
    "K-Means",
    "DBSCAN",
    "PCA",
    "Comparación",
    "Conclusiones",
]

FICHA = {
    "Dataset": "Wine — UCI Machine Learning Repository",
    "Enlace": "https://archive.ics.uci.edu/dataset/109/wine",
    "DOI": "10.24432/C5PC7J",
    "Autores": "Stefan Aeberhard y M. Forina (1992)",
    "Licencia": "CC BY 4.0",
    "Unidad de observación": "Muestra de vino",
    "Etiqueta externa": "class (3 cultivares): se excluye del clustering",
}


@st.cache_data
def cargar_wine() -> pd.DataFrame:
    """Dataset Wine incluido en scikit-learn (13 variables + 'target')."""
    return load_wine(as_frame=True).frame


@st.cache_data
def leer_archivo(archivo) -> pd.DataFrame:
    """Lee un CSV/Excel subido por el usuario."""
    if archivo.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)
    return pd.read_csv(archivo)


def obtener_datos():
    """Devuelve (df, columna_etiqueta) guardados en session_state."""
    return st.session_state.get("df"), st.session_state.get("etiqueta")


def en_construccion(nombre: str):
    """Placeholder para páginas no implementadas."""
    st.header(nombre)
    st.info("Sección en construcción.")


def requerir_datos():
    """Comprueba si hay datos cargados; si no, ofrece cargarlos inmediatamente."""
    df, etiqueta = obtener_datos()
    if df is None:
        st.warning("Para continuar necesitas cargar datos.")
        if st.button("Cargar dataset Wine ahora"):
            st.session_state["df"] = cargar_wine()
            st.session_state["etiqueta"] = "target"
            st.rerun()
        return None, None
    return df, etiqueta


def obtener_datos_preprocesados():
    """
    Devuelve (X_final, X_raw, vars_x, scaler_obj) guardados en session_state.
    Si aún no se han procesado en la sección 'Preprocesamiento', los prepara
    automáticamente con la configuración estándar (StandardScaler).
    Permite que K-Means, DBSCAN y PCA consuman la matriz inmediatamente.
    """
    df, etiqueta = requerir_datos()
    if df is None:
        return None, None, None, None

    if "X_final" not in st.session_state:
        vars_x = [c for c in df.columns if c != etiqueta]
        X_raw = df[vars_x].copy()
        scaler = StandardScaler()
        X_final = pd.DataFrame(scaler.fit_transform(X_raw), columns=vars_x)
        st.session_state["X_final"] = X_final
        st.session_state["X_raw"] = X_raw
        st.session_state["scaler_obj"] = scaler
        st.session_state["scaler_nombre"] = "StandardScaler"
        st.session_state["vars_x"] = vars_x

    return (
        st.session_state["X_final"],
        st.session_state["X_raw"],
        st.session_state["vars_x"],
        st.session_state["scaler_obj"],
    )


def matriz_para_clustering():
    """Obtiene la matriz X de la Parte 2 y valida que sirva para agrupar."""
    try:
        X_final, X_raw, vars_x, scaler = obtener_datos_preprocesados()
    except (ValueError, TypeError) as e:
        st.error(
            "No se pudo preparar la matriz X. Verifica que todas las columnas (salvo la etiqueta) "
            f"sean numéricas y no tengan nulos. Detalle: {e}"
        )
        return None, None, None, None
    if X_final is None:
        return None, None, None, None
    if X_final.isna().any().any():
        st.error("La matriz X contiene valores nulos; K-Means y DBSCAN no pueden calcularse.")
        return None, None, None, None
    if X_final.shape[0] < 3:
        st.error("Se necesitan al menos 3 registros para agrupar y calcular métricas.")
        return None, None, None, None
    st.caption(
        f"Matriz X: {X_final.shape[0]} muestras × {X_final.shape[1]} variables · "
        f"escalado: **{st.session_state.get('scaler_nombre', 'StandardScaler')}** · etiqueta externa excluida."
    )
    return X_final, X_raw, vars_x, scaler

