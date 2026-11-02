"""Descubrimiento de patrones fisicoquímicos en vinos: K-Means, DBSCAN y PCA."""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine

st.set_page_config(page_title="Agrupamiento de vinos", page_icon="🍷", layout="wide")

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
    st.header(nombre)
    st.info("Sección en construcción.")


# ---------------------------------------------------------------- Inicio
def pagina_inicio():
    st.title("🍷 Descubrimiento de patrones fisicoquímicos en vinos")
    st.caption("K-Means, DBSCAN y PCA · Sistemas Inteligentes y Machine Learning · Grupo 03")
    st.markdown(
        """
        **Pregunta central:** ¿qué patrones o segmentos se pueden descubrir en los datos
        y qué evidencia demuestra que los resultados son interpretables y útiles?

        Esta aplicación permite cargar los datos, explorarlos, prepararlos y comparar
        agrupamientos obtenidos con **K-Means** y **DBSCAN**, usando **PCA** para reducir
        la dimensionalidad y visualizar los resultados. La etiqueta original del dataset
        **no** se usa para formar los grupos.
        """
    )
    st.subheader("Cómo usar la aplicación")
    st.markdown(
        "1. Ve a **Datos** y elige el dataset Wine o sube tu propio archivo.\n"
        "2. Recorre las secciones en orden desde la barra lateral."
    )
    df, _ = obtener_datos()
    if df is None:
        st.warning("Aún no hay datos cargados. Ve a la sección **Datos**.")
    else:
        st.success(f"Datos cargados: {df.shape[0]} registros y {df.shape[1]} columnas.")


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
        except Exception as e:  # archivo ilegible
            st.error(f"No se pudo leer el archivo: {e}")
            return
        opciones = ["(ninguna)"] + list(df_up.columns)
        elegida = st.selectbox("Columna de etiqueta externa (se excluye del clustering)", opciones)
        st.session_state["df"] = df_up
        st.session_state["etiqueta"] = None if elegida == "(ninguna)" else elegida

    df, etiqueta = obtener_datos()
    variables = [c for c in df.columns if c != etiqueta]

    st.subheader("Ficha técnica")
    if origen.startswith("Dataset"):
        st.table(pd.Series(FICHA, name="Detalle").to_frame())
    else:
        st.caption("Archivo propio: la ficha del dataset Wine no aplica.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", df.shape[0])
    c2.metric("Variables", len(variables))
    c3.metric("Valores nulos", int(df.isna().sum().sum()))
    c4.metric("Duplicados", int(df.duplicated().sum()))
    if etiqueta:
        st.caption(f"Etiqueta externa: **{etiqueta}** (fuera de X y de la elección de K).")

    st.subheader("Muestra de registros")
    n = st.slider("Número de filas", 5, min(50, len(df)), 10)
    st.dataframe(df.head(n), width="stretch")


# ------------------------------------------------------------------ Main
PAGINAS = {
    "Inicio": pagina_inicio,
    "Datos": pagina_datos,
}

st.sidebar.title("🍷 Navegación")
seccion = st.sidebar.radio("Sección", SECCIONES, label_visibility="collapsed")
PAGINAS.get(seccion, lambda: en_construccion(seccion))()
