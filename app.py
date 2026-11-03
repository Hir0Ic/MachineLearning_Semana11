"""Descubrimiento de patrones fisicoquímicos en vinos: K-Means, DBSCAN y PCA."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

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


def requerir_datos():
    """Comprueba si hay datos cargados; si no, ofrece cargarlos inmediatamente."""
    df, etiqueta = obtener_datos()
    if df is None:
        st.warning("⚠️ Aún no se han cargado datos. Puedes cargarlos en la sección **Datos** o hacer clic abajo:")
        if st.button("🍷 Cargar dataset Wine por defecto"):
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
    Permite que la Parte 3 (K-Means / DBSCAN) y Parte 4 (PCA) consuman la matriz inmediatamente.
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
        except Exception as e:
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
    st.dataframe(df.head(n), use_container_width=True)


# ------------------------------------------------------------------- EDA
def pagina_eda():
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")
    df, etiqueta = requerir_datos()
    if df is None:
        return

    vars_analisis = [c for c in df.columns if c != etiqueta]
    df_vars = df[vars_analisis]

    pestanas = st.tabs([
        "1. Estructura y Calidad",
        "2. Estadísticos Descriptivos",
        "3. Distribuciones (Histogramas)",
        "4. Outliers (Boxplots)",
        "5. Matriz de Correlación",
        "6. Hallazgos Clave",
    ])

    # 1. Estructura y Calidad
    with pestanas[0]:
        st.subheader("1. Estructura y calidad del conjunto de datos")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total observaciones", df.shape[0])
        col_b.metric("Total variables predictoras", len(vars_analisis))
        col_c.metric("Filas duplicadas", int(df_vars.duplicated().sum()))

        st.markdown("**Resumen de tipos de datos y valores faltantes por columna:**")
        info_df = pd.DataFrame({
            "Tipo de Dato": df_vars.dtypes.astype(str),
            "Valores Presentes": df_vars.count(),
            "Valores Nulos": df_vars.isnull().sum(),
            "% Nulos": (df_vars.isnull().mean() * 100).round(2),
            "Valores Únicos": df_vars.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)
        if df_vars.isnull().sum().sum() == 0:
            st.success("✅ No existen valores nulos ni registros duplicados en las variables fisicoquímicas.")

    # 2. Estadísticos Descriptivos
    with pestanas[1]:
        st.subheader("2. Resumen estadístico multivariado")
        desc = df_vars.describe().T
        desc["rango"] = desc["max"] - desc["min"]
        desc["iqr"] = desc["75%"] - desc["25%"]
        desc["skewness"] = df_vars.skew()
        
        columnas_orden = ["mean", "std", "min", "25%", "50%", "75%", "max", "rango", "iqr", "skewness"]
        nombres_amigables = {
            "mean": "Media", "std": "Desv. Est.", "min": "Mín", "25%": "Q1 (25%)",
            "50%": "Mediana", "75%": "Q3 (75%)", "max": "Máx", "rango": "Rango",
            "iqr": "IQR", "skewness": "Asimetría"
        }
        desc_mostrar = desc[columnas_orden].rename(columns=nombres_amigables).round(3)
        st.dataframe(desc_mostrar, use_container_width=True)

        st.info(
            "💡 **Observación crítica de escalas:** Las variables presentan magnitudes y rangos drásticamente dispares. "
            "Por ejemplo, `proline` alcanza valores de hasta 1,680 con un rango de 1,402, mientras que "
            "`nonflavanoid_phenols` varía únicamente entre 0.13 y 0.66. "
            "Esto demuestra la **necesidad obligatoria de estandarización** antes de aplicar algoritmos basados en distancias (K-Means, DBSCAN y PCA)."
        )

    # 3. Distribuciones
    with pestanas[2]:
        st.subheader("3. Distribución de las variables fisicoquímicas")
        modo_vista = st.radio("Modo de visualización:", ["Variable individual a detalle", "Ver todas en cuadrícula"], horizontal=True)

        if modo_vista == "Variable individual a detalle":
            var_sel = st.selectbox("Selecciona la variable a inspeccionar:", vars_analisis)
            media = df_vars[var_sel].mean()
            mediana = df_vars[var_sel].median()
            desv = df_vars[var_sel].std()
            asimetria = df_vars[var_sel].skew()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Media", f"{media:.2f}")
            col_m2.metric("Mediana", f"{mediana:.2f}")
            col_m3.metric("Desv. Estándar", f"{desv:.2f}")
            col_m4.metric("Coef. Asimetría", f"{asimetria:.2f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            if HAS_SEABORN:
                sns.histplot(df_vars[var_sel], kde=True, color="#722f37", ax=ax, bins=20, edgecolor="black", alpha=0.6)
            else:
                ax.hist(df_vars[var_sel], bins=20, color="#722f37", edgecolor="black", alpha=0.6)
            ax.axvline(media, color="blue", linestyle="--", linewidth=1.8, label=f"Media ({media:.2f})")
            ax.axvline(mediana, color="green", linestyle="-.", linewidth=1.8, label=f"Mediana ({mediana:.2f})")
            ax.set_title(f"Distribución de {var_sel}", fontsize=12, fontweight="bold")
            ax.set_xlabel(var_sel)
            ax.set_ylabel("Frecuencia")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
            axes_flat = axes.flatten()
            for i, col in enumerate(vars_analisis):
                if HAS_SEABORN:
                    sns.histplot(df_vars[col], kde=True, ax=axes_flat[i], color="#722f37", bins=15)
                else:
                    axes_flat[i].hist(df_vars[col], bins=15, color="#722f37", edgecolor="black", alpha=0.6)
                axes_flat[i].set_title(col, fontsize=10, fontweight="bold")
                axes_flat[i].set_xlabel("")
                axes_flat[i].set_ylabel("")
            for j in range(len(vars_analisis), len(axes_flat)):
                fig.delaxes(axes_flat[j])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # 4. Outliers (Boxplots)
    with pestanas[3]:
        st.subheader("4. Detección de valores atípicos (Outliers)")
        
        outliers_dict = {}
        for col in vars_analisis:
            q25 = df_vars[col].quantile(0.25)
            q75 = df_vars[col].quantile(0.75)
            iqr = q75 - q25
            lim_inf = q25 - 1.5 * iqr
            lim_sup = q75 + 1.5 * iqr
            conteo = int(((df_vars[col] < lim_inf) | (df_vars[col] > lim_sup)).sum())
            outliers_dict[col] = {
                "Límite Inferior": round(lim_inf, 2),
                "Límite Superior": round(lim_sup, 2),
                "Cant. Outliers": conteo,
                "% Outliers": round((conteo / len(df_vars)) * 100, 2),
            }
        df_outliers = pd.DataFrame(outliers_dict).T.sort_values(by="Cant. Outliers", ascending=False)

        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.markdown("**Resumen de atípicos (Criterio $1.5 \\times IQR$):**")
            st.dataframe(df_outliers, use_container_width=True)
        with col_t2:
            st.markdown("**Inspección visual de boxplots:**")
            var_box = st.selectbox("Selecciona variable para boxplot:", vars_analisis, key="box_sel")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if HAS_SEABORN:
                sns.boxplot(x=df_vars[var_box], color="#deb887", ax=ax, flierprops={"marker": "o", "color": "red", "markersize": 6})
            else:
                ax.boxplot(df_vars[var_box], vert=False, patch_artist=True, boxprops=dict(facecolor="#deb887"))
            ax.set_title(f"Diagrama de Caja de {var_box}", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        st.caption("Variables con outliers detectados: `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `proanthocyanins`, `color_intensity` y `hue`.")

    # 5. Correlación
    with pestanas[4]:
        st.subheader("5. Matriz de correlación lineal de Pearson")
        corr = df_vars.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        if HAS_SEABORN:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                mask=mask,
                cmap="vlag",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8},
            )
        else:
            cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            for r in range(len(corr.columns)):
                for c in range(len(corr.columns)):
                    val = corr.iloc[r, c]
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.6 else "white", fontsize=7)

        ax.set_title("Matriz de Correlación", fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        pares_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col1 = corr.columns[i]
                col2 = corr.columns[j]
                pares_corr.append((col1, col2, corr.iloc[i, j], abs(corr.iloc[i, j])))
        pares_df = pd.DataFrame(pares_corr, columns=["Variable 1", "Variable 2", "Correlación (r)", "Magnitud"])
        pares_df = pares_df.sort_values(by="Magnitud", ascending=False).drop(columns=["Magnitud"]).head(6)

        st.markdown("**Pares de variables con mayor correlación lineal:**")
        st.dataframe(pares_df.reset_index(drop=True), use_container_width=True)

        st.subheader("Gráficos de dispersión de relaciones destacadas")
        st.caption("Dispersión bivariada con línea de tendencia lineal ajustada.")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_d1, ax_d1 = plt.subplots(figsize=(6, 4))
            x1 = df_vars["total_phenols"]
            y1 = df_vars["flavanoids"]
            ax_d1.scatter(x1, y1, color="#722f37", edgecolors="black", linewidth=0.6, alpha=0.7, s=45)
            m1, b1 = np.polyfit(x1, y1, 1)
            x_vals1 = np.linspace(x1.min(), x1.max(), 100)
            ax_d1.plot(x_vals1, m1 * x_vals1 + b1, color="#003366", linestyle="--", linewidth=1.8, label=f"Tendencia (r = +0.865)")
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
            ax_d2.scatter(x2, y2, color="#2e6f40", edgecolors="black", linewidth=0.6, alpha=0.7, s=45)
            m2, b2 = np.polyfit(x2, y2, 1)
            x_vals2 = np.linspace(x2.min(), x2.max(), 100)
            ax_d2.plot(x_vals2, m2 * x_vals2 + b2, color="#8b0000", linestyle="--", linewidth=1.8, label=f"Tendencia (r = +0.644)")
            ax_d2.set_title("alcohol vs. proline", fontsize=11, fontweight="bold")
            ax_d2.set_xlabel("alcohol [% vol]")
            ax_d2.set_ylabel("proline [mg/L]")
            ax_d2.legend(loc="upper left", fontsize=8)
            ax_d2.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_d2)
            plt.close(fig_d2)

    # 6. Hallazgos Clave
    with pestanas[5]:
        st.subheader("6. Síntesis de Hallazgos del EDA (para Informe)")
        st.markdown(
            """
            * **Calidad de datos íntegra:** 178 registros completos, 0 valores nulos y 0 duplicados en las 13 variables químicas.
            * **Escalas heterogéneas:** Se confirma una disparidad de órdenes de magnitud (`proline` y `magnesium` frente a `flavanoids` y `nonflavanoid_phenols`), lo que distorsionaría cualquier distancia euclídea sin un escalado previo.
            * **Presencia de valores atípicos:** 7 de las 13 variables contienen valores fuera del rango intercuartílico $1.5 \\times IQR$, destacando `malic_acid` y `magnesium`.
            * **Colinealidad moderada-alta:** Existe una fuerte asociación positiva entre `flavanoids` y `total_phenols` ($r = 0.86$), y entre `flavanoids` y `od280/od315` ($r = 0.79$), lo que anticipa una excelente compresión dimensional mediante PCA.
            """
        )


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
          $$z = \\frac{x - \\mu}{\\sigma}$$
          Centra los datos con media $\\mu = 0$ y desviación estándar $\\sigma = 1$. Es el estándar óptimo para
          algoritmos basados en distancias euclídeas y varianza (K-Means y PCA).
          
        * **Decisión B: RobustScaler (Mediana e IQR)**
          $$z = \\frac{x - \\text{mediana}}{\\text{IQR}}$$
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
        
        if HAS_SEABORN:
            sns.boxplot(data=X_raw, ax=axes[0], orient="h", color="#b0c4de")
            sns.boxplot(data=X_std, ax=axes[1], orient="h", color="#90ee90")
            sns.boxplot(data=X_rob, ax=axes[2], orient="h", color="#ffb6c1")
        else:
            axes[0].boxplot([X_raw[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True, boxprops=dict(facecolor="#b0c4de"))
            axes[1].boxplot([X_std[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True, boxprops=dict(facecolor="#90ee90"))
            axes[2].boxplot([X_rob[c] for c in vars_x], vert=False, tick_labels=vars_x, patch_artist=True, boxprops=dict(facecolor="#ffb6c1"))

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


# ------------------------------------------------------------------ K-Means
def pagina_kmeans():
    st.header("🎯 Agrupamiento K-Means")
    X_final, X_raw, vars_x, scaler = obtener_datos_preprocesados()
    if X_final is None:
        return

    st.success(f"✅ Conexión con Parte 2 exitosa: Matriz lista con {X_final.shape[0]} muestras y {X_final.shape[1]} variables ({st.session_state.get('scaler_nombre', 'StandardScaler')}).")
    st.info("📌 **Espacio listo para la Parte 3:** Slider interactivo de K (2 a 10), cálculo de curva del codo (inercia), métricas de Silhouette, Davies-Bouldin, Calinski-Harabasz y tabla de centroides/perfil.")


# ------------------------------------------------------------------- DBSCAN
def pagina_dbscan():
    st.header("🔬 Agrupamiento DBSCAN")
    X_final, X_raw, vars_x, scaler = obtener_datos_preprocesados()
    if X_final is None:
        return

    st.success(f"✅ Conexión con Parte 2 exitosa: Matriz lista con {X_final.shape[0]} muestras y {X_final.shape[1]} variables ({st.session_state.get('scaler_nombre', 'StandardScaler')}).")
    st.info("📌 **Espacio listo para la Parte 3:** Controles para eps y min_samples, cálculo del gráfico k-distance, conteo de clústeres válidos (sin -1) y porcentaje de ruido.")


# ------------------------------------------------------------------ Main
PAGINAS = {
    "Inicio": pagina_inicio,
    "Datos": pagina_datos,
    "EDA": pagina_eda,
    "Preprocesamiento": pagina_preprocesamiento,
    "K-Means": pagina_kmeans,
    "DBSCAN": pagina_dbscan,
}

st.sidebar.title("🍷 Navegación")
seccion = st.sidebar.radio("Sección", SECCIONES, label_visibility="collapsed")
PAGINAS.get(seccion, lambda: en_construccion(seccion))()
