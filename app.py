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
        desc_mostrar.insert(0, "Variable", desc_mostrar.index)
        st.dataframe(
            desc_mostrar,
            use_container_width=True,
            hide_index=True,
            height=(len(desc_mostrar) + 1) * 36 + 10,
        )

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
                sns.histplot(df_vars[var_sel], kde=True, color="#f43f5e", ax=ax, bins=20, edgecolor="#161922", alpha=0.75)
            else:
                ax.hist(df_vars[var_sel], bins=20, color="#f43f5e", edgecolor="#161922", alpha=0.75)
            ax.axvline(media, color="#38bdf8", linestyle="--", linewidth=1.8, label=f"Media ({media:.2f})")
            ax.axvline(mediana, color="#34d399", linestyle="-.", linewidth=1.8, label=f"Mediana ({mediana:.2f})")
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
        df_outliers_mostrar = df_outliers.reset_index().rename(columns={"index": "Variable"})

        modo_box = st.radio(
            "Modo de visualización de boxplots:",
            ["Variable individual a detalle", "Ver todas en cuadrícula"],
            horizontal=True,
            key="modo_box_sel",
        )

        col_config_outliers = {
            "Variable": st.column_config.TextColumn("Variable", width="medium"),
            "Límite Inferior": st.column_config.NumberColumn("Lím. Inferior", format="%.2f"),
            "Límite Superior": st.column_config.NumberColumn("Lím. Superior", format="%.2f"),
            "Cant. Outliers": st.column_config.NumberColumn("Cant. Outliers", format="%d"),
            "% Outliers": st.column_config.NumberColumn("% Outliers", format="%.2f%%"),
        }

        if modo_box == "Variable individual a detalle":
            col_t1, col_t2 = st.columns([1.4, 1.0], gap="large")
            with col_t1:
                st.markdown("**Resumen de atípicos (Criterio $1.5 \times IQR$):**")
                st.dataframe(
                    df_outliers_mostrar,
                    use_container_width=True,
                    hide_index=True,
                    height=(len(df_outliers_mostrar) + 1) * 36 + 10,
                    column_config=col_config_outliers,
                )
            with col_t2:
                st.markdown("**Inspección visual de boxplots:**")
                var_box = st.selectbox("Selecciona variable para boxplot:", vars_analisis, key="box_sel")

                q25_v = df_vars[var_box].quantile(0.25)
                q75_v = df_vars[var_box].quantile(0.75)
                n_out_v = outliers_dict[var_box]["Cant. Outliers"]

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Q1 (25%)", f"{q25_v:.2f}")
                col_b2.metric("Mediana", f"{df_vars[var_box].median():.2f}")
                col_b3.metric("Outliers", f"{n_out_v}")

                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                if HAS_SEABORN:
                    sns.boxplot(
                        x=df_vars[var_box],
                        color="#f43f5e",
                        ax=ax,
                        flierprops={"marker": "o", "color": "#f87171", "markersize": 6, "markeredgecolor": "#ffffff"},
                        boxprops={"edgecolor": "#ffffff", "linewidth": 1},
                        whiskerprops={"color": "#e2e8f0", "linewidth": 1},
                        capprops={"color": "#e2e8f0", "linewidth": 1},
                        medianprops={"color": "#fbbf24", "linewidth": 2},
                    )
                else:
                    ax.boxplot(
                        df_vars[var_box],
                        vert=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#f43f5e", edgecolor="#ffffff"),
                        whiskerprops=dict(color="#e2e8f0"),
                        capprops=dict(color="#e2e8f0"),
                        medianprops=dict(color="#fbbf24", linewidth=2),
                        flierprops=dict(marker="o", markerfacecolor="#f87171", markersize=6, markeredgecolor="#ffffff"),
                    )
                ax.set_title(f"Diagrama de Caja de {var_box}", fontsize=11, fontweight="bold")
                ax.set_xlabel(var_box)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.markdown("**Resumen de atípicos (Criterio $1.5 \times IQR$):**")
            st.dataframe(
                df_outliers_mostrar,
                use_container_width=True,
                hide_index=True,
                height=(len(df_outliers_mostrar) + 1) * 36 + 10,
                column_config=col_config_outliers,
            )
            st.markdown("---")
            st.markdown("**Cuadrícula de Diagramas de Caja (todas las variables fisicoquímicas):**")
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
                n_out = outliers_dict[col]["Cant. Outliers"]
                titulo_color = "#c0392b" if n_out > 0 else "#2c3e50"
                ax.set_title(f"{col} ({n_out} atípicos)", fontsize=10, fontweight="bold", color=titulo_color)
                ax.set_xlabel("")
            for j in range(len(vars_analisis), len(axes_flat)):
                fig.delaxes(axes_flat[j])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption("Variables con outliers detectados: `magnesium` (4), `alcalinity_of_ash` (4), `color_intensity` (4), `malic_acid` (3), `ash` (3), `proanthocyanins` (2) y `hue` (1).")

    # 5. Correlación
    with pestanas[4]:
        st.subheader("5. Matriz de correlación lineal de Pearson")
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
            ax.plot(k_sugerido, val_opt, marker="o", markersize=10, color=color, markeredgecolor="#ffffff", markeredgewidth=2)
            ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=8)
            ax.set_xlabel("Número de clústeres (K)", fontsize=9)
            ax.set_xticks(tabla_k["K"])
            ax.grid(True, linestyle="--", alpha=0.35, color="#2c3345")
            ax.legend(fontsize=8, loc="best", framealpha=0.7)
        plt.tight_layout(pad=2.2)
        st.pyplot(fig)
        plt.close(fig)

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
        tam_df = pd.DataFrame({"Clúster": tamanos.index, "Tamaño": tamanos.values,
                               "%": (100 * tamanos.values / n).round(1)})
        col_t, col_g = st.columns([1, 2])
        col_t.dataframe(tam_df, hide_index=True, width="stretch")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(tamanos.index.astype(str), tamanos.values, color=colores_clusters(k), edgecolor="black")
        ax.set_xlabel("Clúster")
        ax.set_ylabel("N.º de muestras")
        ax.set_title("Tamaño de cada clúster", fontsize=11, fontweight="bold")
        col_g.pyplot(fig)
        plt.close(fig)

        st.session_state["labels_kmeans"] = labels
        st.session_state["kmeans_params"] = {"K": k, "n_init": N_INIT, "random_state": RANDOM_STATE, "init": "k-means++"}
        st.session_state["kmeans_metricas"] = {"Inertia": inertia, **(met or {})}

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
        ax.plot(np.arange(1, n + 1), kdist, color="#38bdf8", linewidth=2.4)
        ax.fill_between(np.arange(1, n + 1), kdist, alpha=0.15, color="#38bdf8")
        ax.axhline(eps, color="#f43f5e", linestyle="--", linewidth=1.8, label=f"eps actual = {eps:.2f}")
        ax.set_xlabel("Puntos ordenados por distancia")
        ax.set_ylabel(f"Distancia al {min_samples - 1}.º vecino más cercano")
        ax.set_title(f"k-distance (min_samples = {min_samples})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
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
        st.dataframe(tam_df, hide_index=True, width="stretch")

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
        
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.bar(range(1, n_comp + 1), varianza_exp[:n_comp], alpha=0.75, color="#f43f5e", edgecolor="#ffffff", linewidth=0.5, label="Varianza Individual")
        ax1.set_xlabel("Componente Principal")
        ax1.set_ylabel("Proporción de Varianza Explicada", color="#f43f5e")
        ax1.tick_params(axis='y', labelcolor="#f43f5e")
        ax1.set_xticks(range(1, n_comp + 1))
        
        ax2 = ax1.twinx()
        ax2.plot(range(1, n_comp + 1), varianza_acum[:n_comp], marker="o", color="#38bdf8", linewidth=2.4, label="Varianza Acumulada")
        ax2.set_ylabel("Varianza Acumulada", color="#38bdf8")
        ax2.tick_params(axis='y', labelcolor="#38bdf8")
        ax2.axhline(0.80, color="#94a3b8", linestyle="--", alpha=0.6, label="Umbral 80%")
        
        fig.suptitle("Scree Plot: Varianza por Componente", fontweight="bold", color="#f0f2f6")
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

        fig, ax = plt.subplots(figsize=(8, 5.5))
        if color_array is not None:
            # Mapeo de colores
            unique_labels = np.unique(color_array)
            cmap = plt.get_cmap("tab10")
            for lbl in unique_labels:
                mask = color_array == lbl
                color = "gray" if lbl == 99 else cmap(lbl % 10)
                marker = "x" if lbl == 99 else "o"
                label_name = "Ruido (-1)" if lbl == 99 else f"Clúster {lbl}"
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], marker=marker, 
                           edgecolors="black" if lbl != 99 else None, linewidth=0.4,
                           alpha=0.7, s=45, label=label_name)
            ax.legend()
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="#722f37", edgecolors="black", linewidth=0.4, alpha=0.7, s=45)

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
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(tam_km.index.astype(str), tam_km.values, color=colores_clusters(n_clusters_km), edgecolor="black")
        ax.set_xlabel("Clúster")
        ax.set_ylabel("N.º de muestras")
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.markdown("**Distribución de Clústeres (DBSCAN)**")
        tam_db = pd.Series(labels_db).value_counts().sort_index()
        labels_x = ["Ruido (-1)" if i == -1 else f"C{i}" for i in tam_db.index]
        colors_db = ["gray" if i == -1 else colores_clusters(n_clusters_db)[i] for i in tam_db.index]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels_x, tam_db.values, color=colors_db, edgecolor="black")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("N.º de muestras")
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

