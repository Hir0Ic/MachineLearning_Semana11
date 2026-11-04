# 🍷 Descubrimiento de Patrones Fisicoquímicos en Vinos
## Aprendizaje No Supervisado: K-Means, DBSCAN y PCA

**Curso:** Sistemas Inteligentes y Machine Learning  
**Grupo:** 03 · Universidad Privada del Norte (UPN)  
**Semana:** 11  

---

## 📌 1. Descripción del Proyecto

Aplicación web analítica e interactiva desarrollada con **Streamlit** para el descubrimiento de patrones y segmentación no supervisada sobre el dataset clásico de vinos de UCI (*Wine Recognition Dataset*, Aeberhard & Forina, 1992).

El objetivo central es responder a la pregunta de investigación:  
> **¿Qué patrones o segmentos fisicoquímicos se pueden descubrir en los datos y qué evidencia cuantitativa demuestra que los resultados son interpretables, robustos y útiles para la caracterización de los vinos?**

La solución implementa un flujo completo de Ciencia de Datos:
1. **Auditoría y Análisis Exploratorio de Datos (EDA)** multivariado.
2. **Preprocesamiento y Estandarización** comparando decisiones metodológicas (`StandardScaler` vs. `RobustScaler`).
3. **Agrupamiento Particional con K-Means** (selección de $K$, método del codo, coeficientes de Silhouette, Davies-Bouldin y Calinski-Harabasz, centroides y perfiles enológicos).
4. **Agrupamiento por Densidad con DBSCAN** (calibración mediante gráfico *k-distance*, detección de ruido analítico y matrices de sensibilidad a `eps` y `min_samples`).
5. **Reducción Dimensional con PCA** (*Scree Plot*, varianza acumulada, cargas factoriales *loadings*, biplot y proyecciones 2D).
6. **Comparativa Metodológica Crítica** (K-Means vs. DBSCAN: cobertura vs. ruido, métricas internas y exportación de etiquetas).
7. **Conclusiones, Limitaciones y Recomendaciones** ejecutivas.

---

## 📂 2. Estructura del Repositorio

```text
MachineLearning_Semana11/
│
├── app.py                                  # Aplicación principal interactiva en Streamlit
├── requirements.txt                        # Lista de dependencias del proyecto
├── README.md                               # Guía paso a paso y documentación técnica
├── .gitignore                              # Archivos y carpetas excluidos del control de versiones
│
├── data/
│   └── wine.csv                            # Copia local del dataset UCI Wine (178 muestras, 13 variables)
│
├── imagenes_informe/                       # Figuras en alta resolución (300 DPI) para el informe Word
│   ├── figura_1_histogramas_distribucion.png
│   ├── figura_2_boxplots_outliers.png
│   ├── figura_3_matriz_correlacion.png
│   ├── figura_4_comparacion_escaladores.png
│   └── figura_5_dispersion_fenoles_y_prolina.png
│
├── informe_parte2_datos_y_preparacion.txt  # Resultados, tablas y redacción de la Parte 2
└── informe_parte5_cierre_y_entrega.txt     # Documento maestro con discusión, conclusiones y anexos
```

---

## 🚀 3. Guía de Instalación y Ejecución Paso a Paso

### Paso 1: Clonar o descargar el repositorio
Abre una terminal o consola de comandos y navega hasta la carpeta del proyecto:
```bash
cd MachineLearning_Semana11
```

### Paso 2: Crear y activar un entorno virtual (recomendado)
* **En Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Paso 3: Instalar las dependencias
Instala todas las librerías necesarias especificadas en `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Paso 4: Iniciar la aplicación web en Streamlit
Ejecuta el servidor de desarrollo local:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web predeterminado en la dirección:  
👉 **`http://localhost:8501`**

---

## 🧭 4. Recorrido por las Secciones de la Aplicación

La barra lateral izquierda contiene el menú de navegación organizado en 9 secciones secuenciales:

| N.° | Sección | Contenido y Funcionalidad |
| :---: | :--- | :--- |
| **1** | **Inicio** | Presentación, pregunta central de negocio, objetivo y guía de uso. |
| **2** | **Datos** | Carga del dataset oficial de UCI Wine (`load_wine`) o subida de archivo propio (CSV/Excel). Aislamiento de la etiqueta externa, ficha técnica y validación de estructura. |
| **3** | **EDA** | 6 subpestañas: Estructura y tipos, estadísticas multivariadas, histogramas con KDE (individual o cuadrícula), detección de outliers (regla de Tukey $1.5 \times \text{IQR}$), matriz de correlación de Pearson con gráficos de dispersión bivariados, y síntesis de hallazgos. |
| **4** | **Preprocesamiento** | Separación estricta de la matriz $X$ ($178 \times 13$). Comparación teórica y visual de dos decisiones: `StandardScaler` (Z-score) vs. `RobustScaler` (IQR). Persistencia en `st.session_state` y parámetros reproducibles (`random_state=42`). |
| **5** | **K-Means** | Evaluación automática de $K \in [2, 10]$ con 4 métricas internas (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz). Slider dinámico de $K$, tabla de centroides en unidades químicas originales y escala $Z$, análisis de perfiles y visualización proyectada en 2D. |
| **6** | **DBSCAN** | Gráfico *k-distance* con percentiles para determinar `eps`. Controles interactivos de radio `eps` y `min_samples`. Matriz de calor de sensibilidad, perfil del ruido (muestras atípicas marcadas con `-1`) y evaluación métrica sin ruido. |
| **7** | **PCA** | Scree plot con proporción individual y varianza acumulada (80% alcanzado con 5 componentes). Cargas factoriales (*loadings*) por variable, biplot con vectores directores y dispersión en el plano factorial PC1-PC2 coloreada por clúster. |
| **8** | **Comparación** | Tabla comparativa de indicadores entre K-Means y DBSCAN (cobertura vs. ruido, métricas internas calculadas adecuadamente). Gráficos de barras de distribución y botón de descarga de dataset con etiquetas asignadas (`vinos_agrupados_kmeans_dbscan.csv`). |
| **9** | **Conclusiones** | Síntesis ejecutiva de perfiles de vino, las 5 conclusiones centrales cuantitativas, 3 limitaciones metodológicas, 3 recomendaciones para la industria y ficha ejecutiva descargable. |

---

## 📊 5. Parámetros Técnicos y Reproducibilidad

Para asegurar la exacta reproducibilidad de todos los experimentos:
* **Semilla global (SEED):** `RANDOM_STATE = 42`
* **Dataset:** 178 registros, 13 variables fisicoquímicas (12 continuas y 1 entera: `magnesium`), 0 nulos, 0 duplicados.
* **Escalador principal:** `StandardScaler(with_mean=True, with_std=True)`
* **K-Means óptimo:** $K = 3$, `init='k-means++'`, `n_init=10` ($\text{Silhouette} = 0.285$, $\text{Inertia} = 1277.9$). 172 de 178 muestras (96.6%) se asignan al cultivar dominante de cada grupo.
* **DBSCAN óptimo:** $\text{eps} = 2.30$, $\text{min\_samples} = 5$ (2 grupos densos + 42 muestras de ruido [23.6%], $\text{Silhouette} = 0.331$ sobre las 136 muestras asignadas; $\text{Davies-Bouldin} = 1.182$, $\text{Calinski-Harabasz} = 70.5$).
* **PCA:** 3 componentes principales explican el **66.5%** de la varianza total acumulada (PC1: 36.2%, PC2: 19.2%, PC3: 11.1%).

---

## 🛠️ 6. Tecnologías y Librerías

* **Lenguaje:** Python 3.12+
* **Framework Web:** Streamlit (>= 1.50)
* **Procesamiento de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (KMeans, DBSCAN, PCA, StandardScaler, RobustScaler, métricas internas)
* **Visualización Científica:** Matplotlib, Seaborn

---

## 👥 7. Equipo de Trabajo (Grupo 03 - UPN)

* **Parte 1:** Problema, base y estructura inicial.
* **Parte 2:** Análisis Exploratorio de Datos (EDA) y Preprocesamiento de características.
* **Parte 3:** Agrupamiento particional con K-Means y agrupamiento por densidad con DBSCAN.
* **Parte 4:** Reducción de dimensionalidad con PCA y comparativa metodológica.
* **Parte 5:** Conclusiones, discusión crítica, control de errores, documentación y entrega final.
