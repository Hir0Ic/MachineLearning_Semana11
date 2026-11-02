# Descubrimiento de patrones fisicoquímicos en vinos

Aplicación web en Streamlit que agrupa muestras de vino con K-Means y DBSCAN y usa PCA para reducir dimensionalidad y visualizar resultados.

Curso: Sistemas Inteligentes y Machine Learning · Grupo 03 · UPN

## Dataset

[Wine – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine) (Aeberhard & Forina, 1992). 178 muestras, 13 variables químicas. La etiqueta original no se usa para formar los grupos.

## Estructura

```
├── app.py              # Aplicación Streamlit
├── requirements.txt    # Dependencias
└── data/wine.csv       # Copia del dataset
```

## Instalación y ejecución

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```