# Dashboard de Salud del Sueño

Dashboard interactivo para el análisis y visualización de trastornos del sueño basado en técnicas de Machine Learning (PCA, t-SNE, UMAP) y un motor de reglas clínicas.

## Requisitos

- Python 3.9+
- pip

## Instalación

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

### Levantar el dashboard web

```bash
source .venv/Scripts/activate   # Windows
# o source .venv/bin/activate   # Linux/Mac
python app.py
```

Abrir en el navegador: **http://127.0.0.1:5000**

### Ejecutar el pipeline de ML

```bash
python pipeline_ml.py
```

Esto genera:
- `resultados_ml.json` — métricas y resultados
- `static/matriz_confusion.png` — matriz de confusión
- `static/feature_importance.png` — top 5 variables
- `static/distribucion_clases.png` — distribución de clases

## Endpoints API

| Ruta | Descripción |
|------|-------------|
| `GET /` | Página principal del dashboard |
| `GET /api/data` | Datos completos + embeddings PCA/t-SNE/UMAP |
| `GET /api/ml/results` | Resultados del pipeline de ML |
| `GET /api/riesgo-global` | Resumen de riesgo organizacional |
| `GET /api/alertas/<id>` | Alertas clínicas por empleado |
| `POST /api/predict` | Evaluación de nuevo paciente (envía JSON) |

## Estructura del proyecto

```
dashboarfinal/
├── app.py                              # Servidor Flask + motor de reglas
├── pipeline_ml.py                      # Pipeline de Machine Learning
├── requirements.txt                    # Dependencias
├── vector_caracteristicas_sleep.csv    # Features normalizadas
├── Sleep_health_dataset_es_transformado.csv  # Datos transformados
├── resultados_ml.json                  # Resultados del ML
├── templates/
│   └── index.html                      # Frontend del dashboard
└── static/
    ├── matriz_confusion.png
    ├── feature_importance.png
    └── distribucion_clases.png
```
