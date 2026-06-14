from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_PATH = BASE_DIR / "data" / "vector_caracteristicas_sleep.csv"
DETALLES_PATH = BASE_DIR / "data" / "Sleep_health_dataset_es_transformado.csv"

COLUMNAS_EXCLUIR = {"id_persona", "trastorno_sueno"}
COLUMNAS_DETALLE = [
    "id_persona",
    "genero",
    "edad",
    "ocupacion",
    "duracion_sueno_horas",
    "calidad_sueno",
    "nivel_actividad_fisica",
    "nivel_estres",
    "categoria_imc",
    "presion_arterial",
    "frecuencia_cardiaca",
    "pasos_diarios",
    "trastorno_sueno",
    "calidad_sueno_nivel",
    "estres_nivel",
    "tiene_anomalia_iqr",
]
ATRIBUTOS_BARRA = [
    "duracion_sueno_horas_norm",
    "calidad_sueno_norm",
    "nivel_actividad_fisica_norm",
    "nivel_estres_norm",
    "frecuencia_cardiaca_norm",
    "pasos_diarios_norm",
    "presion_sistolica_norm",
    "presion_diastolica_norm",
    "edad_norm",
    "genero_binario_mujer",
    "calidad_sueno_ordinal_norm",
    "estres_ordinal_norm",
    "categoria_imc_ordinal_norm",
]
ATRIBUTOS_LABELS = {
    "duracion_sueno_horas_norm": "Horas sueño",
    "calidad_sueno_norm": "Calidad sueño",
    "nivel_actividad_fisica_norm": "Actividad física",
    "nivel_estres_norm": "Estrés",
    "frecuencia_cardiaca_norm": "Freq. cardíaca",
    "pasos_diarios_norm": "Pasos diarios",
    "presion_sistolica_norm": "Presión sistólica",
    "presion_diastolica_norm": "Presión diastólica",
    "edad_norm": "Edad",
    "genero_binario_mujer": "Género (Mujer)",
    "calidad_sueno_ordinal_norm": "Calidad ordinal",
    "estres_ordinal_norm": "Estrés ordinal",
    "categoria_imc_ordinal_norm": "IMC ordinal",
}

COLORES_TRASTORNO = {
    "Ninguno": "#2f6f9f",
    "Insomnio": "#c94c4c",
    "Apnea del sueño": "#5c8f3f",
}


def cargar_datos():
    df_vector = pd.read_csv(VECTOR_PATH, encoding="utf-8-sig")
    df_detalles = pd.read_csv(DETALLES_PATH, encoding="utf-8-sig")
    detalle_cols = [col for col in COLUMNAS_DETALLE if col in df_detalles.columns]

    df = df_vector.merge(
        df_detalles[detalle_cols],
        on="id_persona",
        how="left",
        suffixes=("", "_detalle"),
    )
    if "trastorno_sueno_detalle" in df.columns:
        df["trastorno_sueno"] = df["trastorno_sueno_detalle"].fillna(
            df["trastorno_sueno"]
        )
        df = df.drop(columns=["trastorno_sueno_detalle"])

    feature_cols = [c for c in df_vector.columns if c not in COLUMNAS_EXCLUIR]
    X = df[feature_cols].astype(float).to_numpy()
    return df, X, feature_cols


def valor_json(valor):
    if pd.isna(valor):
        return None
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    return valor


def calcular_embedding_pca(X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.tolist()
    return coords.tolist(), var_exp


def calcular_embedding_tsne(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(X)
    return coords.tolist()


def calcular_embedding_umap(X):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X)
    return coords.tolist()


@app.route("/api/data")
def api_data():
    df, X, feature_cols = cargar_datos()

    pca_coords, var_exp = calcular_embedding_pca(X)
    tsne_coords = calcular_embedding_tsne(X)
    umap_coords = calcular_embedding_umap(X)

    histogram_features = [
        {"key": "edad_norm", "label": "Edad"},
        {"key": "duracion_sueno_horas_norm", "label": "Duración sueño"},
        {"key": "calidad_sueno_norm", "label": "Calidad sueño"},
        {"key": "nivel_estres_norm", "label": "Nivel estrés"},
        {"key": "frecuencia_cardiaca_norm", "label": "Freq. cardíaca"},
    ]

    registros = []
    for _, row in df.iterrows():
        r = {
            "id": int(row["id_persona"]),
            "id_persona": int(row["id_persona"]),
            "trastorno": valor_json(row["trastorno_sueno"]),
        }
        for col in ATRIBUTOS_BARRA:
            r[col] = float(row[col]) if col in row and not pd.isna(row[col]) else 0.0
        for col in COLUMNAS_DETALLE:
            if col in row and col not in {"id_persona", "trastorno_sueno"}:
                r[col] = valor_json(row[col])
        registros.append(r)

    bar_atributos = [
        {"key": k, "label": ATRIBUTOS_LABELS.get(k, k)}
        for k in ATRIBUTOS_BARRA
    ]

    return jsonify({
        "registros": registros,
        "feature_cols": feature_cols,
        "pca": {"coords": pca_coords, "varianza_explicada": var_exp},
        "tsne": {"coords": tsne_coords},
        "umap": {"coords": umap_coords},
        "colores_trastorno": COLORES_TRASTORNO,
        "bar_atributos": bar_atributos,
        "histogram_features": histogram_features,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
