"""
Pipeline de Machine Learning para clasificacion de trastornos del sueno.
Metodologia del paper IEEE:
  1. EDA - distribucion de clases (desbalance)
  2. Preprocesamiento: eliminar Person ID, LabelEncoder, 80/20, SMOTEENN
  3. Seleccion de variables: GradientBoostingRegressor, Top 5
  4. Entrenamiento: GradientBoostingClassifier
  5. Evaluacion: matriz de confusion, Accuracy, Precision, Recall, F1, AUC
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_PATH = BASE_DIR / "vector_caracteristicas_sleep.csv"
DETALLES_PATH = BASE_DIR / "Sleep_health_dataset_es_transformado.csv"
RESULTADOS_PATH = BASE_DIR / "resultados_ml.json"
CM_PNG = BASE_DIR / "static" / "matriz_confusion.png"
FEATURES_PNG = BASE_DIR / "static" / "feature_importance.png"
DISTRIBUCION_PNG = BASE_DIR / "static" / "distribucion_clases.png"

COLUMNAS_EXCLUIR = {"id_persona", "trastorno_sueno", "tiene_trastorno_sueno", "tiene_insomnio", "tiene_apnea_sueno"}
TARGET = "trastorno_sueno"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_FEATURES = 5


COLUMNAS_DETALLE = [
    "id_persona", "genero", "edad", "ocupacion", "duracion_sueno_horas",
    "calidad_sueno", "nivel_actividad_fisica", "nivel_estres", "categoria_imc",
    "presion_arterial", "frecuencia_cardiaca", "pasos_diarios", TARGET,
    "calidad_sueno_nivel", "estres_nivel", "tiene_anomalia_iqr",
]


def eda_cargar_datos() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(VECTOR_PATH, encoding="utf-8-sig")
    feature_cols = [c for c in df.columns if c not in COLUMNAS_EXCLUIR]
    X = df[feature_cols].astype(float).to_numpy()
    return df, X, feature_cols


def eda_distribucion(df: pd.DataFrame) -> dict:
    counts = df[TARGET].value_counts()
    plt.figure(figsize=(8, 5))
    colores = {"Ninguno": "#2f6f9f", "Insomnio": "#c94c4c", "Apnea del sueno": "#5c8f3f"}
    bars = plt.bar(
        counts.index,
        counts.values,
        color=[colores.get(c, "#888") for c in counts.index],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            str(int(bar.get_height())),
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    plt.title("Distribucion de Trastornos del Sueno", fontsize=14, fontweight="bold")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de personas")
    plt.tight_layout()
    plt.savefig(DISTRIBUCION_PNG, dpi=160)
    plt.close()

    total = len(df)
    proporcion = {k: round(v / total, 4) for k, v in counts.items()}
    return {
        "total_muestras": int(total),
        "conteo_clases": counts.to_dict(),
        "proporcion_clases": proporcion,
        "clase_mayoritaria": str(counts.idxmax()),
        "clase_minoritaria": str(counts.idxmin()),
        "desbalance_ratio": round(counts.max() / counts.min(), 2),
    }


def ejecutar_pipeline() -> dict:
    print("Cargando datos...")
    df, X, feature_cols = eda_cargar_datos()

    print("EDA - distribucion de clases...")
    resumen_eda = eda_distribucion(df)
    print(f"  Clases: {resumen_eda['conteo_clases']}")
    print(f"  Ratio de desbalance: {resumen_eda['desbalance_ratio']}")

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    clases = le.classes_.tolist()

    print("Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("Aplicando SMOTEENN (balanceo + limpieza)...")
    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    X_res, y_res = smoteenn.fit_resample(X_train, y_train)
    clases_balanceadas = pd.Series(y_res).value_counts().to_dict()
    print(f"  Clases despues de SMOTEENN: {clases_balanceadas}")

    print("Seleccion de variables - GradientBoostingRegressor (Top 5)...")
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE
    )
    gbr.fit(X_res, y_res)
    importancias = gbr.feature_importances_
    indices_top5 = np.argsort(importancias)[::-1][:TOP_N_FEATURES]
    top5_features = [
        {"posicion": i + 1, "feature": feature_cols[idx], "importancia": round(float(importancias[idx]), 4)}
        for i, idx in enumerate(indices_top5)
    ]
    top5_cols = [feature_cols[idx] for idx in indices_top5]
    print(f"  Top 5: {[f['feature'] for f in top5_features]}")

    plt.figure(figsize=(9, 5))
    plt.barh(
        [f["feature"] for f in top5_features[::-1]],
        [f["importancia"] for f in top5_features[::-1]],
        color="#2f9c95",
        edgecolor="white",
    )
    plt.xlabel("Importancia")
    plt.title("Top 5 Variables - GradientBoostingRegressor", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FEATURES_PNG, dpi=160)
    plt.close()

    X_res_top5 = X_res[:, indices_top5]
    X_test_top5 = X_test[:, indices_top5]

    print("Entrenando GradientBoostingClassifier...")
    gbc = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE
    )
    gbc.fit(X_res_top5, y_res)

    print("Evaluando...")
    y_pred = gbc.predict(X_test_top5)
    y_prob = gbc.predict_proba(X_test_top5)

    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    precision = round(
        float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4
    )
    recall = round(
        float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4
    )
    f1 = round(float(f1_score(y_test, y_pred, average="weighted")), 4)

    try:
        auc = round(
            float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")),
            4,
        )
    except Exception:
        auc = None

    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-Score: {f1}")
    print(f"  AUC: {auc}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusion - GradientBoosting", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=160)
    plt.close()

    resultados = {
        "eda": resumen_eda,
        "preprocesamiento": {
            "features_originales": len(feature_cols),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "smoteenn_clases_resultantes": {
                str(k): int(v) for k, v in clases_balanceadas.items()
            },
        },
        "seleccion_variables": {
            "metodo": "GradientBoostingRegressor",
            "top_n": TOP_N_FEATURES,
            "features": top5_features,
        },
        "modelo": {
            "tipo": "GradientBoostingClassifier",
            "hiperparametros": {"n_estimators": 200, "max_depth": 3},
        },
        "evaluacion": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "matriz_confusion": cm.tolist(),
            "etiquetas_clases": clases,
            "reporte_clasificacion": classification_report(
                y_test, y_pred, target_names=clases, output_dict=True, zero_division=0
            ),
        },
    }

    RESULTADOS_PATH.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados guardados en: {RESULTADOS_PATH.name}")
    print(f"Graficos generados:")
    print(f"  - {DISTRIBUCION_PNG.name}")
    print(f"  - {FEATURES_PNG.name}")
    print(f"  - {CM_PNG.name}")

    return resultados


if __name__ == "__main__":
    ejecutar_pipeline()
