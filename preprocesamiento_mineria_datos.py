from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL = BASE_DIR / "Sleep_health_and_lifestyle_dataset.csv"
DATASET_ES = BASE_DIR / "Sleep_health_dataset_es.csv"
DATASET_TRANSFORMADO = BASE_DIR / "Sleep_health_dataset_es_transformado.csv"
TABLA_CALIDAD = BASE_DIR / "tabla_calidad_datos.csv"
VECTOR_CARACTERISTICAS = BASE_DIR / "vector_caracteristicas_sleep.csv"
EMBEDDING_CSV = BASE_DIR / "embedding_sleep_2d.csv"
RESUMEN_JSON = BASE_DIR / "resumen_mineria_datos_sleep.json"
BOXPLOT_PNG = BASE_DIR / "boxplot_anomalias_sleep.png"
EMBEDDING_PNG = BASE_DIR / "embedding_sleep_2d.png"


COLUMNAS_ES = {
    "Person ID": "id_persona",
    "Gender": "genero",
    "Age": "edad",
    "Occupation": "ocupacion",
    "Sleep Duration": "duracion_sueno_horas",
    "Quality of Sleep": "calidad_sueno",
    "Physical Activity Level": "nivel_actividad_fisica",
    "Stress Level": "nivel_estres",
    "BMI Category": "categoria_imc",
    "Blood Pressure": "presion_arterial",
    "Heart Rate": "frecuencia_cardiaca",
    "Daily Steps": "pasos_diarios",
    "Sleep Disorder": "trastorno_sueno",
}

MAPEOS = {
    "genero": {"Male": "Hombre", "Female": "Mujer"},
    "ocupacion": {
        "Accountant": "Contador/a",
        "Doctor": "Doctor/a",
        "Engineer": "Ingeniero/a",
        "Lawyer": "Abogado/a",
        "Manager": "Gerente",
        "Nurse": "Enfermero/a",
        "Sales Representative": "Representante de ventas",
        "Salesperson": "Vendedor/a",
        "Scientist": "Científico/a",
        "Software Engineer": "Ingeniero/a de software",
        "Teacher": "Docente",
    },
    "categoria_imc": {
        "Normal": "Normal",
        "Normal Weight": "Peso normal",
        "Obese": "Obesidad",
        "Overweight": "Sobrepeso",
    },
    "trastorno_sueno": {
        "None": "Ninguno",
        "Insomnia": "Insomnio",
        "Sleep Apnea": "Apnea del sueño",
    },
}


def traducir_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_es = df.rename(columns=COLUMNAS_ES).copy()

    for columna, mapa in MAPEOS.items():
        df_es[columna] = df_es[columna].replace(mapa)

    df_es["trastorno_sueno"] = df_es["trastorno_sueno"].fillna("Ninguno")
    return df_es


def min_max(series: pd.Series) -> pd.Series:
    minimo = series.min()
    maximo = series.max()
    if maximo == minimo:
        return pd.Series(0.0, index=series.index)
    return (series - minimo) / (maximo - minimo)


def marcar_anomalias_iqr(df: pd.DataFrame, columnas: list[str]) -> tuple[pd.DataFrame, dict]:
    salida = df.copy()
    resumen = {}

    for columna in columnas:
        q1 = salida[columna].quantile(0.25)
        q3 = salida[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        flag = f"anomalia_{columna}"
        salida[flag] = (
            (salida[columna] < limite_inferior) | (salida[columna] > limite_superior)
        ).astype(int)
        resumen[columna] = {
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "limite_inferior": round(float(limite_inferior), 4),
            "limite_superior": round(float(limite_superior), 4),
            "anomalias": int(salida[flag].sum()),
        }

    columnas_flag = [f"anomalia_{columna}" for columna in columnas]
    salida["tiene_anomalia_iqr"] = (salida[columnas_flag].sum(axis=1) > 0).astype(int)
    return salida, resumen


def crear_transformado(df_es: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_es.copy()

    presion = df["presion_arterial"].str.split("/", expand=True).astype(int)
    df["presion_sistolica"] = presion[0]
    df["presion_diastolica"] = presion[1]

    numericas = [
        "edad",
        "duracion_sueno_horas",
        "calidad_sueno",
        "nivel_actividad_fisica",
        "nivel_estres",
        "frecuencia_cardiaca",
        "pasos_diarios",
        "presion_sistolica",
        "presion_diastolica",
    ]

    faltantes_antes = df.isna().sum().to_dict()
    for columna in numericas:
        df[columna] = df[columna].fillna(df[columna].median())
    for columna in df.select_dtypes(include="object").columns:
        moda = df[columna].mode(dropna=True)
        df[columna] = df[columna].fillna(moda.iloc[0] if not moda.empty else "Sin dato")
    faltantes_despues = df.isna().sum().to_dict()

    df["genero_binario_mujer"] = (df["genero"] == "Mujer").astype(int)
    df["tiene_trastorno_sueno"] = (df["trastorno_sueno"] != "Ninguno").astype(int)
    df["tiene_insomnio"] = (df["trastorno_sueno"] == "Insomnio").astype(int)
    df["tiene_apnea_sueno"] = (df["trastorno_sueno"] == "Apnea del sueño").astype(int)
    df["categoria_imc_limpia"] = df["categoria_imc"].replace({"Peso normal": "Normal"})

    df["calidad_sueno_nivel"] = pd.cut(
        df["calidad_sueno"],
        bins=[0, 5, 7, 10],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    ).astype(str)
    df["estres_nivel"] = pd.cut(
        df["nivel_estres"],
        bins=[0, 4, 6, 10],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    ).astype(str)

    ordinal_calidad = {"Bajo": 0, "Medio": 1, "Alto": 2}
    ordinal_estres = {"Bajo": 0, "Medio": 1, "Alto": 2}
    ordinal_imc = {"Normal": 0, "Sobrepeso": 1, "Obesidad": 2}
    df["calidad_sueno_ordinal"] = df["calidad_sueno_nivel"].map(ordinal_calidad)
    df["estres_ordinal"] = df["estres_nivel"].map(ordinal_estres)
    df["categoria_imc_ordinal"] = df["categoria_imc_limpia"].map(ordinal_imc)
    df["calidad_sueno_ordinal_norm"] = min_max(df["calidad_sueno_ordinal"]).round(6)
    df["estres_ordinal_norm"] = min_max(df["estres_ordinal"]).round(6)
    df["categoria_imc_ordinal_norm"] = min_max(df["categoria_imc_ordinal"]).round(6)

    df, resumen_anomalias = marcar_anomalias_iqr(df, numericas)

    for columna in numericas:
        df[f"{columna}_norm"] = min_max(df[columna]).round(6)

    ocupacion_dummies = pd.get_dummies(df["ocupacion"], prefix="ocupacion", dtype=int)
    df = pd.concat([df, ocupacion_dummies], axis=1)

    vector_caracteristicas = [
        "edad_norm",
        "duracion_sueno_horas_norm",
        "calidad_sueno_norm",
        "nivel_actividad_fisica_norm",
        "nivel_estres_norm",
        "frecuencia_cardiaca_norm",
        "pasos_diarios_norm",
        "presion_sistolica_norm",
        "presion_diastolica_norm",
        "genero_binario_mujer",
        "tiene_trastorno_sueno",
        "tiene_insomnio",
        "tiene_apnea_sueno",
        "calidad_sueno_ordinal_norm",
        "estres_ordinal_norm",
        "categoria_imc_ordinal_norm",
    ] + list(ocupacion_dummies.columns)

    resumen = {
        "filas": int(df.shape[0]),
        "columnas_originales": int(df_es.shape[1]),
        "columnas_transformadas": int(df.shape[1]),
        "faltantes_antes": {k: int(v) for k, v in faltantes_antes.items()},
        "faltantes_despues": {k: int(v) for k, v in faltantes_despues.items()},
        "anomalias_iqr": resumen_anomalias,
        "registros_con_alguna_anomalia": int(df["tiene_anomalia_iqr"].sum()),
        "duplicados_sin_id": int(df_es.drop(columns=["id_persona"]).duplicated().sum()),
        "vector_caracteristicas": vector_caracteristicas,
    }

    return df, resumen


def crear_tabla_calidad(
    df_original: pd.DataFrame, df_es: pd.DataFrame, resumen: dict
) -> pd.DataFrame:
    faltantes_originales = df_original.isna().sum()
    faltantes_originales.index = [
        COLUMNAS_ES.get(columna, columna) for columna in faltantes_originales.index
    ]

    especificacion = {
        "id_persona": ("Identificador", "Llave del registro", "Excluir del modelo"),
        "genero": ("Categórico binario", "Variable descriptiva", "Codificar 0/1"),
        "edad": ("Numérico", "Característica", "Normalizar 0 a 1"),
        "ocupacion": ("Texto/categórico nominal", "Característica", "One-hot encoding"),
        "duracion_sueno_horas": ("Numérico continuo", "Característica", "Normalizar 0 a 1"),
        "calidad_sueno": ("Ordinal numérico", "Característica", "Agrupar y normalizar"),
        "nivel_actividad_fisica": ("Numérico", "Característica", "Normalizar 0 a 1"),
        "nivel_estres": ("Ordinal numérico", "Característica", "Agrupar y normalizar"),
        "categoria_imc": ("Categórico ordinal", "Característica", "Limpiar y codificar"),
        "presion_arterial": ("Texto compuesto", "Característica", "Separar en sistólica y diastólica"),
        "frecuencia_cardiaca": ("Numérico", "Característica", "Normalizar y revisar IQR"),
        "pasos_diarios": ("Numérico", "Característica", "Normalizar 0 a 1"),
        "trastorno_sueno": ("Categórico nominal", "Objetivo o etiqueta", "Imputar Ninguno y binarizar"),
    }

    filas = []
    for columna, (tipo, rol, transformacion) in especificacion.items():
        faltantes = int(faltantes_originales.get(columna, 0))
        unicos = int(df_es[columna].nunique(dropna=False))
        anomalias = resumen["anomalias_iqr"].get(columna, {}).get("anomalias", 0)
        ruido = ""
        calidad = "Alta"

        if columna == "trastorno_sueno":
            ruido = "Valores vacíos semánticos: se imputan como Ninguno"
            calidad = "Media antes de imputar, alta después"
        elif columna == "categoria_imc":
            ruido = "Normal y Peso normal expresan el mismo nivel"
            calidad = "Media antes de limpiar, alta después"
        elif columna == "presion_arterial":
            ruido = "Variable compuesta en formato sistólica/diastólica"
            calidad = "Alta después de separar"
        elif anomalias > 0:
            ruido = f"{anomalias} posibles anomalías por IQR"
            calidad = "Media: revisar valores extremos"

        filas.append(
            {
                "columna": columna,
                "tipo_dato": tipo,
                "rol": rol,
                "valores_unicos": unicos,
                "faltantes_originales": faltantes,
                "anomalias_iqr": int(anomalias),
                "calidad": calidad,
                "observacion_calidad": ruido or "Sin problemas relevantes detectados",
                "tratamiento": transformacion,
            }
        )

    return pd.DataFrame(filas)


def crear_vector_y_embedding(df: pd.DataFrame, vector_columnas: list[str]) -> dict:
    matriz_vector = df[["id_persona", "trastorno_sueno"] + vector_columnas].copy()
    matriz_vector.to_csv(VECTOR_CARACTERISTICAS, index=False, encoding="utf-8-sig")

    x = matriz_vector[vector_columnas].astype(float).fillna(0.0).to_numpy()
    x_centrada = x - x.mean(axis=0)
    _, valores_singulares, vt = np.linalg.svd(x_centrada, full_matrices=False)
    coordenadas = x_centrada @ vt[:2].T

    varianza = valores_singulares**2
    varianza_explicada = varianza / varianza.sum()

    embedding = pd.DataFrame(
        {
            "id_persona": df["id_persona"],
            "embedding_1": coordenadas[:, 0].round(6),
            "embedding_2": coordenadas[:, 1].round(6),
            "trastorno_sueno": df["trastorno_sueno"],
            "tiene_trastorno_sueno": df["tiene_trastorno_sueno"],
            "calidad_sueno_nivel": df["calidad_sueno_nivel"],
            "estres_nivel": df["estres_nivel"],
        }
    )
    embedding.to_csv(EMBEDDING_CSV, index=False, encoding="utf-8-sig")

    grafico_generado = crear_grafico_embedding(embedding)

    return {
        "archivo_vector": VECTOR_CARACTERISTICAS.name,
        "archivo_embedding": EMBEDDING_CSV.name,
        "columnas_vector": len(vector_columnas),
        "varianza_explicada_embedding_1": round(float(varianza_explicada[0]), 4),
        "varianza_explicada_embedding_2": round(float(varianza_explicada[1]), 4),
        "varianza_explicada_total_2d": round(float(varianza_explicada[:2].sum()), 4),
        "grafico_embedding_generado": grafico_generado,
    }


def crear_grafico_embedding(embedding: pd.DataFrame) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    colores = {
        "Ninguno": "#2f6f9f",
        "Insomnio": "#c94c4c",
        "Apnea del sueño": "#5c8f3f",
    }

    plt.figure(figsize=(9, 6))
    for etiqueta, grupo in embedding.groupby("trastorno_sueno"):
        plt.scatter(
            grupo["embedding_1"],
            grupo["embedding_2"],
            label=etiqueta,
            alpha=0.78,
            s=42,
            color=colores.get(etiqueta, "#666666"),
        )
    plt.title("Embedding 2D del vector de características")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.legend(title="Trastorno del sueño")
    plt.tight_layout()
    plt.savefig(EMBEDDING_PNG, dpi=160)
    plt.close()
    return True


def crear_boxplot(df: pd.DataFrame) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    columnas = [
        "edad",
        "duracion_sueno_horas",
        "calidad_sueno",
        "nivel_actividad_fisica",
        "nivel_estres",
        "frecuencia_cardiaca",
        "pasos_diarios",
        "presion_sistolica",
        "presion_diastolica",
    ]

    plt.figure(figsize=(14, 7))
    df[columnas].boxplot(rot=35)
    plt.title("Boxplot para detectar valores anomalos por IQR")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=160)
    plt.close()
    return True


def main() -> None:
    df_original = pd.read_csv(ORIGINAL)
    faltantes_originales = df_original.isna().sum()
    faltantes_originales.index = [
        COLUMNAS_ES.get(columna, columna) for columna in faltantes_originales.index
    ]

    df_es = traducir_dataset(df_original)
    df_transformado, resumen = crear_transformado(df_es)
    tabla_calidad = crear_tabla_calidad(df_original, df_es, resumen)
    resumen["embedding"] = crear_vector_y_embedding(
        df_transformado, resumen["vector_caracteristicas"]
    )
    resumen["faltantes_dataset_original"] = {
        columna: int(valor) for columna, valor in faltantes_originales.items()
    }
    resumen["boxplot_generado"] = crear_boxplot(df_transformado)

    df_es.to_csv(DATASET_ES, index=False, encoding="utf-8-sig")
    df_transformado.to_csv(DATASET_TRANSFORMADO, index=False, encoding="utf-8-sig")
    tabla_calidad.to_csv(TABLA_CALIDAD, index=False, encoding="utf-8-sig")
    RESUMEN_JSON.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Dataset en español: {DATASET_ES.name}")
    print(f"Dataset transformado: {DATASET_TRANSFORMADO.name}")
    print(f"Tabla de calidad: {TABLA_CALIDAD.name}")
    print(f"Vector de características: {VECTOR_CARACTERISTICAS.name}")
    print(f"Embedding 2D: {EMBEDDING_CSV.name}")
    print(f"Resumen: {RESUMEN_JSON.name}")
    if resumen["boxplot_generado"]:
        print(f"Boxplot: {BOXPLOT_PNG.name}")


if __name__ == "__main__":
    main()
