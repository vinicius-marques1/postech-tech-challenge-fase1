"""Recria o modelo final do notebook a partir da configuração registrada."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "metrics" / "metricas_validacao.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "breast_cancer_wisconsin_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tuned" / "logisticregression_tuned.joblib"
TARGET = "diagnosis"


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    if not isinstance(metrics, dict):
        raise ValueError(f"O arquivo de métricas precisa conter um objeto JSON: {path}")
    return metrics


def main() -> None:
    metrics = load_metrics(METRICS_PATH)
    model_name = metrics["modelo"]
    pipeline_config = metrics["pipeline"]
    split_config = metrics["reproducao"]["divisao"]
    random_state = metrics["reproducao"]["random_state"]
    threshold = float(metrics["decisao"]["limiar"])

    if model_name != "LogisticRegression":
        raise ValueError(f"Modelo não suportado por este script: {model_name}")
    if pipeline_config["classificador"] != "LogisticRegression":
        raise ValueError("O classificador do JSON não corresponde ao modelo selecionado.")
    if pipeline_config["scaler"] != "RobustScaler":
        raise ValueError("Este script espera RobustScaler conforme o JSON de métricas.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"O limiar deve estar entre 0 e 1; recebido: {threshold}")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Base tratada não encontrada: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"A base precisa conter a coluna alvo {TARGET!r}.")
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("A coluna diagnosis deve conter somente os rótulos 0 e 1, sem valores ausentes.")

    features = [column for column in data.columns if column != TARGET]
    if not features or data[features].isna().any().any():
        raise ValueError("As features devem existir e não podem conter valores ausentes.")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in features):
        raise ValueError("Todas as features precisam ser numéricas.")

    x_train, x_test, y_train, y_test = train_test_split(
        data[features],
        data[TARGET],
        test_size=split_config["proporcao_teste"],
        random_state=random_state,
        stratify=data[TARGET] if split_config["estratificada"] else None,
    )
    expected_train = split_config["n_treino"]
    expected_test = split_config["n_teste"]
    if (len(x_train), len(x_test)) != (expected_train, expected_test):
        raise ValueError(
            "As dimensões da base não correspondem à divisão registrada nas métricas: "
            f"esperado {expected_train}/{expected_test}, obtido {len(x_train)}/{len(x_test)}."
        )
    if len(x_test) < 50:
        raise ValueError(f"O conjunto de teste precisa ter pelo menos 50 exames; obtido: {len(x_test)}.")

    hyperparameters = pipeline_config["hiperparametros_ajustados"]
    supported_parameters = LogisticRegression().get_params()
    unknown_parameters = set(hyperparameters) - supported_parameters.keys()
    if unknown_parameters:
        raise ValueError(f"Hiperparâmetros não reconhecidos: {sorted(unknown_parameters)}")

    classifier = LogisticRegression(
        max_iter=10000,
        random_state=random_state,
        **hyperparameters,
    )
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    training_data = x_train.copy()
    training_data[TARGET] = y_train
    demo_examples = (
        training_data.groupby(TARGET, sort=True)[features]
        .median()
        .reindex([0, 1])
    )
    demo_examples.index = ["amostra_sintetica_A", "amostra_sintetica_B"]
    test_examples = x_test.sample(n=50, random_state=random_state)
    test_labels = y_test.loc[test_examples.index].copy()

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_names": features,
        "class_names": {0: "Benigno", 1: "Maligno"},
        "demo_examples": demo_examples,
        "test_examples": test_examples,
        "test_labels": test_labels,
        "metadata": {
            "model": model_name,
            "scaler": pipeline_config["scaler"],
            "hyperparameters": hyperparameters,
            "random_state": random_state,
            "training_rows": len(x_train),
            "held_out_test_rows": len(x_test),
            "test_demo_rows": len(test_examples),
            "demo_profile_method": "medianas por classe calculadas somente no conjunto de treino",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Modelo treinado: {model_name} com {pipeline_config['scaler']}")
    print(f"Linhas de treino: {len(x_train)}; linhas de teste preservadas: {len(x_test)}")
    print(f"Limiar de decisão: {threshold:.2f}")
    print(f"Artefato salvo em: {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
