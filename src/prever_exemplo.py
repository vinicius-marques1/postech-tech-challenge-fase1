"""Demonstra previsões do modelo treinado usando perfis fictícios."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "tuned" / "logisticregression_tuned.joblib"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Artefato do modelo não encontrado: {MODEL_PATH}\n"
            "Execute primeiro: uv run python src/treinar_modelo.py"
        )

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError("O artefato do modelo tem formato inválido.")

    pipeline = artifact.get("pipeline")
    feature_names = artifact.get("feature_names")
    class_names = artifact.get("class_names")
    examples = artifact.get("demo_examples")
    test_examples = artifact.get("test_examples")
    test_labels = artifact.get("test_labels")
    threshold = artifact.get("threshold")

    if not isinstance(pipeline, Pipeline):
        raise ValueError("O artefato não contém um pipeline scikit-learn válido.")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("O artefato não contém a lista de features esperada.")
    if not isinstance(class_names, dict) or set(class_names) != {0, 1}:
        raise ValueError("O artefato não contém os nomes de classe esperados.")
    if not isinstance(examples, pd.DataFrame):
        raise ValueError("O artefato não contém os perfis fictícios para demonstração.")
    if not isinstance(test_examples, pd.DataFrame) or len(test_examples) != 50:
        raise ValueError("O artefato precisa conter exatamente 50 exames de teste.")
    if not isinstance(test_labels, pd.Series) or not test_labels.index.equals(test_examples.index):
        raise ValueError("Os rótulos verdadeiros não correspondem aos exames de teste.")
    if "diagnosis" in examples.columns:
        raise ValueError("Os exemplos de inferência não devem incluir a coluna diagnosis.")
    if set(examples.columns) != set(feature_names):
        raise ValueError("As features dos perfis fictícios não correspondem às do modelo.")
    if "diagnosis" in test_examples.columns:
        raise ValueError("Os exames de teste não devem incluir a coluna diagnosis nas features.")
    if set(test_examples.columns) != set(feature_names):
        raise ValueError("As features dos exames de teste não correspondem às do modelo.")
    if test_labels.isna().any() or not set(test_labels.unique()).issubset({0, 1}):
        raise ValueError("Os rótulos verdadeiros dos exames de teste devem ser 0 ou 1.")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("O limiar armazenado no artefato é inválido.")

    positive_class_index = list(pipeline.classes_).index(1)
    demo_inputs = examples.loc[:, feature_names]
    demo_probabilities = pipeline.predict_proba(demo_inputs)[:, positive_class_index]
    demo_predictions = (demo_probabilities >= threshold).astype(int)

    demo_results = pd.DataFrame(
        {
            "probabilidade_malignidade": demo_probabilities,
            "limiar": threshold,
            "previsao": [class_names[int(label)] for label in demo_predictions],
        },
        index=examples.index,
    )
    demo_results.index.name = "exemplo_ficticio"

    print("Previsões demonstrativas:")
    print(demo_results.to_string(float_format=lambda value: f"{value:.3f}"))

    test_inputs = test_examples.loc[:, feature_names]
    test_probabilities = pipeline.predict_proba(test_inputs)[:, positive_class_index]
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_results = pd.DataFrame(
        {
            "y_verdadeiro": [class_names[int(label)] for label in test_labels],
            "y_previsto": [class_names[int(label)] for label in test_predictions],
            "probabilidade_malignidade": test_probabilities,
            "acerto": test_labels.to_numpy() == test_predictions,
        },
        index=test_examples.index,
    )
    test_results.index.name = "indice_teste"

    print(f"\nPrevisões nos {len(test_examples)} exames reservados para teste:")
    print(test_results.to_string(float_format=lambda value: f"{value:.3f}"))

    correct = int(test_results["acerto"].sum())
    accuracy = correct / len(test_results)
    print(f"\nAcurácia nos 50 exames: {accuracy:.1%} ({correct}/{len(test_results)})")
    print(
        "\nOs perfis são fictícios e foram sintetizados a partir das medianas "
        "do conjunto de treino. Os exames de teste permaneceram fora do treinamento. "
        "Esta demonstração é educacional e não constitui "
        "diagnóstico médico."
    )


if __name__ == "__main__":
    main()
