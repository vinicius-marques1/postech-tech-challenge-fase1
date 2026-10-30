# Models

Esta pasta tem como objetivo armazenar os artefatos gerados pelos experimentos de machine learning, como modelos treinados, métricas relevantes e arquivos auxiliares usados para inferência ou reuso.

## Finalidade

- guardar modelos finais e intermediários
- preservar versões de treinamento
- facilitar a comparação entre experimentos
- permitir reutilização do modelo em testes, APIs ou avaliação posterior

## Estrutura recomendada

```text
models/
├── README.md
├── baseline/
│   └── modelo_baseline.joblib
├── tuned/
│   └── logisticregression_tuned.joblib
├── metrics/
│   └── metricas_validacao.json
└── artifacts/
    └── preprocessador.joblib
```

## Boas práticas

- use nomes descritivos e versionados
- mantenha o modelo junto com metadados relevantes
- salve também o preprocessador, quando houver
- registre métricas e parâmetros usados no treinamento
- evite salvar arquivos muito grandes no controle de versão, quando a equipe preferir manter artefatos em armazenamento externo

## Exemplos de nomeação

- `logistic_regression_v1.joblib`
- `random_forest_baseline.joblib`
- `modelo_classificador_tuned.pkl`
- `preprocessador_pipeline.joblib`

## Recomendação para o projeto

O notebook `notebooks/03_modelagem.ipynb` registra o modelo final e suas métricas. Para reproduzir o treinamento sem executar o notebook, use:

```bash
uv run python src/treinar_modelo.py
```

O script lê os hiperparâmetros, o limiar e os parâmetros de divisão de `metrics/metricas_validacao.json`, treina o pipeline com a mesma divisão estratificada e salva `tuned/logisticregression_tuned.joblib`. O artefato Joblib inclui o pipeline (pré-processamento e classificador), a ordem das features, o limiar, os dois perfis fictícios e 50 exemplos de teste com seus rótulos verdadeiros para a demonstração de acurácia.

Para executar a demonstração de previsões:

```bash
uv run python src/prever_exemplo.py
```

Esse script usa dois perfis fictícios construídos a partir das medianas por classe do conjunto de treino e apresenta também previsões para 50 exames do conjunto de teste preservado. Para esses exames, exibe o rótulo verdadeiro, o previsto, a probabilidade de malignidade e a acurácia total da amostra. Os 50 exames não participam do treinamento. Os perfis fictícios servem apenas para demonstrar o fluxo; nenhum resultado constitui diagnóstico médico.

Os demais modelos gerados em notebooks ou scripts podem ser salvos aqui com a convenção:

```python
import joblib

joblib.dump(modelo, "models/baseline/modelo_baseline.joblib")
```

Se houver uma etapa de pré-processamento, também vale salvar separadamente:

```python
joblib.dump(preprocessador, "models/artifacts/preprocessador.joblib")
```

## Observação

Essa pasta deve ser usada para artefatos de produção e experimentação local, mantendo um histórico simples e organizado do que foi treinado e validado. Os artefatos de modelo são ignorados pelo Git; as métricas registradas em `metrics/` permanecem como referência reproduzível.
