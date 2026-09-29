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
│   └── modelo_tuned.joblib
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

Os modelos gerados em notebooks ou scripts podem ser salvos aqui com a convenção:

```python
import joblib

joblib.dump(modelo, "models/baseline/modelo_baseline.joblib")
```

Se houver uma etapa de pré-processamento, também vale salvar separadamente:

```python
joblib.dump(preprocessador, "models/artifacts/preprocessador.joblib")
```

## Observação

Essa pasta deve ser usada para artefatos de produção e experimentação local, mantendo um histórico simples e organizado do que foi treinado e validado. Ela não é versionada no git.
