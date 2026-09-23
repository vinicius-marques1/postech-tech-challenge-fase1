# Tech Challenge Fase 1

## Dados

O projeto usa o dataset **Breast Cancer Wisconsin (Diagnostic)**, com 569 exames de punção aspirativa por agulha fina classificados como tumor maligno ou benigno.

- Fonte primária: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), licença CC BY 4.0.
- Cópia utilizada: [Kaggle (uciml/breast-cancer-wisconsin-data)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), salva em `data/raw/breast_cancer_wisconsin.csv`.
- Referência: Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis*.

A inspeção inicial da base está em `notebooks/01_data_inspection.ipynb`.

## Setup do ambiente de trabalho

Este projeto usa Python e bibliotecas de ciência de dados e machine learning. O ambiente pode ser configurado de duas formas: com uv (recomendado) ou com pip tradicional.

### Opção recomendada: uv

O `uv` é um gerenciador de ambientes e dependências mais moderno e rápido. Para quem está começando, a instalação é simples:

link da documentação do UV: https://docs.astral.sh/uv/

#### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Depois, reinicie o terminal ou execute:

```bash
source $HOME/.cargo/env
```

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Verificar instalação

```bash
uv --version
```

No diretório do projeto:

```bash
uv sync
source .venv/bin/activate
```

No Windows, use:

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
```

Se o ambiente ainda não existir, o comando acima cria o ambiente virtual e instala as dependências definidas em `pyproject.toml`.

#### Comandos úteis do uv

Adicionar uma nova dependência ao projeto:

```bash
uv add nome-do-pacote
```

Adicionar uma dependência apenas para desenvolvimento:

```bash
uv add --dev nome-do-pacote
```

Atualizar todas as dependências do projeto:

```bash
uv lock --upgrade
```

Sincronizar o ambiente com o arquivo de lock:

```bash
uv sync
```

Ver todas as dependências instaladas no ambiente atual:

```bash
uv pip list
```

### Opção alternativa: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> O comando `pip install -e .` usa o arquivo `pyproject.toml` como fonte das dependências principais do projeto.

**Recomendo fortemente usar o UV porque ele é muito mais rapido!**

## Criando notebooks

A pasta `notebooks/` do projeto é o local destinado para exploração de dados, treinamento e validação de modelos, comparação de experimentos, análise de métricas e registro de resultados.

### No VS Code

1. Abra a pasta do projeto no VS Code.
2. Crie os notebooks dentro de `notebooks/`.
3. No prompt de seleção de kernel, escolha o Python do ambiente virtual criado acima (`.venv`).
4. Use arquivos `.ipynb` para explorar dados, preparar features, treinar modelos, validar desempenho, comparar experimentos e registrar resultados.

### No navegador

Com o ambiente ativado, execute:

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

Isso abre o Jupyter no navegador e permite criar e executar notebooks diretamente no projeto.

## Se o notebook for criado primeiro no Colab

Quando o trabalho começar no Google Colab, instale primeiro as dependências do projeto:

```python
!pip install -U pip
!pip install joblib jupyterlab matplotlib numpy pandas scikit-learn scipy seaborn shap
```

Se for necessário trabalhar no entregável extra de visão computacional, também instale as dependências opcionais:

```python
!pip install -r requirements-extra.txt
```

Depois disso, o notebook pode ser exportado para o repositório com a extensão `.ipynb` e você continua editando ele localmente no VS Code ou no Jupyter.

## Dependências extras para visão computacional

O arquivo `requirements-extra.txt` deve ser usado somente para a tarefa extra de visão computacional / CNN. Ele não faz parte do ambiente principal do projeto e deve ser instalado apenas quando essa etapa for necessária.

Uso:

```bash
pip install -r requirements-extra.txt
```

ou, com uv:

```bash
uv pip install -r requirements-extra.txt
```

> Mantém o ambiente principal limpo e separa as dependências opcionais da etapa extra.

## Guia de contribuição

Para quem está começando no Git, consulte o guia completo em [docs/guia-contribuicao-git.md](docs/guia-contribuicao-git.md).
