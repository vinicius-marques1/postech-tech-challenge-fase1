# Guia de contribuição no repositório para iniciantes em Git

Este guia explica o fluxo básico para contribuir em um projeto usando Git e GitHub, de forma simples e direta.

## 1. Pré-requisitos

Antes de começar, você precisa ter instalado:

- Git
- GitHub Desktop ou terminal do Git
- Uma conta no GitHub
- A cópia do projeto em sua máquina

## 2. Configuração básica do Git

Configure seu nome e seu e-mail para que os commits sejam identificados corretamente. Use o mesmo e-mail associado à sua conta do GitHub, quando possível:

```bash
git config --global user.name "Seu Nome"
git config --global user.email "seu-email@exemplo.com"
```

Como a branch principal deste projeto se chama `main`, defina esse nome como padrão para novos repositórios:

```bash
git config --global init.defaultBranch main
```

Confira as configurações salvas:

```bash
git config --global --list
```

Essas opções são configuradas uma única vez na máquina. O parâmetro `--global` aplica a configuração a todos os repositórios do seu usuário.

## 3. Clonar o repositório

No terminal, navegue até a pasta onde deseja guardar o projeto e execute:

```bash
git clone <url-do-repositorio>
```

Exemplo:

```bash
git clone https://github.com/usuario/projeto.git
```

Depois entre na pasta do projeto:

```bash
cd projeto
```

## 4. Entender o `.gitignore`

O arquivo `.gitignore` informa ao Git quais arquivos e pastas devem ser ignorados. Ele é útil para não versionar arquivos gerados localmente, dependências, caches, credenciais ou configurações específicas da máquina.

Exemplo de conteúdo:

```gitignore
__pycache__/
.venv/
*.pyc
.env
```

Cada linha representa um padrão. Por exemplo, `*.pyc` ignora arquivos Python compilados e `.env` ignora um arquivo que pode conter variáveis de ambiente e segredos. Arquivos ignorados não são apagados da máquina; apenas deixam de aparecer como alterações novas no Git.

Para confirmar por que um arquivo está sendo ignorado, use:

```bash
git check-ignore -v caminho/do/arquivo
```

O `.gitignore` não remove do controle de versão um arquivo que já foi commitado. Nesse caso, é necessário removê-lo do índice, preservando o arquivo local:

```bash
git rm --cached caminho/do/arquivo
```

Nunca adicione senhas, tokens, chaves privadas ou outros dados sensíveis ao repositório. Se um segredo já tiver sido enviado, removê-lo do `.gitignore` não é suficiente: revogue ou troque o segredo imediatamente.

## 5. Verificar o status

Para ver o que está acontecendo no repositório, use:

```bash
git status
```

Esse comando mostra:

- arquivos alterados
- arquivos novos
- arquivos preparados para commit
- branch atual

## 6. Criar uma branch

A branch principal deste projeto é `main`. Não trabalhe diretamente nela; crie uma branch separada para cada tarefa.

```bash
git checkout -b minha-feature
```

Ou, em versões mais novas do Git:

```bash
git switch -c minha-feature
```

Exemplo de nomes de branch:

- `feature/analise-exploratoria`
- `fix/correcao-valor`
- `docs/atualiza-readme`

## 7. Fazer alterações

Edite os arquivos do projeto, crie notebooks, scripts ou documentos e depois verifique o resultado:

```bash
git status
```

## 8. Preparar arquivos para commit

Quando estiver satisfeito com as alterações, adicione os arquivos ao stage:

```bash
git add .
```

Se quiser adicionar apenas alguns arquivos:

```bash
git add README.md notebooks/meu_notebook.ipynb
```

## 9. Fazer o commit

Crie uma mensagem clara e objetiva:

```bash
git commit -m "Adiciona guia de contribuição e setup inicial"
```

Boas práticas para mensagens de commit:

- use verbos no imperativo
- descreva o que foi feito
- mantenha a mensagem curta e objetiva

Exemplos:

- `Adiciona estrutura inicial do projeto`
- `Corrige bug na limpeza de dados`
- `Atualiza instruções de setup do ambiente`

## 10. Enviar para o GitHub

Depois do commit, envie a branch para o repositório remoto:

```bash
git push origin minha-feature
```

Se a branch ainda não existir no remoto, esse comando cria automaticamente.

## 11. Abrir um Pull Request

No GitHub:

1. Vá para o repositório.
2. O GitHub geralmente mostra um aviso para criar um Pull Request.
3. Clique em "Compare & pull request".
4. Escreva um título e uma descrição clara.
5. Solicite revisão se necessário.
6. Clique em "Create pull request".

## 12. Atualizar a branch antes do merge

Se a branch principal mudou enquanto você estava trabalhando, atualize sua branch antes de abrir ou finalizar o PR:

```bash
git fetch origin
git rebase origin/main
```

Ou, se o projeto usa merge:

```bash
git pull origin main
```

## 13. Sincronizar com a branch principal

Depois que o PR for aceito e integrado à `main`, atualize sua cópia local:

```bash
git checkout main
git pull origin main
```

Se quiser continuar trabalhando em outra feature, repita o processo criando uma nova branch.

## 14. Fluxo recomendado para iniciantes

Um fluxo simples e seguro é:

```bash
git clone <url>
cd projeto
git checkout -b minha-feature
# faz alterações
git add .
git commit -m "Descreve a alteração"
git push origin minha-feature
```

Depois, abra o Pull Request no GitHub.

## 15. Dicas úteis

### Ver todos os branches

```bash
git branch -a
```

### Voltar para uma branch anterior

```bash
git checkout main
```

### Ver histórico de commits

```bash
git log --oneline
```

### Desfazer alteração em um arquivo antes do commit

```bash
git restore <arquivo>
```

### Remover arquivos do stage

```bash
git restore --staged <arquivo>
```

## 16. Conclusão

Para contribuir em um repositório, o fluxo mais importante é:

1. clonar o projeto
2. criar uma branch
3. editar arquivos
4. fazer commit
5. enviar para o GitHub
6. abrir pull request

Esse processo ajuda a manter o repositório organizado e facilita a revisão e a colaboração entre os membros do time.
