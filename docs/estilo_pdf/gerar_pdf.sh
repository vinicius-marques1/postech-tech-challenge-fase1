#!/usr/bin/env bash
# Gera docs/relatorio_tecnico.pdf a partir de docs/relatorio_tecnico.md.
# Requisitos: uv, xelatex e as fontes Lato e Ubuntu Mono instaladas.
# Uso (a partir da raiz do projeto): bash docs/estilo_pdf/gerar_pdf.sh
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --no-project --with pypandoc_binary python - <<'EOF'
import pypandoc

pypandoc.convert_file(
    "relatorio_tecnico.md",
    "pdf",
    outputfile="relatorio_tecnico.pdf",
    extra_args=[
        "--pdf-engine=xelatex",
        "--lua-filter=estilo_pdf/filtros.lua",
        "-H", "estilo_pdf/estilo.tex",
        "-V", "geometry:top=2.3cm,bottom=2.2cm,left=2.3cm,right=2.3cm,headsep=0.8cm",
        "-V", "fontsize=10pt",
        "-V", "classoption=table",
        "-V", "lang=pt-BR",
        "-V", "colorlinks=true",
        "-V", "linkcolor=link",
        "-V", "urlcolor=link",
        "-V", "toccolor=primaria",
    ],
)
print("PDF gerado em docs/relatorio_tecnico.pdf")
EOF
