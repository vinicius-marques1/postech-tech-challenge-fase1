-- Filtros do pandoc para o PDF do relatório:
--   * troca o título padrão por uma capa (title, subtitle, date e author do YAML)
--     seguida do sumário;
--   * transforma citações (> ...) em caixas de destaque.

local abertura = nil

function Meta(meta)
  if meta.title then
    local titulo = pandoc.utils.stringify(meta.title)
    local subtitulo = meta.subtitle and pandoc.utils.stringify(meta.subtitle) or ""
    local data = meta.date and pandoc.utils.stringify(meta.date) or ""
    local nomes = {}
    if meta.author then
      for _, a in ipairs(meta.author) do
        table.insert(nomes, pandoc.utils.stringify(a))
      end
    end
    abertura = pandoc.RawBlock("latex",
      string.format("\\capa{%s}{%s}{%s}{%s}\n\\sumario",
        titulo, subtitulo, data, table.concat(nomes, "\\\\")))
    meta.pagetitle = meta.title
    meta.title = nil
    meta.subtitle = nil
    meta.date = nil
    meta.author = nil
  end
  return meta
end

function BlockQuote(el)
  local blocks = { pandoc.RawBlock("latex", "\\begin{destaque}") }
  for _, b in ipairs(el.content) do table.insert(blocks, b) end
  table.insert(blocks, pandoc.RawBlock("latex", "\\end{destaque}"))
  return blocks
end

function Pandoc(doc)
  if abertura then table.insert(doc.blocks, 1, abertura) end
  return doc
end

return {
  { Meta = Meta },
  { BlockQuote = BlockQuote, Pandoc = Pandoc },
}
