# MC859 — Grafos da Steam

Projeto da disciplina MC859 para comparar duas estruturas de relação entre jogos da Steam:

- similaridade estrutural baseada em metadados do catálogo;
- similaridade comportamental baseada em coocorrência de usuários.

## Estrutura

- `src/`: scripts reproduzíveis via CLI
- `data/raw/`: dados brutos locais, fora do versionamento
- `data/processed/`: saídas processadas e estatísticas
- `graphs/`: grafos exportados em GraphML
- `figures/`: figuras geradas para análise e relatório
- `notebooks/`: exploração complementar selecionada

## Estado do projeto

### Catálogo
- pipeline principal: `src/clean_catalog.py`
- bipartido: `graphs/game_tag.graphml`

### Comunidade
- pipeline base: `src/clean_community.py`
- pipeline enhanced: `src/build_enhanced_user_game_edges.py`
- filtro oficial da fase final:
  - `2 <= jogos_por_usuario <= 30`
  - `usuarios_por_jogo >= 10`
  - poda iterativa

## Scripts relevantes

- `src/build_bipartite_graphs.py`
- `src/analyze_bipartite_graphs.py`
- `src/plot_bipartite_degree_loglog.py`

## Notebooks mantidos

- `notebooks/00_data_audit.ipynb`: auditoria inicial dos dados brutos
- `notebooks/02_testes_limpeza.ipynb`: exploração da limpeza de tags
- `notebooks/04_user_game_filter_sensitivity.ipynb`: análise de sensibilidade dos filtros da comunidade

## Observações

- Dados brutos não devem ser versionados.
- Artefatos muito grandes da comunidade devem ficar fora do git convencional.
- Comparações entre catálogo e comunidade devem usar a interseção de jogos.
- Relatórios auxiliares em Markdown gerados a partir de notebooks não fazem parte do repositório oficial.
