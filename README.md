# MC859 — Deteccao de comunidades e jogos-pontes na Steam a partir de metadados do catalogo e da comunidade de jogadores

Projeto da disciplina MC859 dedicado a comparar duas redes jogo-jogo construidas a partir de fontes distintas de relacao entre jogos da Steam:

- **catalogo**: similaridade estrutural baseada em tags e metadados;
- **comunidade**: similaridade comportamental baseada em usuarios que jogam os mesmos titulos.

O objetivo central da fase final foi verificar em que medida essas duas nocoes de proximidade produzem estruturas semelhantes ou diferentes quando observadas como grafos.

## Estrutura do repositorio

- `src/`: scripts reproduziveis via linha de comando;
- `data/raw/`: dados brutos locais, fora do versionamento;
- `data/processed/`: tabelas processadas e estatisticas intermediarias/finais;
- `graphs/`: grafos bipartidos exportados em GraphML;
- `figures/`: figuras geradas para analise e relatorio;
- `notebooks/`: exploracao complementar mantida apenas quando util ao projeto.

## Pipeline adotado

A execucao do projeto foi organizada em seis etapas principais.

1. limpeza do catalogo;
2. limpeza da comunidade;
3. construcao do universo comum de jogos;
4. construcao e analise dos grafos bipartidos;
5. construcao das projecoes jogo-jogo;
6. comparacao final entre as projecoes.

## Scripts principais

### Limpeza e instancias bipartidas

- `src/clean_catalog.py`
- `src/clean_community.py`
- `src/build_common_game_universe.py`
- `src/build_bipartite_graphs.py`
- `src/analyze_bipartite_graphs.py`
- `src/plot_bipartite_degree_loglog.py`

### Projecoes jogo-jogo

- `src/build_catalog_projection.py`
- `src/build_community_projection.py`
- `src/sweep_projection_thresholds.py`
- `src/build_final_filtered_projections.py`

### Comparacao final

- `src/compare_final_projections.py`
- `src/detect_projection_communities.py`
- `src/analyze_projection_centrality.py`
- `src/analyze_projected_graphs.py`

## Como executar

Os scripts foram escritos para serem executados a partir da raiz do repositorio.

### 1. Limpeza do catalogo

```bash
python src/clean_catalog.py --base-dir .
```

Saidas principais:

- `data/processed/game_tag_edges_filtered.csv`
- `data/processed/games_with_tags.csv`
- `data/processed/catalog_stats.json`

### 2. Limpeza da comunidade

```bash
python src/clean_community.py --base-dir .
```

Filtro oficial da fase final:

- `2 <= jogos_por_usuario <= 30`
- `usuarios_por_jogo >= 10`
- poda iterativa ate estabilizacao

Saidas principais:

- `data/processed/user_game_edges.csv`
- `data/processed/community_stats.json`
- `data/processed/community_filter_summary.json`

### 3. Universo comum de jogos

```bash
python src/build_common_game_universe.py --base-dir .
```

Saidas principais:

- `data/processed/common_game_ids.csv`
- `data/processed/common_game_id_map.csv`
- `data/processed/common_game_stats.json`

Todas as comparacoes entre catalogo e comunidade na fase final usam esse universo comum.

### 4. Grafos bipartidos

```bash
python src/build_bipartite_graphs.py --base-dir .
python src/analyze_bipartite_graphs.py --base-dir .
python src/plot_bipartite_degree_loglog.py --base-dir .
```

Saidas principais:

- `graphs/game_tag.graphml`
- `graphs/user_game.graphml`
- `data/processed/bipartite_stats.json`
- `data/processed/bipartite_analysis_stats.json`

### 5. Projecoes jogo-jogo

```bash
python src/build_catalog_projection.py --base-dir .
python src/build_community_projection.py --base-dir .
python src/sweep_projection_thresholds.py --base-dir .
python src/build_final_filtered_projections.py --base-dir .
```

Saidas principais:

- `data/processed/game_game_catalog_edges.csv`
- `data/processed/game_game_community_edges.csv`
- `data/processed/game_game_catalog_edges_final.csv`
- `data/processed/game_game_community_edges_final.csv`
- `data/processed/final_filtered_projection_stats.json`
- `data/processed/projection_threshold_sweep_summary.json`

### 6. Comparacao final entre as projecoes

```bash
python src/compare_final_projections.py --base-dir .
python src/detect_projection_communities.py --base-dir .
python src/analyze_projection_centrality.py --base-dir .
```

Saidas principais:

- `data/processed/final_projection_comparison_stats.json`
- `data/processed/final_projection_shared_edge_weights.csv`
- `data/processed/final_projection_community_comparison.json`
- `data/processed/catalog_final_communities.csv`
- `data/processed/community_final_communities.csv`
- `data/processed/final_projection_centrality_stats.json`
- `data/processed/catalog_final_centrality_topk.csv`
- `data/processed/community_final_centrality_topk.csv`

## Artefatos finais mais importantes

Para leitura rapida da fase final, os arquivos mais importantes sao:

- `data/processed/game_game_catalog_edges_final.csv`
- `data/processed/game_game_community_edges_final.csv`
- `data/processed/final_projection_comparison_stats.json`
- `data/processed/final_projection_community_comparison.json`
- `data/processed/final_projection_centrality_stats.json`
- `data/processed/catalog_final_communities.csv`
- `data/processed/community_final_communities.csv`

## Observacoes sobre os artefatos versionados

- Os **grafos bipartidos oficiais** foram exportados em GraphML e estao em `graphs/`.
- As **projecoes finais** foram mantidas principalmente como **listas de arestas em CSV** e estatisticas em JSON.
- Optou-se por nao versionar GraphMLs das projecoes finais porque as listas de arestas ja sao os artefatos oficiais reproduziveis usados em toda a analise, com menor redundancia e custo de armazenamento.
- Os dados brutos originais e alguns artefatos historicos de apoio ficam fora do versionamento normal.

## Metricas usadas na fase final

A comparacao entre catalogo e comunidade foi baseada em quatro eixos principais:

- sobreposicao de arestas;
- correlacao entre pesos das arestas compartilhadas;
- comparacao de comunidades;
- comparacao de centralidades e jogos-ponte.

Na fase final, as principais metricas calculadas foram:

- similaridade de Jaccard entre conjuntos de arestas;
- correlacoes de Spearman e Pearson nos pesos compartilhados;
- ARI e NMI para comunidades;
- rankings de forca, betweenness e PageRank.

## Dependencias

O projeto usa bibliotecas cientificas padrao do ecossistema Python, incluindo:

- `numpy`
- `pandas`
- `networkx`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `igraph`
- `leidenalg`

## Observacoes finais

- Dados brutos nao devem ser versionados neste repositorio.
- Comparacoes entre catalogo e comunidade devem sempre respeitar a intersecao de jogos.
- Scripts auxiliares de exploracao visual e preparacao de apresentacao nao fazem parte do pipeline oficial documentado aqui.
