# MC859 — Deteccao de comunidades e jogos-ponte na Steam a partir de metadados do catalogo e da comunidade de jogadores

Projeto da disciplina MC859 dedicado a comparar duas redes jogo-jogo construidas a partir de fontes distintas de relacao entre jogos da Steam:

- **catalogo**: similaridade estrutural baseada em tags e metadados;
- **comunidade**: similaridade comportamental baseada em usuarios que jogam os mesmos titulos.

O objetivo central da fase final foi verificar em que medida essas duas nocoes de proximidade produzem estruturas semelhantes ou diferentes quando observadas como grafos.

## Estrutura do repositorio

- `src/`: scripts reproduziveis via linha de comando;
- `data/processed/`: tabelas processadas e estatisticas intermediarias/finais;
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

- `src/build_community_base.py`
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

Os dados brutos nao sao versionados. Baixe o dataset
[Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
e coloque pelo menos estes arquivos em `data/raw/`:

- `games.csv`;
- `games_metadata.json`;
- `recommendations.csv`.

Instale as dependencias antes de executar o pipeline:

```bash
python -m pip install -r requirements.txt
```

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
python src/build_community_base.py --base-dir .
python src/clean_community.py --base-dir .
```

O primeiro script le `data/raw/recommendations.csv`, deduplica os pares
`(user_id, app_id)` e reproduz o pre-filtro historico de 2 a 100 jogos por
usuario. O segundo aplica os filtros oficiais da fase final sobre essa base.

Filtro oficial da fase final:

- `2 <= jogos_por_usuario <= 30`
- `usuarios_por_jogo >= 10`
- poda iterativa ate estabilizacao

Saidas principais:

- `data/processed/user_game_edges_base.csv` (intermediario local nao versionado)
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
python src/build_bipartite_graphs.py --base-dir . --force-user-game-graphml
python src/analyze_bipartite_graphs.py --base-dir .
python src/plot_bipartite_degree_loglog.py --base-dir .
```

O `user_game.graphml` ocupa aproximadamente 2 GB. Se o objetivo for apenas
reproduzir as estatisticas tabulares, omita `--force-user-game-graphml`. Para as
etapas que validam a instancia formal em GraphML, gere o arquivo com a opcao acima
ou baixe a versao publicada no Mendeley Data.

Saidas principais:

- `data/processed/bipartite_stats.json`
- `data/processed/bipartite_analysis_stats.json`

Os dois grafos bipartidos oficiais em GraphML (`game_tag.graphml` e `user_game.graphml`) foram disponibilizados externamente no Mendeley Data:

- <https://data.mendeley.com/datasets/mmy9rsng2y/1>

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

#### Onde consultar comunidades, centralidades e jogos-ponte

As comunidades, a betweenness e o PageRank foram calculados em grafos de
trabalho top-50 simetrizados por uniao: para cada jogo, o pipeline seleciona
as 50 relacoes de maior peso e preserva uma aresta quando ela e selecionada
por pelo menos uma das extremidades. A forca, por sua vez, e calculada
diretamente nas projecoes finais completas.

Os resultados podem ser consultados diretamente nos seguintes arquivos:

- `data/processed/final_projection_community_comparison.json`: parametros dos grafos de
  trabalho, modularidade, ARI e NMI;
- `data/processed/catalog_final_communities.csv` e
  `data/processed/community_final_communities.csv`:
  comunidade atribuida a cada jogo;
- `data/processed/final_projection_centrality_stats.json`: correlacoes de Spearman e
  sobreposicao dos rankings top-25 de forca, betweenness e PageRank;
- `data/processed/catalog_final_centrality_topk.csv` e
  `data/processed/community_final_centrality_topk.csv`: nomes, posicoes e valores dos 25
  jogos mais bem colocados em cada metrica.

Assim, "top-50" descreve a construcao dos grafos de trabalho, enquanto
"top-25" descreve somente a quantidade de resultados apresentados nos
rankings finais. Na execucao oficial, a sobreposicao dos top-25 foi zero
para forca, betweenness e PageRank.

#### Onde consultar os dados das tabelas do relatorio

As tabelas do relatorio podem ser conferidas a partir dos seguintes
artefatos:

- limpeza do catalogo e da comunidade:
  `data/processed/catalog_stats.json`,
  `data/processed/community_stats.json` e
  `data/processed/community_filter_summary.json`;
- dimensoes e estatisticas dos grafos bipartidos:
  `data/processed/bipartite_stats.json` e
  `data/processed/bipartite_analysis_stats.json`;
- varredura usada para escolher os thresholds:
  `data/processed/projection_threshold_sweep_table.csv`,
  `data/processed/projection_threshold_sweep_catalog_extended.csv` e
  `data/processed/projection_threshold_sweep_summary.json`;
- dimensoes das projecoes apos os cortes finais:
  `data/processed/final_filtered_projection_stats.json`;
- sobreposicao de arestas e comparacao de pesos:
  `data/processed/final_projection_comparison_stats.json` e
  `data/processed/final_projection_shared_edge_weights.csv`;
- comparacao de comunidades e centralidades: os arquivos indicados na
  secao anterior.

Os thresholds finais foram `jaccard_tags >= 0.25` para o catalogo e
`jaccard_users >= 0.002` para a comunidade. A varredura mostra a reducao do
numero de arestas e do grau medio em cada corte, juntamente com o numero de
componentes, jogos isolados e tamanho da componente gigante. Nos cortes
escolhidos, as duas projecoes preservam os 22.708 jogos em uma unica
componente conexa, sem vertices isolados. As figuras
`figures/projection_threshold_*.png` apresentam graficamente a mesma
analise.

## Artefatos finais mais importantes

Para leitura rapida da fase final, os arquivos mais importantes sao:

- `data/processed/game_game_catalog_edges_final.csv`
- `data/processed/game_game_community_edges_final.csv`
- `data/processed/final_projection_comparison_stats.json`
- `data/processed/final_projection_community_comparison.json`
- `data/processed/final_projection_centrality_stats.json`
- `data/processed/catalog_final_communities.csv`
- `data/processed/community_final_communities.csv`
- `data/processed/catalog_final_centrality_topk.csv`
- `data/processed/community_final_centrality_topk.csv`

## Mapa rapido de `data/processed`

Os artefatos de `data/processed` estao organizados, na pratica, em cinco grupos:

1. **limpeza do catalogo**
   - `game_tag_edges_filtered.csv`
   - `games_with_tags.csv`
   - `catalog_stats.json`

2. **limpeza da comunidade**
   - `user_game_edges.csv`
   - `community_stats.json`
   - `community_filter_summary.json`

3. **universo comum de jogos**
   - `common_game_ids.csv`
   - `common_game_id_map.csv`
   - `common_game_stats.json`

4. **projecoes jogo-jogo**
   - `game_game_catalog_edges.csv` / `game_game_catalog_edges_final.csv`
   - `game_game_community_edges.csv` / `game_game_community_edges_final.csv`
   - `final_filtered_projection_stats.json`
   - `projection_threshold_sweep_summary.json`

5. **comparacao final**
   - `final_projection_comparison_stats.json`
   - `final_projection_shared_edge_weights.csv`
   - `final_projection_community_comparison.json`
   - `catalog_final_communities.csv` / `community_final_communities.csv`
   - `final_projection_centrality_stats.json`
   - `catalog_final_centrality_topk.csv` / `community_final_centrality_topk.csv`

Arquivos como `*.sqlite`, e tabelas exploratorias adicionais foram mantidos no repositorio como apoio de implementacao e validacao, mas nao sao os artefatos centrais dos resultados finais.

## Observacoes sobre os artefatos versionados

- Os **grafos bipartidos oficiais** foram exportados em GraphML
- Eles podem ser obtidos no Mendeley Data: <https://data.mendeley.com/datasets/mmy9rsng2y/1>.
- As **projecoes finais** foram mantidas principalmente como **listas de arestas em CSV** e estatisticas em JSON.

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
