from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter
from pathlib import Path

import igraph as ig
import numpy as np
from scipy.stats import spearmanr


DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_NAME_MAP_INPUT = Path("data/processed/common_game_id_map.csv")
DEFAULT_CATALOG_INPUT = Path("data/processed/game_game_catalog_edges_final.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/game_game_community_edges_final.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/final_projection_centrality_stats.json")
DEFAULT_CATALOG_TOP_OUTPUT = Path("data/processed/catalog_final_centrality_topk.csv")
DEFAULT_COMMUNITY_TOP_OUTPUT = Path("data/processed/community_final_centrality_topk.csv")
DEFAULT_TOP_K = 50
DEFAULT_SELECTION_MODE = "union"
DEFAULT_REPORT_TOP_K = 25
DEFAULT_PAGERANK_DAMPING = 0.85


class DisjointSet:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        counts: Counter[int] = Counter()
        for value in self.parent:
            counts[self.find(value)] += 1
        return sorted(counts.values(), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa centralidade e jogos-ponte nas projecoes finais filtradas. "
            "Forca e calculada nas redes finais; betweenness e PageRank nos grafos de trabalho top-k."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--common-ids-input", type=Path, default=DEFAULT_COMMON_IDS_INPUT)
    parser.add_argument("--name-map-input", type=Path, default=DEFAULT_NAME_MAP_INPUT)
    parser.add_argument("--catalog-input", type=Path, default=DEFAULT_CATALOG_INPUT)
    parser.add_argument("--community-input", type=Path, default=DEFAULT_COMMUNITY_INPUT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--catalog-top-output", type=Path, default=DEFAULT_CATALOG_TOP_OUTPUT)
    parser.add_argument("--community-top-output", type=Path, default=DEFAULT_COMMUNITY_TOP_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--selection-mode", choices=("union", "mutual"), default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--report-top-k", type=int, default=DEFAULT_REPORT_TOP_K)
    parser.add_argument("--pagerank-damping", type=float, default=DEFAULT_PAGERANK_DAMPING)
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def validate_required_columns(fieldnames: list[str] | None, required: tuple[str, ...]) -> None:
    if fieldnames is None:
        raise AssertionError("CSV sem cabecalho")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise AssertionError(f"CSV precisa conter as colunas {required}; faltando={missing}")


def load_common_game_ids(input_path: Path) -> list[int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    ids: list[int] = []
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id",))
        for row in reader:
            ids.append(int(str(row["app_id"]).strip()))
    if not ids:
        raise AssertionError("A lista de jogos comuns nao pode ser vazia")
    return sorted(ids)


def load_name_map(input_path: Path) -> dict[int, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    names: dict[int, str] = {}
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id", "name"))
        for row in reader:
            names[int(str(row["app_id"]).strip())] = str(row["name"]).strip()
    return names


def update_heap(heap: list[tuple[float, int]], weight: float, neighbor: int, k: int) -> None:
    item = (weight, neighbor)
    if len(heap) < k:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def build_selected_edge_set(
    input_path: Path,
    common_id_set: set[int],
    top_k: int,
    selection_mode: str,
    weight_column: str,
) -> set[tuple[int, int]]:
    top_heaps: dict[int, list[tuple[float, int]]] = {app_id: [] for app_id in common_id_set}
    previous_key: tuple[int, int] | None = None
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("source_app_id", "target_app_id", weight_column))
        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)
            if previous_key is not None and key <= previous_key:
                raise AssertionError(f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}")
            previous_key = key
            if source not in common_id_set or target not in common_id_set:
                raise AssertionError(f"Aresta fora do universo comum encontrada em {input_path.name}: {key}")
            update_heap(top_heaps[source], weight, target, top_k)
            update_heap(top_heaps[target], weight, source, top_k)

    selected_neighbors = {node: {neighbor for _, neighbor in heap} for node, heap in top_heaps.items()}
    selected_edges: set[tuple[int, int]] = set()
    for source, neighbors in selected_neighbors.items():
        for target in neighbors:
            left, right = (source, target) if source < target else (target, source)
            if selection_mode == "mutual":
                if source not in selected_neighbors[target] or target not in selected_neighbors[source]:
                    continue
            selected_edges.add((left, right))
    return selected_edges


def compute_strengths(
    input_path: Path,
    common_game_ids: list[int],
    weight_column: str,
) -> dict[int, float]:
    strengths = {app_id: 0.0 for app_id in common_game_ids}
    previous_key: tuple[int, int] | None = None
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("source_app_id", "target_app_id", weight_column))
        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)
            if previous_key is not None and key <= previous_key:
                raise AssertionError(f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}")
            previous_key = key
            strengths[source] += weight
            strengths[target] += weight
    return strengths


def build_working_graph(
    input_path: Path,
    common_game_ids: list[int],
    selected_edges: set[tuple[int, int]],
    weight_column: str,
) -> ig.Graph:
    index_map = {app_id: idx for idx, app_id in enumerate(common_game_ids)}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    previous_key: tuple[int, int] | None = None
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("source_app_id", "target_app_id", weight_column))
        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)
            if previous_key is not None and key <= previous_key:
                raise AssertionError(f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}")
            previous_key = key
            if key not in selected_edges:
                continue
            edges.append((index_map[source], index_map[target]))
            weights.append(weight)
    graph = ig.Graph(n=len(common_game_ids), edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.es["distance"] = [1.0 / weight for weight in weights]
    graph.vs["app_id"] = common_game_ids
    return graph


def build_rank(scores: dict[int, float]) -> list[tuple[int, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def top_k_overlap(rank_left: list[tuple[int, float]], rank_right: list[tuple[int, float]], k: int) -> dict[str, object]:
    left_ids = [app_id for app_id, _ in rank_left[:k]]
    right_ids = [app_id for app_id, _ in rank_right[:k]]
    overlap = sorted(set(left_ids).intersection(right_ids))
    return {
        "k": k,
        "overlap_count": len(overlap),
        "overlap_pct": round(100 * len(overlap) / k, 6) if k else 0.0,
        "overlap_app_ids": overlap,
    }


def full_spearman(scores_left: dict[int, float], scores_right: dict[int, float]) -> float | None:
    ids = sorted(scores_left)
    left = np.array([scores_left[app_id] for app_id in ids], dtype=np.float64)
    right = np.array([scores_right[app_id] for app_id in ids], dtype=np.float64)
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = spearmanr(left, right).correlation
    if value is None or np.isnan(value):
        return None
    return round(float(value), 6)


def igraph_scores(graph: ig.Graph, metric: str, damping: float) -> dict[int, float]:
    if metric == "betweenness":
        values = graph.betweenness(weights=graph.es["distance"])
    elif metric == "pagerank":
        values = graph.pagerank(weights=graph.es["weight"], damping=damping)
    else:
        raise ValueError(metric)
    return {int(graph.vs[idx]["app_id"]): float(values[idx]) for idx in range(graph.vcount())}


def summarize_top_rows(
    projection_name: str,
    names: dict[int, str],
    strength_rank: list[tuple[int, float]],
    betweenness_rank: list[tuple[int, float]],
    pagerank_rank: list[tuple[int, float]],
    k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_name, ranking in (
        ("strength", strength_rank),
        ("betweenness", betweenness_rank),
        ("pagerank", pagerank_rank),
    ):
        for position, (app_id, score) in enumerate(ranking[:k], start=1):
            rows.append(
                {
                    "projection": projection_name,
                    "metric": metric_name,
                    "rank": position,
                    "app_id": app_id,
                    "name": names.get(app_id, str(app_id)),
                    "score": round(score, 10),
                }
            )
    return rows


def save_top_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["projection", "metric", "rank", "app_id", "name", "score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_stats(output_path: Path, stats: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    name_map_input = resolve_path(base_dir, args.name_map_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    stats_output = resolve_path(base_dir, args.stats_output)
    catalog_top_output = resolve_path(base_dir, args.catalog_top_output)
    community_top_output = resolve_path(base_dir, args.community_top_output)

    common_game_ids = load_common_game_ids(common_ids_input)
    names = load_name_map(name_map_input)
    common_id_set = set(common_game_ids)

    catalog_strength = compute_strengths(catalog_input, common_game_ids, "jaccard_tags")
    community_strength = compute_strengths(community_input, common_game_ids, "jaccard_users")

    catalog_selected = build_selected_edge_set(catalog_input, common_id_set, args.top_k, args.selection_mode, "jaccard_tags")
    community_selected = build_selected_edge_set(community_input, common_id_set, args.top_k, args.selection_mode, "jaccard_users")

    catalog_graph = build_working_graph(catalog_input, common_game_ids, catalog_selected, "jaccard_tags")
    community_graph = build_working_graph(community_input, common_game_ids, community_selected, "jaccard_users")

    catalog_betweenness = igraph_scores(catalog_graph, "betweenness", args.pagerank_damping)
    community_betweenness = igraph_scores(community_graph, "betweenness", args.pagerank_damping)
    catalog_pagerank = igraph_scores(catalog_graph, "pagerank", args.pagerank_damping)
    community_pagerank = igraph_scores(community_graph, "pagerank", args.pagerank_damping)

    catalog_strength_rank = build_rank(catalog_strength)
    community_strength_rank = build_rank(community_strength)
    catalog_betweenness_rank = build_rank(catalog_betweenness)
    community_betweenness_rank = build_rank(community_betweenness)
    catalog_pagerank_rank = build_rank(catalog_pagerank)
    community_pagerank_rank = build_rank(community_pagerank)

    save_top_rows(
        catalog_top_output,
        summarize_top_rows(
            "catalog",
            names,
            catalog_strength_rank,
            catalog_betweenness_rank,
            catalog_pagerank_rank,
            args.report_top_k,
        ),
    )
    save_top_rows(
        community_top_output,
        summarize_top_rows(
            "community",
            names,
            community_strength_rank,
            community_betweenness_rank,
            community_pagerank_rank,
            args.report_top_k,
        ),
    )

    stats = {
        "common_game_universe_size": len(common_game_ids),
        "working_graph_parameters": {
            "top_k": args.top_k,
            "selection_mode": args.selection_mode,
            "pagerank_damping": args.pagerank_damping,
        },
        "strength_comparison": {
            "catalog_top_k": top_k_overlap(catalog_strength_rank, community_strength_rank, args.report_top_k),
            "spearman_full_universe": full_spearman(catalog_strength, community_strength),
        },
        "betweenness_comparison": {
            "catalog_top_k": top_k_overlap(catalog_betweenness_rank, community_betweenness_rank, args.report_top_k),
            "spearman_full_universe": full_spearman(catalog_betweenness, community_betweenness),
        },
        "pagerank_comparison": {
            "catalog_top_k": top_k_overlap(catalog_pagerank_rank, community_pagerank_rank, args.report_top_k),
            "spearman_full_universe": full_spearman(catalog_pagerank, community_pagerank),
        },
        "catalog_top_examples": {
            "strength": [app_id for app_id, _ in catalog_strength_rank[:args.report_top_k]],
            "betweenness": [app_id for app_id, _ in catalog_betweenness_rank[:args.report_top_k]],
            "pagerank": [app_id for app_id, _ in catalog_pagerank_rank[:args.report_top_k]],
        },
        "community_top_examples": {
            "strength": [app_id for app_id, _ in community_strength_rank[:args.report_top_k]],
            "betweenness": [app_id for app_id, _ in community_betweenness_rank[:args.report_top_k]],
            "pagerank": [app_id for app_id, _ in community_pagerank_rank[:args.report_top_k]],
        },
    }
    save_stats(stats_output, stats)

    print(f"common_games: {len(common_game_ids)}")
    print(f"catalog_strength_top1: {catalog_strength_rank[0][0]}")
    print(f"community_strength_top1: {community_strength_rank[0][0]}")
    print(f"strength_spearman: {stats['strength_comparison']['spearman_full_universe']}")
    print(f"betweenness_spearman: {stats['betweenness_comparison']['spearman_full_universe']}")
    print(f"pagerank_spearman: {stats['pagerank_comparison']['spearman_full_universe']}")
    print(f"catalog_top_output: {catalog_top_output}")
    print(f"community_top_output: {community_top_output}")
    print(f"stats_output: {stats_output}")


if __name__ == "__main__":
    main()
