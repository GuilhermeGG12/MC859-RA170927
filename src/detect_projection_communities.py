from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_NAME_MAP_INPUT = Path("data/processed/common_game_id_map.csv")
DEFAULT_CATALOG_INPUT = Path("data/processed/game_game_catalog_edges_final.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/game_game_community_edges_final.csv")
DEFAULT_CATALOG_OUTPUT = Path("data/processed/catalog_final_communities.csv")
DEFAULT_COMMUNITY_OUTPUT = Path("data/processed/community_final_communities.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/final_projection_community_comparison.json")
DEFAULT_TOP_K = 50
DEFAULT_SELECTION_MODE = "union"
DEFAULT_RESOLUTION = 1.0
DEFAULT_SEED = 42


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
            "Detecta comunidades nas projecoes finais filtradas usando Leiden sobre "
            "grafos de trabalho top-k simetrizados."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--common-ids-input", type=Path, default=DEFAULT_COMMON_IDS_INPUT)
    parser.add_argument("--name-map-input", type=Path, default=DEFAULT_NAME_MAP_INPUT)
    parser.add_argument("--catalog-input", type=Path, default=DEFAULT_CATALOG_INPUT)
    parser.add_argument("--community-input", type=Path, default=DEFAULT_COMMUNITY_INPUT)
    parser.add_argument("--catalog-output", type=Path, default=DEFAULT_CATALOG_OUTPUT)
    parser.add_argument("--community-output", type=Path, default=DEFAULT_COMMUNITY_OUTPUT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--selection-mode",
        choices=("union", "mutual"),
        default=DEFAULT_SELECTION_MODE,
        help="Criterio de simetrizacao do top-k local de cada jogo.",
    )
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def validate_required_columns(fieldnames: list[str] | None, required: tuple[str, ...]) -> None:
    if fieldnames is None:
        raise AssertionError("CSV sem cabecalho")
    missing_columns = [column for column in required if column not in fieldnames]
    if missing_columns:
        raise AssertionError(
            f"CSV precisa conter as colunas {required}; faltando={missing_columns}"
        )


def load_common_game_ids(input_path: Path) -> list[int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    game_ids: list[int] = []
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id",))
        for row in reader:
            game_ids.append(int(str(row["app_id"]).strip()))
    if not game_ids:
        raise AssertionError("A lista de jogos comuns nao pode ser vazia")
    return sorted(game_ids)


def load_name_map(input_path: Path) -> dict[int, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    names: dict[int, str] = {}
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id", "name"))
        for row in reader:
            app_id = int(str(row["app_id"]).strip())
            names[app_id] = str(row["name"]).strip()
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
            if source not in common_id_set or target not in common_id_set:
                raise AssertionError(
                    f"Aresta fora do universo comum encontrada em {input_path.name}: {key}"
                )
            if previous_key is not None and key <= previous_key:
                raise AssertionError(
                    f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}"
                )
            previous_key = key
            update_heap(top_heaps[source], weight, target, top_k)
            update_heap(top_heaps[target], weight, source, top_k)

    selected_neighbors = {node: {neighbor for _, neighbor in heap} for node, heap in top_heaps.items()}
    selected_edges: set[tuple[int, int]] = set()
    for source, neighbors in selected_neighbors.items():
        for target in neighbors:
            left, right = (source, target) if source < target else (target, source)
            if selection_mode == "mutual":
                if left not in selected_neighbors[right] or right not in selected_neighbors[left]:
                    continue
            selected_edges.add((left, right))
    return selected_edges


def build_working_graph(
    input_path: Path,
    common_game_ids: list[int],
    selected_edges: set[tuple[int, int]],
    weight_column: str,
) -> tuple[ig.Graph, dict[str, object]]:
    index_map = {app_id: idx for idx, app_id in enumerate(common_game_ids)}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    degrees: Counter[int] = Counter({app_id: 0 for app_id in common_game_ids})
    dsu = DisjointSet(common_game_ids)
    previous_key: tuple[int, int] | None = None
    weight_sum = 0.0
    min_weight = None
    max_weight = None

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("source_app_id", "target_app_id", weight_column))
        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)
            if previous_key is not None and key <= previous_key:
                raise AssertionError(
                    f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}"
                )
            previous_key = key
            if key not in selected_edges:
                continue
            edges.append((index_map[source], index_map[target]))
            weights.append(weight)
            degrees[source] += 1
            degrees[target] += 1
            dsu.union(source, target)
            weight_sum += weight
            min_weight = weight if min_weight is None else min(min_weight, weight)
            max_weight = weight if max_weight is None else max(max_weight, weight)

    graph = ig.Graph(n=len(common_game_ids), edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.vs["app_id"] = common_game_ids

    component_sizes = dsu.component_sizes()
    giant_component_size = component_sizes[0] if component_sizes else 0
    n_edges = len(edges)
    n_nodes = len(common_game_ids)

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round((2 * n_edges) / n_nodes, 4) if n_nodes else 0.0,
        "n_components": len(component_sizes),
        "giant_component_size": int(giant_component_size),
        "giant_component_pct": round(100 * giant_component_size / n_nodes, 4) if n_nodes else 0.0,
        "n_isolated_games": int(sum(1 for degree in degrees.values() if degree == 0)),
        "weight_min": round(float(min_weight), 6) if min_weight is not None else 0.0,
        "weight_max": round(float(max_weight), 6) if max_weight is not None else 0.0,
        "weight_avg": round(weight_sum / n_edges, 6) if n_edges else 0.0,
    }
    return graph, stats


def run_leiden(graph: ig.Graph, resolution: float, seed: int) -> tuple[list[int], dict[str, object]]:
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = partition.membership
    community_sizes = sorted((len(block) for block in partition), reverse=True)
    return membership, {
        "n_communities": len(partition),
        "largest_community_size": int(community_sizes[0]) if community_sizes else 0,
        "largest_community_pct": round(100 * community_sizes[0] / graph.vcount(), 4) if community_sizes else 0.0,
        "community_sizes_top10": [int(size) for size in community_sizes[:10]],
        "modularity": round(float(partition.modularity), 6),
    }


def save_assignments(
    output_path: Path,
    common_game_ids: list[int],
    names: dict[int, str],
    membership: list[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    community_counts = Counter(membership)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["app_id", "name", "community_id", "community_size"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for app_id, community_id in sorted(zip(common_game_ids, membership), key=lambda item: (item[1], item[0])):
            writer.writerow(
                {
                    "app_id": app_id,
                    "name": names.get(app_id, str(app_id)),
                    "community_id": int(community_id),
                    "community_size": int(community_counts[community_id]),
                }
            )


def save_stats(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    name_map_input = resolve_path(base_dir, args.name_map_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    catalog_output = resolve_path(base_dir, args.catalog_output)
    community_output = resolve_path(base_dir, args.community_output)
    stats_output = resolve_path(base_dir, args.stats_output)

    common_game_ids = load_common_game_ids(common_ids_input)
    names = load_name_map(name_map_input)
    common_id_set = set(common_game_ids)

    catalog_selected = build_selected_edge_set(
        input_path=catalog_input,
        common_id_set=common_id_set,
        top_k=args.top_k,
        selection_mode=args.selection_mode,
        weight_column="jaccard_tags",
    )
    community_selected = build_selected_edge_set(
        input_path=community_input,
        common_id_set=common_id_set,
        top_k=args.top_k,
        selection_mode=args.selection_mode,
        weight_column="jaccard_users",
    )

    catalog_graph, catalog_working_stats = build_working_graph(
        input_path=catalog_input,
        common_game_ids=common_game_ids,
        selected_edges=catalog_selected,
        weight_column="jaccard_tags",
    )
    community_graph, community_working_stats = build_working_graph(
        input_path=community_input,
        common_game_ids=common_game_ids,
        selected_edges=community_selected,
        weight_column="jaccard_users",
    )

    catalog_membership, catalog_partition_stats = run_leiden(
        graph=catalog_graph,
        resolution=args.resolution,
        seed=args.seed,
    )
    community_membership, community_partition_stats = run_leiden(
        graph=community_graph,
        resolution=args.resolution,
        seed=args.seed,
    )

    save_assignments(catalog_output, common_game_ids, names, catalog_membership)
    save_assignments(community_output, common_game_ids, names, community_membership)

    ari = round(float(adjusted_rand_score(catalog_membership, community_membership)), 6)
    nmi = round(float(normalized_mutual_info_score(catalog_membership, community_membership)), 6)

    stats = {
        "common_game_universe_size": len(common_game_ids),
        "algorithm": "leiden",
        "top_k": args.top_k,
        "selection_mode": args.selection_mode,
        "resolution": args.resolution,
        "seed": args.seed,
        "catalog_working_graph": catalog_working_stats,
        "catalog_partition": catalog_partition_stats,
        "community_working_graph": community_working_stats,
        "community_partition": community_partition_stats,
        "partition_comparison": {
            "ari": ari,
            "nmi": nmi,
        },
    }
    save_stats(stats, stats_output)

    print(f"common_games: {len(common_game_ids)}")
    print(f"catalog_working_edges: {catalog_working_stats['n_edges']}")
    print(f"community_working_edges: {community_working_stats['n_edges']}")
    print(f"catalog_communities: {catalog_partition_stats['n_communities']}")
    print(f"community_communities: {community_partition_stats['n_communities']}")
    print(f"ari: {ari}")
    print(f"nmi: {nmi}")
    print(f"catalog_output: {catalog_output}")
    print(f"community_output: {community_output}")
    print(f"stats_output: {stats_output}")


if __name__ == "__main__":
    main()
