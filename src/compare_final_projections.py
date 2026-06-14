from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_CATALOG_INPUT = Path("data/processed/game_game_catalog_edges_final.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/game_game_community_edges_final.csv")
DEFAULT_STATS_INPUT = Path("data/processed/final_filtered_projection_stats.json")
DEFAULT_OUTPUT = Path("data/processed/final_projection_comparison_stats.json")
DEFAULT_SHARED_WEIGHTS_OUTPUT = Path("data/processed/final_projection_shared_edge_weights.csv")
DEFAULT_WEIGHT_SAMPLE_SIZE = 200000


class ReservoirSampler:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.values: list[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.max_size:
            self.values.append(value)
            return
        position = ((self.seen * 1103515245 + 12345) & 0x7FFFFFFF) % self.seen
        if position < self.max_size:
            self.values[position] = value

    def as_array(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara estruturalmente as projecoes finais filtradas de catalogo e comunidade."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--common-ids-input",
        type=Path,
        default=DEFAULT_COMMON_IDS_INPUT,
        help="Caminho relativo ao base-dir para common_game_ids.csv.",
    )
    parser.add_argument(
        "--catalog-input",
        type=Path,
        default=DEFAULT_CATALOG_INPUT,
        help="Caminho relativo ao base-dir para game_game_catalog_edges_final.csv.",
    )
    parser.add_argument(
        "--community-input",
        type=Path,
        default=DEFAULT_COMMUNITY_INPUT,
        help="Caminho relativo ao base-dir para game_game_community_edges_final.csv.",
    )
    parser.add_argument(
        "--stats-input",
        type=Path,
        default=DEFAULT_STATS_INPUT,
        help="Caminho relativo ao base-dir para final_filtered_projection_stats.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Caminho relativo ao base-dir para final_projection_comparison_stats.json.",
    )
    parser.add_argument(
        "--shared-weights-output",
        type=Path,
        default=DEFAULT_SHARED_WEIGHTS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar pesos das arestas compartilhadas.",
    )
    parser.add_argument(
        "--weight-sample-size",
        type=int,
        default=DEFAULT_WEIGHT_SAMPLE_SIZE,
        help="Tamanho maximo da amostra deterministica de pesos por projecao.",
    )
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


def load_json(input_path: Path) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def summarize_degree_distribution(n_nodes: int, n_edges: int, n_components: int, giant_component_pct: float, n_isolated_games: int) -> dict[str, float | int]:
    density = 0.0
    if n_nodes > 1:
        density = round((2 * n_edges) / (n_nodes * (n_nodes - 1)), 8)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round((2 * n_edges) / n_nodes, 4) if n_nodes else 0.0,
        "density": density,
        "n_components": n_components,
        "giant_component_pct": giant_component_pct,
        "n_isolated_games": n_isolated_games,
    }


def projection_weight_summary(input_path: Path, weight_column: str, sample_size: int) -> dict[str, float | int]:
    sampler = ReservoirSampler(sample_size)
    min_weight = None
    max_weight = None
    weight_sum = 0.0
    n_edges = 0
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("source_app_id", "target_app_id", weight_column))
        previous_key: tuple[int, int] | None = None
        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)
            if previous_key is not None and key <= previous_key:
                raise AssertionError(f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}")
            previous_key = key
            n_edges += 1
            weight_sum += weight
            sampler.add(weight)
            min_weight = weight if min_weight is None else min(min_weight, weight)
            max_weight = weight if max_weight is None else max(max_weight, weight)
    sample = sampler.as_array()
    return {
        "min": round(float(min_weight), 6) if min_weight is not None else 0.0,
        "max": round(float(max_weight), 6) if max_weight is not None else 0.0,
        "avg": round(weight_sum / n_edges, 6) if n_edges else 0.0,
        "q25_sampled": round(float(np.quantile(sample, 0.25)), 6) if sample.size else 0.0,
        "median_sampled": round(float(np.quantile(sample, 0.5)), 6) if sample.size else 0.0,
        "q75_sampled": round(float(np.quantile(sample, 0.75)), 6) if sample.size else 0.0,
        "q90_sampled": round(float(np.quantile(sample, 0.9)), 6) if sample.size else 0.0,
        "q99_sampled": round(float(np.quantile(sample, 0.99)), 6) if sample.size else 0.0,
        "sample_size": int(sample.size),
    }


def read_edge(reader: csv.DictReader, weight_column: str) -> tuple[tuple[int, int], float] | None:
    try:
        row = next(reader)
    except StopIteration:
        return None
    source = int(str(row["source_app_id"]).strip())
    target = int(str(row["target_app_id"]).strip())
    weight = float(str(row[weight_column]).strip())
    return (source, target), weight


def compute_overlap_and_weight_correlation(
    catalog_input: Path,
    community_input: Path,
    shared_weights_output: Path,
) -> dict[str, object]:
    shared_weights_output.parent.mkdir(parents=True, exist_ok=True)

    catalog_weights_shared: list[float] = []
    community_weights_shared: list[float] = []
    n_shared_edges = 0

    with catalog_input.open("r", encoding="utf-8", newline="") as catalog_file, community_input.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as community_file, shared_weights_output.open("w", encoding="utf-8", newline="") as output_file:
        catalog_reader = csv.DictReader(catalog_file)
        community_reader = csv.DictReader(community_file)
        validate_required_columns(catalog_reader.fieldnames, ("source_app_id", "target_app_id", "jaccard_tags"))
        validate_required_columns(community_reader.fieldnames, ("source_app_id", "target_app_id", "jaccard_users"))

        writer = csv.DictWriter(
            output_file,
            fieldnames=["source_app_id", "target_app_id", "jaccard_tags", "jaccard_users"],
        )
        writer.writeheader()

        catalog_current = read_edge(catalog_reader, "jaccard_tags")
        community_current = read_edge(community_reader, "jaccard_users")

        while catalog_current is not None and community_current is not None:
            catalog_key, catalog_weight = catalog_current
            community_key, community_weight = community_current

            if catalog_key == community_key:
                n_shared_edges += 1
                catalog_weights_shared.append(catalog_weight)
                community_weights_shared.append(community_weight)
                writer.writerow(
                    {
                        "source_app_id": catalog_key[0],
                        "target_app_id": catalog_key[1],
                        "jaccard_tags": round(catalog_weight, 6),
                        "jaccard_users": round(community_weight, 6),
                    }
                )
                catalog_current = read_edge(catalog_reader, "jaccard_tags")
                community_current = read_edge(community_reader, "jaccard_users")
            elif catalog_key < community_key:
                catalog_current = read_edge(catalog_reader, "jaccard_tags")
            else:
                community_current = read_edge(community_reader, "jaccard_users")

    catalog_array = np.array(catalog_weights_shared, dtype=np.float64)
    community_array = np.array(community_weights_shared, dtype=np.float64)

    pearson = compute_pearson(catalog_array, community_array)
    spearman = compute_spearman(catalog_array, community_array)

    return {
        "n_shared_edges": n_shared_edges,
        "pearson_shared_weights": pearson,
        "spearman_shared_weights": spearman,
        "catalog_shared_weight_summary": summarize_array(catalog_array),
        "community_shared_weight_summary": summarize_array(community_array),
    }


def summarize_array(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"n": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0}
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "avg": round(float(values.mean()), 6),
        "q25": round(float(np.quantile(values, 0.25)), 6),
        "median": round(float(np.quantile(values, 0.5)), 6),
        "q75": round(float(np.quantile(values, 0.75)), 6),
    }


def compute_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    if left.size != right.size:
        raise AssertionError("Arrays de Pearson com tamanhos diferentes")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return round(float(np.corrcoef(left, right)[0, 1]), 6)


def average_ranks(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        next_index = index + 1
        while next_index < values.size and sorted_values[next_index] == sorted_values[index]:
            next_index += 1
        average_rank = ((index + 1) + next_index) / 2.0
        ranks[order[index:next_index]] = average_rank
        index = next_index
    return ranks


def compute_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    if left.size != right.size:
        raise AssertionError("Arrays de Spearman com tamanhos diferentes")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    return compute_pearson(left_rank, right_rank)


def save_stats(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    stats_input = resolve_path(base_dir, args.stats_input)
    output = resolve_path(base_dir, args.output)
    shared_weights_output = resolve_path(base_dir, args.shared_weights_output)

    common_game_ids = load_common_game_ids(common_ids_input)
    filtered_stats = load_json(stats_input)

    catalog_final = filtered_stats["catalog_final_projection"]
    community_final = filtered_stats["community_final_projection"]

    catalog_structural = summarize_degree_distribution(
        n_nodes=int(catalog_final["n_nodes"]),
        n_edges=int(catalog_final["n_edges"]),
        n_components=int(catalog_final["n_components"]),
        giant_component_pct=float(catalog_final["giant_component_pct"]),
        n_isolated_games=int(catalog_final["n_isolated_games"]),
    )
    community_structural = summarize_degree_distribution(
        n_nodes=int(community_final["n_nodes"]),
        n_edges=int(community_final["n_edges"]),
        n_components=int(community_final["n_components"]),
        giant_component_pct=float(community_final["giant_component_pct"]),
        n_isolated_games=int(community_final["n_isolated_games"]),
    )

    catalog_weight_distribution = projection_weight_summary(
        input_path=catalog_input,
        weight_column="jaccard_tags",
        sample_size=args.weight_sample_size,
    )
    community_weight_distribution = projection_weight_summary(
        input_path=community_input,
        weight_column="jaccard_users",
        sample_size=args.weight_sample_size,
    )

    overlap = compute_overlap_and_weight_correlation(
        catalog_input=catalog_input,
        community_input=community_input,
        shared_weights_output=shared_weights_output,
    )

    catalog_edges = int(catalog_final["n_edges"])
    community_edges = int(community_final["n_edges"])
    shared_edges = int(overlap["n_shared_edges"])
    union_edges = catalog_edges + community_edges - shared_edges

    overlap_summary = {
        "catalog_edges": catalog_edges,
        "community_edges": community_edges,
        "shared_edges": shared_edges,
        "union_edges": union_edges,
        "edge_jaccard": round(shared_edges / union_edges, 6) if union_edges else 0.0,
        "catalog_overlap_pct": round(100 * shared_edges / catalog_edges, 6) if catalog_edges else 0.0,
        "community_overlap_pct": round(100 * shared_edges / community_edges, 6) if community_edges else 0.0,
    }

    stats = {
        "common_game_universe_size": len(common_game_ids),
        "catalog_projection": {
            "threshold": catalog_final["threshold"],
            "structural": catalog_structural,
            "weight_distribution_sampled": catalog_weight_distribution,
        },
        "community_projection": {
            "threshold": community_final["threshold"],
            "structural": community_structural,
            "weight_distribution_sampled": community_weight_distribution,
        },
        "edge_overlap": overlap_summary,
        "shared_edge_weight_comparison": overlap,
    }
    save_stats(stats, output)

    print(f"common_games: {len(common_game_ids)}")
    print(f"catalog_edges: {catalog_edges}")
    print(f"community_edges: {community_edges}")
    print(f"shared_edges: {shared_edges}")
    print(f"edge_jaccard: {overlap_summary['edge_jaccard']:.6f}")
    print(f"spearman_shared_weights: {overlap['spearman_shared_weights']}")
    print(f"shared_weights_output: {shared_weights_output}")
    print(f"stats_output: {output}")


if __name__ == "__main__":
    main()
