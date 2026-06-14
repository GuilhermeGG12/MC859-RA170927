from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_CATALOG_INPUT = Path("data/processed/game_game_catalog_edges.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/game_game_community_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/projected_graphs_analysis_stats.json")
DEFAULT_CATALOG_DEGREE_OUTPUT = Path("figures/catalog_projection_degree_loglog.png")
DEFAULT_COMMUNITY_DEGREE_OUTPUT = Path("figures/community_projection_degree_loglog.png")
DEFAULT_CATALOG_WEIGHT_OUTPUT = Path("figures/catalog_projection_weight_histogram.png")
DEFAULT_COMMUNITY_WEIGHT_OUTPUT = Path("figures/community_projection_weight_histogram.png")
DEFAULT_COMPONENT_OUTPUT = Path("figures/projected_graphs_component_sizes.png")
DEFAULT_WEIGHT_BINS = 50


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
        description="Analisa as projecoes jogo-jogo de catalogo e comunidade."
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
        help="Caminho relativo ao base-dir para game_game_catalog_edges.csv.",
    )
    parser.add_argument(
        "--community-input",
        type=Path,
        default=DEFAULT_COMMUNITY_INPUT,
        help="Caminho relativo ao base-dir para game_game_community_edges.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar projected_graphs_analysis_stats.json.",
    )
    parser.add_argument(
        "--catalog-degree-output",
        type=Path,
        default=DEFAULT_CATALOG_DEGREE_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o histograma de graus do catalogo.",
    )
    parser.add_argument(
        "--community-degree-output",
        type=Path,
        default=DEFAULT_COMMUNITY_DEGREE_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o histograma de graus da comunidade.",
    )
    parser.add_argument(
        "--catalog-weight-output",
        type=Path,
        default=DEFAULT_CATALOG_WEIGHT_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o histograma de pesos do catalogo.",
    )
    parser.add_argument(
        "--community-weight-output",
        type=Path,
        default=DEFAULT_COMMUNITY_WEIGHT_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o histograma de pesos da comunidade.",
    )
    parser.add_argument(
        "--component-output",
        type=Path,
        default=DEFAULT_COMPONENT_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a comparacao de componentes.",
    )
    parser.add_argument(
        "--weight-bins",
        type=int,
        default=DEFAULT_WEIGHT_BINS,
        help="Numero de bins para os histogramas de peso.",
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


def counter_to_json(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def build_log_bins(values: np.ndarray, target_bins: int = 24) -> np.ndarray:
    min_value = int(values.min())
    max_value = int(values.max())
    if min_value <= 0:
        raise AssertionError("Escala log requer valores estritamente positivos")
    if min_value == max_value:
        return np.array([min_value, max_value + 1], dtype=float)
    return np.logspace(
        np.log10(min_value),
        np.log10(max_value),
        num=target_bins,
        endpoint=True,
    )


def save_degree_histogram(distribution: dict[str, int], output_path: Path, title: str, color: str) -> None:
    degrees = np.array([int(key) for key in distribution], dtype=np.int64)
    frequencies = np.array([int(value) for value in distribution.values()], dtype=np.int64)
    positive_mask = degrees > 0
    degrees = degrees[positive_mask]
    frequencies = frequencies[positive_mask]
    if degrees.size == 0:
        raise AssertionError("Distribuicao sem graus positivos para plot em escala log")
    order = np.argsort(degrees)
    degrees = degrees[order]
    frequencies = frequencies[order]
    bins = build_log_bins(degrees)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(
        degrees,
        bins=bins,
        weights=frequencies,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Grau")
    ax.set_ylabel("Frequencia")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_weight_histogram(values: np.ndarray, output_path: Path, title: str, color: str, bins: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(
        values,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("Peso normalizado")
    ax.set_ylabel("Numero de arestas")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_component_comparison(catalog_sizes: list[int], community_sizes: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Catalogo", "Comunidade"]
    giant_sizes = [catalog_sizes[0], community_sizes[0]]
    component_counts = [len(catalog_sizes), len(community_sizes)]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    axes[0].bar(labels, giant_sizes, color=["#2a6f97", "#6a994e"])
    axes[0].set_title("Tamanho da componente gigante")
    axes[0].set_ylabel("Numero de jogos")

    axes[1].bar(labels, component_counts, color=["#468faf", "#90a955"])
    axes[1].set_title("Numero de componentes")
    axes[1].set_ylabel("Numero de componentes")

    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def analyze_projection(
    input_path: Path,
    common_game_ids: list[int],
    shared_column: str,
    weight_column: str,
) -> tuple[dict[str, object], np.ndarray, list[int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    common_id_set = set(common_game_ids)
    degrees: Counter[int] = Counter({app_id: 0 for app_id in common_game_ids})
    dsu = DisjointSet(common_game_ids)
    weights: list[float] = []
    previous_key: tuple[int, int] | None = None
    edge_count = 0
    shared_sum = 0
    min_shared = None
    max_shared = None
    min_weight = None
    max_weight = None
    weight_sum = 0.0

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(
            reader.fieldnames,
            ("source_app_id", "target_app_id", shared_column, weight_column),
        )

        for row in reader:
            source = int(str(row["source_app_id"]).strip())
            target = int(str(row["target_app_id"]).strip())
            shared_value = int(str(row[shared_column]).strip())
            weight = float(str(row[weight_column]).strip())
            key = (source, target)

            if source not in common_id_set or target not in common_id_set:
                raise AssertionError(
                    f"Aresta fora do universo comum encontrada em {input_path.name}: {key}"
                )
            if source == target:
                raise AssertionError(f"Auto-aresta invalida encontrada em {input_path.name}: {key}")
            if previous_key is not None and key <= previous_key:
                raise AssertionError(
                    f"Arestas fora de ordem ou duplicadas em {input_path.name}: {key}"
                )
            previous_key = key
            if not 0.0 < weight <= 1.0:
                raise AssertionError(f"Peso invalido encontrado em {input_path.name}: {weight}")

            edge_count += 1
            degrees[source] += 1
            degrees[target] += 1
            dsu.union(source, target)
            shared_sum += shared_value
            weight_sum += weight
            weights.append(weight)

            min_shared = shared_value if min_shared is None else min(min_shared, shared_value)
            max_shared = shared_value if max_shared is None else max(max_shared, shared_value)
            min_weight = weight if min_weight is None else min(min_weight, weight)
            max_weight = weight if max_weight is None else max(max_weight, weight)

    degree_distribution = Counter(degrees.values())
    component_sizes = dsu.component_sizes()
    giant_component_size = component_sizes[0] if component_sizes else 0
    isolated_nodes = degree_distribution.get(0, 0)
    weight_array = np.array(weights, dtype=np.float64)

    stats = {
        "n_nodes": len(common_game_ids),
        "n_edges": edge_count,
        "avg_degree": round((2 * edge_count) / len(common_game_ids), 4),
        "degree_distribution": counter_to_json(degree_distribution),
        "n_components": len(component_sizes),
        "giant_component_size": int(giant_component_size),
        "giant_component_pct": round(100 * giant_component_size / len(common_game_ids), 4),
        "n_isolated_games": int(isolated_nodes),
        shared_column: {
            "min": int(min_shared) if min_shared is not None else 0,
            "max": int(max_shared) if max_shared is not None else 0,
            "avg": round(shared_sum / edge_count, 6) if edge_count else 0.0,
        },
        weight_column: {
            "min": round(float(min_weight), 6) if min_weight is not None else 0.0,
            "max": round(float(max_weight), 6) if max_weight is not None else 0.0,
            "avg": round(weight_sum / edge_count, 6) if edge_count else 0.0,
            "q25": round(float(np.quantile(weight_array, 0.25)), 6) if edge_count else 0.0,
            "median": round(float(np.quantile(weight_array, 0.5)), 6) if edge_count else 0.0,
            "q75": round(float(np.quantile(weight_array, 0.75)), 6) if edge_count else 0.0,
            "q90": round(float(np.quantile(weight_array, 0.9)), 6) if edge_count else 0.0,
            "q99": round(float(np.quantile(weight_array, 0.99)), 6) if edge_count else 0.0,
        },
    }
    return stats, weight_array, component_sizes


def save_stats(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    stats_output = resolve_path(base_dir, args.stats_output)
    catalog_degree_output = resolve_path(base_dir, args.catalog_degree_output)
    community_degree_output = resolve_path(base_dir, args.community_degree_output)
    catalog_weight_output = resolve_path(base_dir, args.catalog_weight_output)
    community_weight_output = resolve_path(base_dir, args.community_weight_output)
    component_output = resolve_path(base_dir, args.component_output)

    common_game_ids = load_common_game_ids(common_ids_input)

    catalog_stats, catalog_weights, catalog_component_sizes = analyze_projection(
        input_path=catalog_input,
        common_game_ids=common_game_ids,
        shared_column="shared_tags",
        weight_column="jaccard_tags",
    )
    community_stats, community_weights, community_component_sizes = analyze_projection(
        input_path=community_input,
        common_game_ids=common_game_ids,
        shared_column="shared_users",
        weight_column="jaccard_users",
    )

    save_degree_histogram(
        distribution=catalog_stats["degree_distribution"],
        output_path=catalog_degree_output,
        title="Catalogo: distribuicao de graus da projecao jogo-jogo",
        color="#2a6f97",
    )
    save_degree_histogram(
        distribution=community_stats["degree_distribution"],
        output_path=community_degree_output,
        title="Comunidade: distribuicao de graus da projecao jogo-jogo",
        color="#6a994e",
    )
    save_weight_histogram(
        values=catalog_weights,
        output_path=catalog_weight_output,
        title="Catalogo: distribuicao de pesos jaccard_tags",
        color="#468faf",
        bins=args.weight_bins,
    )
    save_weight_histogram(
        values=community_weights,
        output_path=community_weight_output,
        title="Comunidade: distribuicao de pesos jaccard_users",
        color="#90a955",
        bins=args.weight_bins,
    )
    save_component_comparison(
        catalog_sizes=catalog_component_sizes,
        community_sizes=community_component_sizes,
        output_path=component_output,
    )

    stats = {
        "common_game_universe_size": len(common_game_ids),
        "catalog_projection": catalog_stats,
        "community_projection": community_stats,
    }
    save_stats(stats, stats_output)

    print(f"common_games: {stats['common_game_universe_size']}")
    print(f"catalog_edges: {catalog_stats['n_edges']}")
    print(f"community_edges: {community_stats['n_edges']}")
    print(f"catalog_components: {catalog_stats['n_components']}")
    print(f"community_components: {community_stats['n_components']}")
    print(f"Arquivos salvos em: {stats_output}")


if __name__ == "__main__":
    main()
