from __future__ import annotations

import argparse
import csv
import json
import math
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
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/projection_threshold_sweep_summary.json")
DEFAULT_TABLE_OUTPUT = Path("data/processed/projection_threshold_sweep_table.csv")
DEFAULT_EDGES_FIGURE = Path("figures/projection_threshold_edges.png")
DEFAULT_COMPONENTS_FIGURE = Path("figures/projection_threshold_components.png")
DEFAULT_GIANT_FIGURE = Path("figures/projection_threshold_giant_component.png")
DEFAULT_DEGREE_FIGURE = Path("figures/projection_threshold_avg_degree.png")
DEFAULT_CATALOG_THRESHOLDS = "0.0,0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2"
DEFAULT_COMMUNITY_THRESHOLDS = "0.0,0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05"
DEFAULT_SAMPLE_SIZE = 200000


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
        description="Varre thresholds de peso para as projecoes de catalogo e comunidade."
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
        "--catalog-thresholds",
        type=str,
        default=DEFAULT_CATALOG_THRESHOLDS,
        help="Lista separada por virgula de thresholds para jaccard_tags.",
    )
    parser.add_argument(
        "--community-thresholds",
        type=str,
        default=DEFAULT_COMMUNITY_THRESHOLDS,
        help="Lista separada por virgula de thresholds para jaccard_users.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Caminho relativo ao base-dir para salvar projection_threshold_sweep_summary.json.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=DEFAULT_TABLE_OUTPUT,
        help="Caminho relativo ao base-dir para salvar projection_threshold_sweep_table.csv.",
    )
    parser.add_argument(
        "--edges-figure",
        type=Path,
        default=DEFAULT_EDGES_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de arestas por threshold.",
    )
    parser.add_argument(
        "--components-figure",
        type=Path,
        default=DEFAULT_COMPONENTS_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de componentes por threshold.",
    )
    parser.add_argument(
        "--giant-figure",
        type=Path,
        default=DEFAULT_GIANT_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura da componente gigante por threshold.",
    )
    parser.add_argument(
        "--degree-figure",
        type=Path,
        default=DEFAULT_DEGREE_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de grau medio por threshold.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Tamanho maximo da amostra deterministica de pesos por threshold.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def parse_thresholds(raw_value: str) -> list[float]:
    thresholds: list[float] = []
    for chunk in raw_value.split(","):
        value = float(chunk.strip())
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"Threshold fora do intervalo [0, 1]: {value}")
        thresholds.append(value)
    ordered = sorted(set(thresholds))
    if not ordered:
        raise AssertionError("A lista de thresholds nao pode ser vazia")
    return ordered


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


def analyze_single_threshold(
    input_path: Path,
    common_game_ids: list[int],
    threshold: float,
    weight_column: str,
    sample_size: int,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    common_id_set = set(common_game_ids)
    degrees: Counter[int] = Counter({app_id: 0 for app_id in common_game_ids})
    dsu = DisjointSet(common_game_ids)
    sampler = ReservoirSampler(sample_size)
    previous_key: tuple[int, int] | None = None
    n_edges = 0
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
            if weight < threshold:
                continue

            n_edges += 1
            degrees[source] += 1
            degrees[target] += 1
            dsu.union(source, target)
            weight_sum += weight
            sampler.add(weight)
            min_weight = weight if min_weight is None else min(min_weight, weight)
            max_weight = weight if max_weight is None else max(max_weight, weight)

    degree_distribution = Counter(degrees.values())
    component_sizes = dsu.component_sizes()
    giant_component_size = component_sizes[0] if component_sizes else 0
    sample = sampler.as_array()
    n_nodes = len(common_game_ids)

    return {
        "threshold": threshold,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round((2 * n_edges) / n_nodes, 4),
        "n_components": len(component_sizes),
        "giant_component_size": int(giant_component_size),
        "giant_component_pct": round(100 * giant_component_size / n_nodes, 4),
        "n_isolated_games": int(degree_distribution.get(0, 0)),
        "weight_min": round(float(min_weight), 6) if min_weight is not None else 0.0,
        "weight_max": round(float(max_weight), 6) if max_weight is not None else 0.0,
        "weight_avg": round(weight_sum / n_edges, 6) if n_edges else 0.0,
        "weight_q25": round(float(np.quantile(sample, 0.25)), 6) if sample.size else 0.0,
        "weight_median": round(float(np.quantile(sample, 0.5)), 6) if sample.size else 0.0,
        "weight_q75": round(float(np.quantile(sample, 0.75)), 6) if sample.size else 0.0,
        "weight_q90": round(float(np.quantile(sample, 0.9)), 6) if sample.size else 0.0,
        "weight_q99": round(float(np.quantile(sample, 0.99)), 6) if sample.size else 0.0,
        "weight_sample_size": int(sample.size),
    }


def analyze_threshold_series(
    input_path: Path,
    common_game_ids: list[int],
    thresholds: list[float],
    weight_column: str,
    sample_size: int,
) -> list[dict[str, object]]:
    results = [
        analyze_single_threshold(
            input_path=input_path,
            common_game_ids=common_game_ids,
            threshold=threshold,
            weight_column=weight_column,
            sample_size=sample_size,
        )
        for threshold in thresholds
    ]
    if results:
        base_edges = max(1, int(results[0]["n_edges"]))
        for row in results:
            row["edge_retention_pct"] = round(100 * int(row["n_edges"]) / base_edges, 4)
    return results


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def save_table(catalog_results: list[dict[str, object]], community_results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection",
        "threshold",
        "n_nodes",
        "n_edges",
        "edge_retention_pct",
        "avg_degree",
        "n_components",
        "giant_component_size",
        "giant_component_pct",
        "n_isolated_games",
        "weight_min",
        "weight_max",
        "weight_avg",
        "weight_q25",
        "weight_median",
        "weight_q75",
        "weight_q90",
        "weight_q99",
        "weight_sample_size",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for projection_name, results in (("catalog", catalog_results), ("community", community_results)):
            for row in results:
                writer.writerow({"projection": projection_name, **row})


def save_line_figure(
    catalog_results: list[dict[str, object]],
    community_results: list[dict[str, object]],
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    y_label: str,
    y_scale: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        [float(row[x_key]) for row in catalog_results],
        [float(row[y_key]) for row in catalog_results],
        marker="o",
        color="#2a6f97",
        label="Catalogo",
    )
    ax.plot(
        [float(row[x_key]) for row in community_results],
        [float(row[y_key]) for row in community_results],
        marker="o",
        color="#6a994e",
        label="Comunidade",
    )
    if y_scale is not None:
        ax.set_yscale(y_scale)
    ax.set_title(title)
    ax.set_xlabel("Threshold de peso")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    summary_output = resolve_path(base_dir, args.summary_output)
    table_output = resolve_path(base_dir, args.table_output)
    edges_figure = resolve_path(base_dir, args.edges_figure)
    components_figure = resolve_path(base_dir, args.components_figure)
    giant_figure = resolve_path(base_dir, args.giant_figure)
    degree_figure = resolve_path(base_dir, args.degree_figure)

    common_game_ids = load_common_game_ids(common_ids_input)
    catalog_thresholds = parse_thresholds(args.catalog_thresholds)
    community_thresholds = parse_thresholds(args.community_thresholds)

    catalog_results = analyze_threshold_series(
        input_path=catalog_input,
        common_game_ids=common_game_ids,
        thresholds=catalog_thresholds,
        weight_column="jaccard_tags",
        sample_size=args.sample_size,
    )
    community_results = analyze_threshold_series(
        input_path=community_input,
        common_game_ids=common_game_ids,
        thresholds=community_thresholds,
        weight_column="jaccard_users",
        sample_size=args.sample_size,
    )

    summary = {
        "common_game_universe_size": len(common_game_ids),
        "weight_sample_size": args.sample_size,
        "catalog_thresholds": catalog_results,
        "community_thresholds": community_results,
    }
    save_summary(summary, summary_output)
    save_table(catalog_results, community_results, table_output)

    save_line_figure(
        catalog_results=catalog_results,
        community_results=community_results,
        x_key="threshold",
        y_key="n_edges",
        output_path=edges_figure,
        title="Arestas restantes por threshold",
        y_label="Numero de arestas",
        y_scale="log",
    )
    save_line_figure(
        catalog_results=catalog_results,
        community_results=community_results,
        x_key="threshold",
        y_key="n_components",
        output_path=components_figure,
        title="Numero de componentes por threshold",
        y_label="Numero de componentes",
    )
    save_line_figure(
        catalog_results=catalog_results,
        community_results=community_results,
        x_key="threshold",
        y_key="giant_component_pct",
        output_path=giant_figure,
        title="Componente gigante por threshold",
        y_label="Componente gigante (%)",
    )
    save_line_figure(
        catalog_results=catalog_results,
        community_results=community_results,
        x_key="threshold",
        y_key="avg_degree",
        output_path=degree_figure,
        title="Grau medio por threshold",
        y_label="Grau medio",
        y_scale="log",
    )

    print(f"common_games: {len(common_game_ids)}")
    print(f"catalog_thresholds_tested: {len(catalog_results)}")
    print(f"community_thresholds_tested: {len(community_results)}")
    print(f"weight_sample_size: {args.sample_size}")
    print(f"Arquivos salvos em: {summary_output} e {table_output}")


if __name__ == "__main__":
    main()
