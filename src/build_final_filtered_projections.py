from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_CATALOG_INPUT = Path("data/processed/game_game_catalog_edges.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/game_game_community_edges.csv")
DEFAULT_CATALOG_OUTPUT = Path("data/processed/game_game_catalog_edges_final.csv")
DEFAULT_COMMUNITY_OUTPUT = Path("data/processed/game_game_community_edges_final.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/final_filtered_projection_stats.json")
DEFAULT_CATALOG_THRESHOLD = 0.25
DEFAULT_COMMUNITY_THRESHOLD = 0.002


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
        description="Materializa as versoes finais filtradas das projecoes de catalogo e comunidade."
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
        "--catalog-output",
        type=Path,
        default=DEFAULT_CATALOG_OUTPUT,
        help="Caminho relativo ao base-dir para game_game_catalog_edges_final.csv.",
    )
    parser.add_argument(
        "--community-output",
        type=Path,
        default=DEFAULT_COMMUNITY_OUTPUT,
        help="Caminho relativo ao base-dir para game_game_community_edges_final.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para final_filtered_projection_stats.json.",
    )
    parser.add_argument(
        "--catalog-threshold",
        type=float,
        default=DEFAULT_CATALOG_THRESHOLD,
        help="Threshold final de jaccard_tags para o catalogo.",
    )
    parser.add_argument(
        "--community-threshold",
        type=float,
        default=DEFAULT_COMMUNITY_THRESHOLD,
        help="Threshold final de jaccard_users para a comunidade.",
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
            app_id_raw = str(row["app_id"]).strip()
            if app_id_raw == "":
                raise AssertionError(f"app_id vazio em {input_path.name}: {row}")
            game_ids.append(int(app_id_raw))

    if not game_ids:
        raise AssertionError("A lista de jogos comuns nao pode ser vazia")

    return sorted(game_ids)


def filter_projection(
    input_path: Path,
    output_path: Path,
    common_game_ids: list[int],
    threshold: float,
    shared_column: str,
    weight_column: str,
) -> dict[str, int | float | str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    common_id_set = set(common_game_ids)
    degrees: Counter[int] = Counter({app_id: 0 for app_id in common_game_ids})
    dsu = DisjointSet(common_game_ids)
    previous_key: tuple[int, int] | None = None
    n_edges = 0
    shared_sum = 0
    weight_sum = 0.0
    min_shared = None
    max_shared = None
    min_weight = None
    max_weight = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="") as input_file, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        reader = csv.DictReader(input_file)
        validate_required_columns(
            reader.fieldnames,
            ("source_app_id", "target_app_id", shared_column, weight_column),
        )
        writer = csv.DictWriter(output_file, fieldnames=list(reader.fieldnames))
        writer.writeheader()

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
            if weight < threshold:
                continue

            writer.writerow(row)
            n_edges += 1
            degrees[source] += 1
            degrees[target] += 1
            dsu.union(source, target)
            shared_sum += shared_value
            weight_sum += weight
            min_shared = shared_value if min_shared is None else min(min_shared, shared_value)
            max_shared = shared_value if max_shared is None else max(max_shared, shared_value)
            min_weight = weight if min_weight is None else min(min_weight, weight)
            max_weight = weight if max_weight is None else max(max_weight, weight)

    degree_distribution = Counter(degrees.values())
    component_sizes = dsu.component_sizes()
    giant_component_size = component_sizes[0] if component_sizes else 0
    n_nodes = len(common_game_ids)

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "threshold": threshold,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round((2 * n_edges) / n_nodes, 4),
        "n_components": len(component_sizes),
        "giant_component_size": int(giant_component_size),
        "giant_component_pct": round(100 * giant_component_size / n_nodes, 4),
        "n_isolated_games": int(degree_distribution.get(0, 0)),
        f"min_{shared_column}": int(min_shared) if min_shared is not None else 0,
        f"max_{shared_column}": int(max_shared) if max_shared is not None else 0,
        f"avg_{shared_column}": round(shared_sum / n_edges, 6) if n_edges else 0.0,
        f"min_{weight_column}": round(float(min_weight), 6) if min_weight is not None else 0.0,
        f"max_{weight_column}": round(float(max_weight), 6) if max_weight is not None else 0.0,
        f"avg_{weight_column}": round(weight_sum / n_edges, 6) if n_edges else 0.0,
    }


def save_stats(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    catalog_output = resolve_path(base_dir, args.catalog_output)
    community_output = resolve_path(base_dir, args.community_output)
    stats_output = resolve_path(base_dir, args.stats_output)

    common_game_ids = load_common_game_ids(common_ids_input)

    catalog_stats = filter_projection(
        input_path=catalog_input,
        output_path=catalog_output,
        common_game_ids=common_game_ids,
        threshold=args.catalog_threshold,
        shared_column="shared_tags",
        weight_column="jaccard_tags",
    )
    community_stats = filter_projection(
        input_path=community_input,
        output_path=community_output,
        common_game_ids=common_game_ids,
        threshold=args.community_threshold,
        shared_column="shared_users",
        weight_column="jaccard_users",
    )

    stats = {
        "common_game_universe_size": len(common_game_ids),
        "catalog_final_projection": catalog_stats,
        "community_final_projection": community_stats,
    }
    save_stats(stats, stats_output)

    print(f"common_games: {stats['common_game_universe_size']}")
    print(f"catalog_final_edges: {catalog_stats['n_edges']}")
    print(f"community_final_edges: {community_stats['n_edges']}")
    print(f"catalog_output: {catalog_output}")
    print(f"community_output: {community_output}")
    print(f"stats_output: {stats_output}")


if __name__ == "__main__":
    main()
