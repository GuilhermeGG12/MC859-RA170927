from __future__ import annotations

import argparse
import csv
import json
import os
from array import array
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_GAME_TAG_GRAPH_INPUT = Path("graphs/game_tag.graphml")
DEFAULT_USER_GAME_EDGES_INPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/bipartite_analysis_stats.json")
DEFAULT_GAME_TAG_DEGREE_FIGURE = Path("figures/game_tag_degree_distribution.png")
DEFAULT_GAME_TAG_COMPONENT_FIGURE = Path(
    "figures/game_tag_component_size_distribution.png"
)
DEFAULT_USER_GAME_DEGREE_FIGURE = Path("figures/user_game_degree_distribution.png")
DEFAULT_USER_GAME_COMPONENT_FIGURE = Path("figures/user_game_component_size_distribution.png")


class DisjointSetUnion:
    def __init__(self, n_nodes: int) -> None:
        self.parent = array("I", range(n_nodes))
        self.size = array("I", [1]) * n_nodes
        self.rank = bytearray(n_nodes)

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent

        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa os grafos bipartidos jogo-tag e usuario-jogo."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--game-tag-graph-input",
        type=Path,
        default=DEFAULT_GAME_TAG_GRAPH_INPUT,
        help="Caminho relativo ao base-dir para graphs/game_tag.graphml.",
    )
    parser.add_argument(
        "--user-game-edges-input",
        type=Path,
        default=DEFAULT_USER_GAME_EDGES_INPUT,
        help="Caminho relativo ao base-dir para data/processed/user_game_edges.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar bipartite_analysis_stats.json.",
    )
    parser.add_argument(
        "--game-tag-degree-figure",
        type=Path,
        default=DEFAULT_GAME_TAG_DEGREE_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de graus de game_tag.",
    )
    parser.add_argument(
        "--game-tag-component-figure",
        type=Path,
        default=DEFAULT_GAME_TAG_COMPONENT_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de componentes de game_tag.",
    )
    parser.add_argument(
        "--user-game-degree-figure",
        type=Path,
        default=DEFAULT_USER_GAME_DEGREE_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de graus de user_game.",
    )
    parser.add_argument(
        "--user-game-component-figure",
        type=Path,
        default=DEFAULT_USER_GAME_COMPONENT_FIGURE,
        help="Caminho relativo ao base-dir para salvar a figura de componentes de user_game.",
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


def counter_to_json(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def average_degree(n_nodes: int, n_edges: int) -> float:
    if n_nodes == 0:
        return 0.0
    return round((2 * n_edges) / n_nodes, 4)


def save_distribution_figure(
    distribution: Counter[int],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = sorted(distribution)
    y_values = [distribution[x] for x in x_values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_values, y_values, width=0.9, color="#2a6f97", edgecolor="#184e77")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    if max(x_values) > 100:
        ax.set_xscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_degree_figure(
    global_distribution: Counter[int],
    left_distribution: Counter[int],
    right_distribution: Counter[int],
    output_path: Path,
    graph_name: str,
    left_label: str,
    right_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    distributions = [
        ("Distribuicao global de graus", global_distribution),
        (f"Distribuicao de graus ({left_label})", left_distribution),
        (f"Distribuicao de graus ({right_label})", right_distribution),
    ]

    for ax, (title, distribution) in zip(axes, distributions):
        x_values = sorted(distribution)
        y_values = [distribution[x] for x in x_values]
        ax.bar(x_values, y_values, width=0.9, color="#2a6f97", edgecolor="#184e77")
        ax.set_title(f"{graph_name}: {title}")
        ax.set_xlabel("Grau")
        ax.set_ylabel("Frequencia")
        ax.set_yscale("log")
        if max(x_values) > 100:
            ax.set_xscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_game_tag_graph(
    input_path: Path,
    degree_figure_path: Path,
    component_figure_path: Path,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    graph = nx.read_graphml(input_path)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise AssertionError("graphs/game_tag.graphml nao pode estar vazio")

    node_types = nx.get_node_attributes(graph, "type")
    if len(node_types) != graph.number_of_nodes():
        raise AssertionError("Todos os nos de game_tag.graphml precisam conter atributo type")

    global_degree_distribution: Counter[int] = Counter()
    game_degree_distribution: Counter[int] = Counter()
    tag_degree_distribution: Counter[int] = Counter()

    for node, degree in graph.degree():
        global_degree_distribution[degree] += 1
        node_type = node_types[node]
        if node_type == "game":
            game_degree_distribution[degree] += 1
        elif node_type == "tag":
            tag_degree_distribution[degree] += 1
        else:
            raise AssertionError(f"Tipo de no invalido em game_tag.graphml: {node_type}")

    component_sizes = sorted(
        (len(component) for component in nx.connected_components(graph)), reverse=True
    )
    component_size_distribution = Counter(component_sizes)

    save_degree_figure(
        global_distribution=global_degree_distribution,
        left_distribution=game_degree_distribution,
        right_distribution=tag_degree_distribution,
        output_path=degree_figure_path,
        graph_name="game_tag",
        left_label="jogos",
        right_label="tags",
    )

    component_figure_generated = False
    if len(component_sizes) > 1:
        save_distribution_figure(
            distribution=component_size_distribution,
            output_path=component_figure_path,
            title="game_tag: distribuicao de tamanhos das componentes",
            x_label="Tamanho da componente",
            y_label="Numero de componentes",
        )
        component_figure_generated = True

    return {
        "n_vertices": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "avg_degree": average_degree(graph.number_of_nodes(), graph.number_of_edges()),
        "degree_distribution": counter_to_json(global_degree_distribution),
        "degree_distribution_by_partition": {
            "game": counter_to_json(game_degree_distribution),
            "tag": counter_to_json(tag_degree_distribution),
        },
        "n_components": int(len(component_sizes)),
        "component_size_distribution": (
            counter_to_json(component_size_distribution) if len(component_sizes) > 1 else None
        ),
        "component_size_distribution_figure_generated": component_figure_generated,
    }


def scan_user_game_metadata(
    input_path: Path,
) -> tuple[dict[str, object], dict[int, int], Counter[int], Counter[int], Counter[int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    n_edges = 0
    n_users = 0
    current_user_id: int | None = None
    current_user_degree = 0
    previous_pair: tuple[int, int] | None = None

    game_index_map: dict[int, int] = {}
    game_degree_counts: dict[int, int] = {}
    user_degree_distribution: Counter[int] = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("user_id", "app_id"))

        for row in reader:
            user_raw = str(row["user_id"]).strip()
            app_raw = str(row["app_id"]).strip()
            if user_raw == "" or app_raw == "":
                raise AssertionError(
                    f"Encontrada aresta invalida em {input_path.name}: {row}"
                )

            user_id = int(user_raw)
            app_id = int(app_raw)
            pair = (user_id, app_id)

            if previous_pair is not None:
                if pair == previous_pair:
                    raise AssertionError(
                        f"Existe aresta duplicada em {input_path.name}: {pair}"
                    )
                if pair < previous_pair:
                    raise AssertionError(
                        "user_game_edges.csv precisa estar ordenado por (user_id, app_id)"
                    )
            previous_pair = pair

            if current_user_id is None:
                current_user_id = user_id
                n_users = 1
                current_user_degree = 0
            elif user_id != current_user_id:
                user_degree_distribution[current_user_degree] += 1
                current_user_id = user_id
                n_users += 1
                current_user_degree = 0

            current_user_degree += 1
            n_edges += 1

            if app_id not in game_index_map:
                game_index_map[app_id] = len(game_index_map)
                game_degree_counts[app_id] = 0
            game_degree_counts[app_id] += 1

    if current_user_id is None:
        raise AssertionError("user_game_edges.csv nao pode estar vazio")

    user_degree_distribution[current_user_degree] += 1

    game_degree_distribution: Counter[int] = Counter(game_degree_counts.values())
    global_degree_distribution = user_degree_distribution + game_degree_distribution

    n_games = len(game_index_map)
    n_vertices = n_users + n_games

    stats = {
        "n_vertices": int(n_vertices),
        "n_edges": int(n_edges),
        "avg_degree": average_degree(n_vertices, n_edges),
        "n_users": int(n_users),
        "n_games": int(n_games),
    }
    return (
        stats,
        game_index_map,
        global_degree_distribution,
        user_degree_distribution,
        game_degree_distribution,
    )


def compute_user_game_components(
    input_path: Path, n_users: int, game_index_map: dict[int, int]
) -> Counter[int]:
    n_nodes = n_users + len(game_index_map)
    dsu = DisjointSetUnion(n_nodes)

    current_user_id: int | None = None
    current_user_index = -1

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("user_id", "app_id"))

        for row in reader:
            user_id = int(str(row["user_id"]).strip())
            app_id = int(str(row["app_id"]).strip())

            if current_user_id is None or user_id != current_user_id:
                current_user_index += 1
                current_user_id = user_id

            game_index = n_users + game_index_map[app_id]
            dsu.union(current_user_index, game_index)

    component_size_distribution: Counter[int] = Counter()
    for node_index in range(n_nodes):
        root = dsu.find(node_index)
        if root == node_index:
            component_size_distribution[int(dsu.size[root])] += 1

    return component_size_distribution


def analyze_user_game_graph(
    input_path: Path,
    degree_figure_path: Path,
    component_figure_path: Path,
) -> dict[str, object]:
    (
        stats,
        game_index_map,
        global_degree_distribution,
        user_degree_distribution,
        game_degree_distribution,
    ) = scan_user_game_metadata(input_path)

    component_size_distribution = compute_user_game_components(
        input_path=input_path,
        n_users=int(stats["n_users"]),
        game_index_map=game_index_map,
    )

    save_degree_figure(
        global_distribution=global_degree_distribution,
        left_distribution=user_degree_distribution,
        right_distribution=game_degree_distribution,
        output_path=degree_figure_path,
        graph_name="user_game",
        left_label="usuarios",
        right_label="jogos",
    )

    save_distribution_figure(
        distribution=component_size_distribution,
        output_path=component_figure_path,
        title="user_game: distribuicao de tamanhos das componentes",
        x_label="Tamanho da componente",
        y_label="Numero de componentes",
    )

    return {
        **stats,
        "degree_distribution": counter_to_json(global_degree_distribution),
        "degree_distribution_by_partition": {
            "user": counter_to_json(user_degree_distribution),
            "game": counter_to_json(game_degree_distribution),
        },
        "n_components": int(sum(component_size_distribution.values())),
        "component_size_distribution": counter_to_json(component_size_distribution),
        "component_size_distribution_figure_generated": True,
    }


def write_stats(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    game_tag_graph_input = resolve_path(base_dir, args.game_tag_graph_input)
    user_game_edges_input = resolve_path(base_dir, args.user_game_edges_input)
    stats_output = resolve_path(base_dir, args.stats_output)
    game_tag_degree_figure = resolve_path(base_dir, args.game_tag_degree_figure)
    game_tag_component_figure = resolve_path(base_dir, args.game_tag_component_figure)
    user_game_degree_figure = resolve_path(base_dir, args.user_game_degree_figure)
    user_game_component_figure = resolve_path(base_dir, args.user_game_component_figure)

    game_tag_stats = analyze_game_tag_graph(
        input_path=game_tag_graph_input,
        degree_figure_path=game_tag_degree_figure,
        component_figure_path=game_tag_component_figure,
    )
    user_game_stats = analyze_user_game_graph(
        input_path=user_game_edges_input,
        degree_figure_path=user_game_degree_figure,
        component_figure_path=user_game_component_figure,
    )

    write_stats(
        {"game_tag": game_tag_stats, "user_game": user_game_stats},
        stats_output,
    )


if __name__ == "__main__":
    main()
