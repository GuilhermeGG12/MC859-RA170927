from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx


DEFAULT_GAME_TAG_INPUT = Path("data/processed/game_tag_edges_filtered.csv")
DEFAULT_USER_GAME_INPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_GAME_TAG_OUTPUT = Path("graphs/game_tag.graphml")
DEFAULT_USER_GAME_OUTPUT = Path("graphs/user_game.graphml")
DEFAULT_STATS_OUTPUT = Path("data/processed/bipartite_stats.json")
DEFAULT_MAX_USER_GRAPHML_SIZE_GB = 1.0

USER_GAME_GRAPHML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="type" attr.type="string"/>
  <key id="d1" for="node" attr.name="user_id" attr.type="long"/>
  <key id="d2" for="node" attr.name="app_id" attr.type="long"/>
  <graph id="G" edgedefault="undirected">
"""
USER_GAME_GRAPHML_FOOTER = """  </graph>
</graphml>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constroi os grafos bipartidos jogo-tag e usuario-jogo em GraphML."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--game-tag-input",
        type=Path,
        default=DEFAULT_GAME_TAG_INPUT,
        help="Caminho relativo ao base-dir para game_tag_edges_filtered.csv.",
    )
    parser.add_argument(
        "--user-game-input",
        type=Path,
        default=DEFAULT_USER_GAME_INPUT,
        help="Caminho relativo ao base-dir para user_game_edges.csv.",
    )
    parser.add_argument(
        "--game-tag-output",
        type=Path,
        default=DEFAULT_GAME_TAG_OUTPUT,
        help="Caminho relativo ao base-dir para salvar game_tag.graphml.",
    )
    parser.add_argument(
        "--user-game-output",
        type=Path,
        default=DEFAULT_USER_GAME_OUTPUT,
        help="Caminho relativo ao base-dir para salvar user_game.graphml.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar bipartite_stats.json.",
    )
    parser.add_argument(
        "--max-user-graphml-size-gb",
        type=float,
        default=DEFAULT_MAX_USER_GRAPHML_SIZE_GB,
        help=(
            "Tamanho maximo estimado do user_game.graphml para escrita automatica. "
            "Use --force-user-game-graphml para ignorar o limite."
        ),
    )
    parser.add_argument(
        "--force-user-game-graphml",
        action="store_true",
        help="Escreve user_game.graphml mesmo se a estimativa ultrapassar o limite.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def node_id(node_type: str, raw_value: str) -> str:
    return f"{node_type}:{raw_value}"


def average_degree_from_counts(n_nodes: int, n_edges: int) -> float:
    if n_nodes == 0:
        return 0.0
    return round((2 * n_edges) / n_nodes, 4)


def validate_required_columns(fieldnames: list[str] | None, required: tuple[str, str]) -> None:
    if fieldnames is None:
        raise AssertionError("CSV sem cabecalho")

    missing_columns = [column for column in required if column not in fieldnames]
    if missing_columns:
        raise AssertionError(
            f"CSV precisa conter as colunas {required}; faltando={missing_columns}"
        )


def build_game_tag_graph(input_path: Path) -> nx.Graph:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    graph = nx.Graph()
    graph.graph["graph_type"] = "bipartite"
    graph.graph["left_partition"] = "game"
    graph.graph["right_partition"] = "tag"

    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id", "tag"))

        for row in reader:
            app_id_raw = str(row["app_id"]).strip()
            tag = str(row["tag"]).strip()

            if app_id_raw == "" or tag == "":
                raise AssertionError(
                    f"Encontrada aresta invalida em {input_path.name}: {row}"
                )

            game_node = node_id("game", app_id_raw)
            tag_node = node_id("tag", tag)

            if game_node not in graph:
                graph.add_node(game_node, type="game", app_id=int(app_id_raw))
            if tag_node not in graph:
                graph.add_node(tag_node, type="tag", tag=tag)
            if graph.has_edge(game_node, tag_node):
                raise AssertionError(
                    f"Existe aresta duplicada em {input_path.name}: ({app_id_raw}, {tag})"
                )

            graph.add_edge(game_node, tag_node)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise AssertionError(f"Grafo vazio gerado a partir de {input_path.name}")

    return graph


def build_game_tag_stats(graph: nx.Graph) -> dict[str, int | float]:
    n_nodes = int(graph.number_of_nodes())
    n_edges = int(graph.number_of_edges())
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": average_degree_from_counts(n_nodes, n_edges),
    }


def user_node_xml(user_id: str) -> str:
    return (
        f'    <node id="user:{user_id}">'
        f'<data key="d0">user</data>'
        f'<data key="d1">{user_id}</data>'
        f"</node>\n"
    )


def game_node_xml(app_id: str) -> str:
    return (
        f'    <node id="game:{app_id}">'
        f'<data key="d0">game</data>'
        f'<data key="d2">{app_id}</data>'
        f"</node>\n"
    )


def user_game_edge_xml(user_id: str, app_id: str) -> str:
    return f'    <edge source="user:{user_id}" target="game:{app_id}"/>\n'


def scan_user_game_graph(input_path: Path) -> tuple[dict[str, int | float | bool], list[int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    n_users = 0
    n_edges = 0
    unique_games: set[int] = set()
    estimated_bytes = len(USER_GAME_GRAPHML_HEADER) + len(USER_GAME_GRAPHML_FOOTER)

    previous_pair: tuple[int, int] | None = None
    last_user_id: int | None = None

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

            if user_id != last_user_id:
                n_users += 1
                estimated_bytes += len(user_node_xml(user_raw))
                last_user_id = user_id

            if app_id not in unique_games:
                unique_games.add(app_id)

            n_edges += 1
            estimated_bytes += len(user_game_edge_xml(user_raw, app_raw))

    sorted_game_ids = sorted(unique_games)
    for app_id in sorted_game_ids:
        estimated_bytes += len(game_node_xml(str(app_id)))

    n_games = len(sorted_game_ids)
    n_nodes = n_users + n_games

    if n_nodes == 0 or n_edges == 0:
        raise AssertionError(f"Grafo vazio gerado a partir de {input_path.name}")

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": average_degree_from_counts(n_nodes, n_edges),
        "n_users": n_users,
        "n_games": n_games,
        "graphml_estimated_size_bytes": estimated_bytes,
        "graphml_estimated_size_gb": round(estimated_bytes / (1024 ** 3), 4),
        "graphml_written": False,
    }
    return stats, sorted_game_ids


def write_game_tag_graph(graph: nx.Graph, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
    return int(output_path.stat().st_size)


def write_user_game_graphml(
    input_path: Path, output_path: Path, sorted_game_ids: list[int]
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(USER_GAME_GRAPHML_HEADER)

        last_user_id: int | None = None
        with input_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            validate_required_columns(reader.fieldnames, ("user_id", "app_id"))

            for row in reader:
                user_raw = str(row["user_id"]).strip()
                user_id = int(user_raw)
                if user_id != last_user_id:
                    output_file.write(user_node_xml(user_raw))
                    last_user_id = user_id

        for app_id in sorted_game_ids:
            output_file.write(game_node_xml(str(app_id)))

        previous_pair: tuple[int, int] | None = None
        with input_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            validate_required_columns(reader.fieldnames, ("user_id", "app_id"))

            for row in reader:
                user_raw = str(row["user_id"]).strip()
                app_raw = str(row["app_id"]).strip()
                pair = (int(user_raw), int(app_raw))

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
                output_file.write(user_game_edge_xml(user_raw, app_raw))

        output_file.write(USER_GAME_GRAPHML_FOOTER)

    return int(output_path.stat().st_size)


def write_stats(stats: dict[str, dict[str, int | float | bool]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    game_tag_input = resolve_path(base_dir, args.game_tag_input)
    user_game_input = resolve_path(base_dir, args.user_game_input)
    game_tag_output = resolve_path(base_dir, args.game_tag_output)
    user_game_output = resolve_path(base_dir, args.user_game_output)
    stats_output = resolve_path(base_dir, args.stats_output)

    game_tag_graph = build_game_tag_graph(game_tag_input)
    game_tag_stats = build_game_tag_stats(game_tag_graph)
    game_tag_stats["graphml_size_bytes"] = write_game_tag_graph(
        game_tag_graph, game_tag_output
    )

    user_game_stats, sorted_game_ids = scan_user_game_graph(user_game_input)
    max_size_bytes = int(args.max_user_graphml_size_gb * (1024 ** 3))
    should_write_user_graph = (
        args.force_user_game_graphml
        or int(user_game_stats["graphml_estimated_size_bytes"]) <= max_size_bytes
    )

    if should_write_user_graph:
        user_game_stats["graphml_size_bytes"] = write_user_game_graphml(
            user_game_input, user_game_output, sorted_game_ids
        )
        user_game_stats["graphml_written"] = True
    else:
        print(
            "user_game.graphml nao foi escrito: estimativa acima do limite "
            f"({user_game_stats['graphml_estimated_size_gb']} GB > "
            f"{args.max_user_graphml_size_gb} GB). "
            "Use --force-user-game-graphml para gerar mesmo assim."
        )

    stats = {
        "game_tag": game_tag_stats,
        "user_game": user_game_stats,
    }
    write_stats(stats, stats_output)


if __name__ == "__main__":
    main()
