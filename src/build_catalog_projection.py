from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx


DEFAULT_GRAPH_INPUT = Path("graphs/game_tag.graphml")
DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_EDGES_OUTPUT = Path("data/processed/game_game_catalog_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/game_game_catalog_stats.json")
DEFAULT_SQLITE_OUTPUT = Path("data/processed/game_game_catalog_pairs.sqlite")
DEFAULT_MAX_RAW_PAIR_INCIDENCES = 100_000_000
SQLITE_BATCH_SIZE = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constroi a projecao jogo-jogo do catalogo a partir de game_tag.graphml."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--graph-input",
        type=Path,
        default=DEFAULT_GRAPH_INPUT,
        help="Caminho relativo ao base-dir para graphs/game_tag.graphml.",
    )
    parser.add_argument(
        "--common-ids-input",
        type=Path,
        default=DEFAULT_COMMON_IDS_INPUT,
        help="Caminho relativo ao base-dir para common_game_ids.csv.",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=DEFAULT_EDGES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar game_game_catalog_edges.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar game_game_catalog_stats.json.",
    )
    parser.add_argument(
        "--sqlite-output",
        type=Path,
        default=DEFAULT_SQLITE_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o banco SQLite intermediario.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Apenas estima a escala da projecao sem materializar arestas.",
    )
    parser.add_argument(
        "--max-raw-pair-incidences",
        type=int,
        default=DEFAULT_MAX_RAW_PAIR_INCIDENCES,
        help=(
            "Limite de incidencias brutas tag-induzidas permitido para a projecao automatica. "
            "Use --force-large-projection para ignorar o limite."
        ),
    )
    parser.add_argument(
        "--force-large-projection",
        action="store_true",
        help="Executa a projecao mesmo se a estimativa de incidencias brutas exceder o limite.",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Pula a agregacao dos pares e reutiliza um banco SQLite ja existente.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Executa apenas a agregacao dos pares sem exportar CSV ou estatisticas finais.",
    )
    parser.add_argument(
        "--delete-sqlite-after-success",
        action="store_true",
        help="Remove o banco SQLite intermediario ao final de uma execucao bem-sucedida.",
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


def load_common_game_ids(input_path: Path) -> set[int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    common_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id",))

        for row in reader:
            app_id_raw = str(row["app_id"]).strip()
            if app_id_raw == "":
                raise AssertionError(f"app_id vazio em {input_path.name}: {row}")
            common_ids.add(int(app_id_raw))

    if not common_ids:
        raise AssertionError("A lista de jogos comuns nao pode ser vazia")

    return common_ids


def load_game_tag_graph(input_path: Path) -> nx.Graph:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    graph = nx.read_graphml(input_path)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise AssertionError("graphs/game_tag.graphml nao pode estar vazio")

    return graph


def extract_catalog_neighborhoods(
    graph: nx.Graph,
    common_game_ids: set[int],
) -> tuple[dict[int, set[str]], dict[str, list[int]], int]:
    node_types = nx.get_node_attributes(graph, "type")
    if len(node_types) != graph.number_of_nodes():
        raise AssertionError("Todos os nos de game_tag.graphml precisam conter atributo type")

    game_to_tags: dict[int, set[str]] = {}
    tag_to_games: dict[str, list[int]] = defaultdict(list)
    total_graph_games = 0

    for node, attrs in graph.nodes(data=True):
        node_type = node_types[node]
        if node_type != "game":
            continue

        total_graph_games += 1
        app_id = int(attrs["app_id"])
        if app_id not in common_game_ids:
            continue

        tags: set[str] = set()
        for neighbor in graph.neighbors(node):
            neighbor_type = node_types[neighbor]
            if neighbor_type != "tag":
                raise AssertionError("game_tag.graphml deve ligar apenas jogos a tags")
            tag_value = str(graph.nodes[neighbor]["tag"]).strip()
            if tag_value == "":
                raise AssertionError("Encontrada tag vazia em game_tag.graphml")
            tags.add(tag_value)

        if not tags:
            raise AssertionError(f"Jogo {app_id} da intersecao ficou sem tags")

        game_to_tags[app_id] = tags

    if not game_to_tags:
        raise AssertionError("Nenhum jogo da intersecao foi encontrado em game_tag.graphml")

    missing_games = sorted(common_game_ids.difference(game_to_tags))
    if missing_games:
        raise AssertionError(
            "Existem jogos da intersecao ausentes em game_tag.graphml: "
            f"{missing_games[:10]}"
        )

    for app_id, tags in game_to_tags.items():
        for tag in tags:
            tag_to_games[tag].append(app_id)

    for tag, game_ids in tag_to_games.items():
        tag_to_games[tag] = sorted(game_ids)

    return game_to_tags, dict(tag_to_games), total_graph_games


def estimate_raw_pair_incidences(tag_to_games: dict[str, list[int]]) -> int:
    return int(
        sum(len(game_ids) * (len(game_ids) - 1) // 2 for game_ids in tag_to_games.values())
    )


def build_estimate_stats(
    common_game_ids: set[int],
    game_to_tags: dict[int, set[str]],
    tag_to_games: dict[str, list[int]],
    total_graph_games: int,
) -> dict[str, int | float | str]:
    return {
        "source_graph": "graphs/game_tag.graphml",
        "n_graph_games_total": total_graph_games,
        "n_games_in_common_universe": len(common_game_ids),
        "n_games_projected": len(game_to_tags),
        "n_tags_incident_to_common_games": len(tag_to_games),
        "raw_pair_incidences": estimate_raw_pair_incidences(tag_to_games),
        "n_edges": 0,
        "avg_degree": 0.0,
        "min_shared_tags": 0,
        "max_shared_tags": 0,
        "min_jaccard_tags": 0.0,
        "max_jaccard_tags": 0.0,
        "avg_jaccard_tags": 0.0,
    }


def initialize_sqlite_database(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    connection = sqlite3.connect(sqlite_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute(
        """
        CREATE TABLE game_tag_counts (
            app_id INTEGER PRIMARY KEY,
            tag_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE catalog_pairs (
            source_app_id INTEGER NOT NULL,
            target_app_id INTEGER NOT NULL,
            shared_tags INTEGER NOT NULL,
            PRIMARY KEY (source_app_id, target_app_id)
        )
        """
    )
    return connection


def open_existing_sqlite_database(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Banco SQLite nao encontrado: {sqlite_path}")

    connection = sqlite3.connect(sqlite_path)
    return connection


def persist_game_tag_counts(
    connection: sqlite3.Connection,
    game_to_tags: dict[int, set[str]],
) -> None:
    rows = sorted((app_id, len(tags)) for app_id, tags in game_to_tags.items())
    connection.executemany(
        "INSERT INTO game_tag_counts (app_id, tag_count) VALUES (?, ?)",
        rows,
    )
    connection.commit()


def flush_pair_batch(
    connection: sqlite3.Connection,
    pair_batch: dict[tuple[int, int], int],
) -> None:
    if not pair_batch:
        return

    connection.executemany(
        """
        INSERT INTO catalog_pairs (source_app_id, target_app_id, shared_tags)
        VALUES (?, ?, ?)
        ON CONFLICT(source_app_id, target_app_id)
        DO UPDATE SET shared_tags = shared_tags + excluded.shared_tags
        """,
        ((left, right, shared_tags) for (left, right), shared_tags in pair_batch.items()),
    )
    connection.commit()
    pair_batch.clear()


def materialize_projection_pairs(
    connection: sqlite3.Connection,
    tag_to_games: dict[str, list[int]],
) -> None:
    pair_batch: dict[tuple[int, int], int] = {}

    for game_ids in tag_to_games.values():
        if len(game_ids) < 2:
            continue

        for left, right in combinations(game_ids, 2):
            pair_key = (left, right)
            pair_batch[pair_key] = pair_batch.get(pair_key, 0) + 1
            if len(pair_batch) >= SQLITE_BATCH_SIZE:
                flush_pair_batch(connection, pair_batch)

    flush_pair_batch(connection, pair_batch)


def validate_existing_sqlite_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required_tables = {"game_tag_counts", "catalog_pairs"}
    missing_tables = sorted(required_tables.difference(tables))
    if missing_tables:
        raise AssertionError(
            f"Banco SQLite incompleto; faltando tabelas obrigatorias: {missing_tables}"
        )


def export_projection_edges(
    connection: sqlite3.Connection,
    output_path: Path,
) -> dict[str, int | float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            p.source_app_id,
            p.target_app_id,
            p.shared_tags,
            c1.tag_count,
            c2.tag_count
        FROM catalog_pairs AS p
        JOIN game_tag_counts AS c1
            ON c1.app_id = p.source_app_id
        JOIN game_tag_counts AS c2
            ON c2.app_id = p.target_app_id
        ORDER BY p.source_app_id, p.target_app_id
    """

    edge_count = 0
    shared_sum = 0
    min_shared = None
    max_shared = None
    jaccard_sum = 0.0
    min_jaccard = None
    max_jaccard = None

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["source_app_id", "target_app_id", "shared_tags", "jaccard_tags"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for source_app_id, target_app_id, shared_tags, source_tag_count, target_tag_count in (
            connection.execute(query)
        ):
            if source_app_id == target_app_id:
                raise AssertionError("A projecao do catalogo nao pode conter auto-arestas")

            union_size = source_tag_count + target_tag_count - shared_tags
            if union_size <= 0:
                raise AssertionError(
                    f"Uniao invalida para o par ({source_app_id}, {target_app_id})"
                )

            jaccard_tags = round(shared_tags / union_size, 6)
            if not 0.0 < jaccard_tags <= 1.0:
                raise AssertionError(
                    f"Peso Jaccard invalido no par ({source_app_id}, {target_app_id})"
                )

            writer.writerow(
                {
                    "source_app_id": int(source_app_id),
                    "target_app_id": int(target_app_id),
                    "shared_tags": int(shared_tags),
                    "jaccard_tags": jaccard_tags,
                }
            )

            edge_count += 1
            shared_sum += int(shared_tags)
            jaccard_sum += jaccard_tags
            min_shared = shared_tags if min_shared is None else min(min_shared, shared_tags)
            max_shared = shared_tags if max_shared is None else max(max_shared, shared_tags)
            min_jaccard = jaccard_tags if min_jaccard is None else min(min_jaccard, jaccard_tags)
            max_jaccard = jaccard_tags if max_jaccard is None else max(max_jaccard, jaccard_tags)

    return {
        "n_edges": edge_count,
        "min_shared_tags": int(min_shared) if min_shared is not None else 0,
        "max_shared_tags": int(max_shared) if max_shared is not None else 0,
        "avg_shared_tags": round(shared_sum / edge_count, 6) if edge_count else 0.0,
        "min_jaccard_tags": round(min_jaccard, 6) if min_jaccard is not None else 0.0,
        "max_jaccard_tags": round(max_jaccard, 6) if max_jaccard is not None else 0.0,
        "avg_jaccard_tags": round(jaccard_sum / edge_count, 6) if edge_count else 0.0,
    }


def build_final_stats(
    common_game_ids: set[int],
    game_to_tags: dict[int, set[str]],
    tag_to_games: dict[str, list[int]],
    total_graph_games: int,
    edge_stats: dict[str, int | float],
) -> dict[str, int | float | str]:
    n_nodes = len(common_game_ids)
    n_edges = int(edge_stats["n_edges"])
    avg_degree = round((2 * n_edges) / n_nodes, 4) if n_nodes else 0.0

    return {
        "source_graph": "graphs/game_tag.graphml",
        "n_graph_games_total": total_graph_games,
        "n_games_in_common_universe": len(common_game_ids),
        "n_games_projected": len(game_to_tags),
        "n_tags_incident_to_common_games": len(tag_to_games),
        "raw_pair_incidences": estimate_raw_pair_incidences(tag_to_games),
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "min_shared_tags": int(edge_stats["min_shared_tags"]),
        "max_shared_tags": int(edge_stats["max_shared_tags"]),
        "avg_shared_tags": float(edge_stats["avg_shared_tags"]),
        "min_jaccard_tags": float(edge_stats["min_jaccard_tags"]),
        "max_jaccard_tags": float(edge_stats["max_jaccard_tags"]),
        "avg_jaccard_tags": float(edge_stats["avg_jaccard_tags"]),
    }


def save_stats(stats: dict[str, int | float | str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    graph_input = resolve_path(base_dir, args.graph_input)
    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    edges_output = resolve_path(base_dir, args.edges_output)
    stats_output = resolve_path(base_dir, args.stats_output)
    sqlite_output = resolve_path(base_dir, args.sqlite_output)

    common_game_ids = load_common_game_ids(common_ids_input)
    graph = load_game_tag_graph(graph_input)
    game_to_tags, tag_to_games, total_graph_games = extract_catalog_neighborhoods(
        graph,
        common_game_ids,
    )
    estimate_stats = build_estimate_stats(
        common_game_ids,
        game_to_tags,
        tag_to_games,
        total_graph_games,
    )

    if args.estimate_only:
        save_stats(estimate_stats, stats_output)
        print(f"graph_games_total: {estimate_stats['n_graph_games_total']}")
        print(f"common_games: {estimate_stats['n_games_in_common_universe']}")
        print(f"projected_games: {estimate_stats['n_games_projected']}")
        print(f"raw_pair_incidences: {estimate_stats['raw_pair_incidences']}")
        print(f"Arquivos salvos em: {stats_output}")
        return

    if (
        not args.force_large_projection
        and int(estimate_stats["raw_pair_incidences"]) > args.max_raw_pair_incidences
    ):
        raise AssertionError(
            "A projecao do catalogo excede o limite de incidencias brutas permitido. "
            f"raw_pair_incidences={estimate_stats['raw_pair_incidences']} > "
            f"max_raw_pair_incidences={args.max_raw_pair_incidences}. "
            "Use --estimate-only para diagnostico ou --force-large-projection para executar mesmo assim."
        )

    if args.skip_aggregation:
        connection = open_existing_sqlite_database(sqlite_output)
        validate_existing_sqlite_schema(connection)
    else:
        connection = initialize_sqlite_database(sqlite_output)

    run_succeeded = False
    try:
        if not args.skip_aggregation:
            persist_game_tag_counts(connection, game_to_tags)
            materialize_projection_pairs(connection, tag_to_games)

        if args.skip_export:
            print(f"graph_games_total: {estimate_stats['n_graph_games_total']}")
            print(f"common_games: {estimate_stats['n_games_in_common_universe']}")
            print(f"projected_games: {estimate_stats['n_games_projected']}")
            print(f"raw_pair_incidences: {estimate_stats['raw_pair_incidences']}")
            print(f"Banco SQLite salvo em: {sqlite_output}")
            return

        edge_stats = export_projection_edges(connection, edges_output)
        final_stats = build_final_stats(
            common_game_ids,
            game_to_tags,
            tag_to_games,
            total_graph_games,
            edge_stats,
        )
        save_stats(final_stats, stats_output)
        run_succeeded = True
    finally:
        connection.close()
        if run_succeeded and args.delete_sqlite_after_success and sqlite_output.exists():
            sqlite_output.unlink()

    print(f"graph_games_total: {final_stats['n_graph_games_total']}")
    print(f"common_games: {final_stats['n_games_in_common_universe']}")
    print(f"projected_games: {final_stats['n_games_projected']}")
    print(f"projected_edges: {final_stats['n_edges']}")
    print(f"avg_degree: {final_stats['avg_degree']:.4f}")
    print(f"max_shared_tags: {final_stats['max_shared_tags']}")
    print(f"max_jaccard_tags: {final_stats['max_jaccard_tags']:.6f}")
    print("Arquivos salvos em: " f"{edges_output} e {stats_output}")


if __name__ == "__main__":
    main()
