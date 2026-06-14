from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from itertools import combinations
from pathlib import Path


DEFAULT_GRAPH_REFERENCE = Path("graphs/user_game.graphml")
DEFAULT_USER_GAME_INPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_COMMON_IDS_INPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_EDGES_OUTPUT = Path("data/processed/game_game_community_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/game_game_community_stats.json")
DEFAULT_SQLITE_OUTPUT = Path("data/processed/game_game_community_pairs.sqlite")
DEFAULT_MAX_RAW_PAIR_INCIDENCES = 300_000_000
SQLITE_BATCH_SIZE = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Constroi a projecao jogo-jogo da comunidade a partir da lista de arestas "
            "equivalente ao bipartido oficial user_game.graphml."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--graph-reference",
        type=Path,
        default=DEFAULT_GRAPH_REFERENCE,
        help="Caminho relativo ao base-dir para graphs/user_game.graphml.",
    )
    parser.add_argument(
        "--user-game-input",
        type=Path,
        default=DEFAULT_USER_GAME_INPUT,
        help="Caminho relativo ao base-dir para user_game_edges.csv.",
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
        help="Caminho relativo ao base-dir para salvar game_game_community_edges.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar game_game_community_stats.json.",
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
            "Limite de incidencias brutas usuario-induzidas permitido para a projecao automatica. "
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


def validate_inputs(graph_reference: Path, user_game_input: Path) -> None:
    if not graph_reference.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {graph_reference}")
    if not user_game_input.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {user_game_input}")


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
        CREATE TABLE game_user_counts (
            app_id INTEGER PRIMARY KEY,
            user_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE community_pairs (
            source_app_id INTEGER NOT NULL,
            target_app_id INTEGER NOT NULL,
            shared_users INTEGER NOT NULL,
            PRIMARY KEY (source_app_id, target_app_id)
        )
        """
    )
    return connection


def open_existing_sqlite_database(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Banco SQLite nao encontrado: {sqlite_path}")
    return sqlite3.connect(sqlite_path)


def validate_existing_sqlite_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required_tables = {"game_user_counts", "community_pairs"}
    missing_tables = sorted(required_tables.difference(tables))
    if missing_tables:
        raise AssertionError(
            f"Banco SQLite incompleto; faltando tabelas obrigatorias: {missing_tables}"
        )


def flush_counts_batch(
    connection: sqlite3.Connection,
    count_batch: dict[int, int],
) -> None:
    if not count_batch:
        return

    connection.executemany(
        """
        INSERT INTO game_user_counts (app_id, user_count)
        VALUES (?, ?)
        ON CONFLICT(app_id) DO UPDATE SET user_count = user_count + excluded.user_count
        """,
        sorted(count_batch.items()),
    )
    connection.commit()
    count_batch.clear()


def flush_pairs_batch(
    connection: sqlite3.Connection,
    pair_batch: dict[tuple[int, int], int],
) -> None:
    if not pair_batch:
        return

    connection.executemany(
        """
        INSERT INTO community_pairs (source_app_id, target_app_id, shared_users)
        VALUES (?, ?, ?)
        ON CONFLICT(source_app_id, target_app_id)
        DO UPDATE SET shared_users = shared_users + excluded.shared_users
        """,
        ((left, right, shared) for (left, right), shared in pair_batch.items()),
    )
    connection.commit()
    pair_batch.clear()


def process_user_games(
    game_ids: list[int],
    count_batch: dict[int, int],
    pair_batch: dict[tuple[int, int], int],
) -> int:
    if not game_ids:
        return 0

    for app_id in game_ids:
        count_batch[app_id] = count_batch.get(app_id, 0) + 1

    raw_pair_incidences = 0
    if len(game_ids) >= 2:
        raw_pair_incidences = len(game_ids) * (len(game_ids) - 1) // 2
        for left, right in combinations(game_ids, 2):
            pair_key = (left, right)
            pair_batch[pair_key] = pair_batch.get(pair_key, 0) + 1

    return raw_pair_incidences


def scan_or_aggregate_projection(
    user_game_input: Path,
    common_game_ids: set[int],
    connection: sqlite3.Connection | None,
) -> dict[str, int]:
    n_rows_total = 0
    n_users_total = 0
    n_users_with_common_games = 0
    raw_pair_incidences = 0
    previous_pair: tuple[int, int] | None = None
    current_user_id: int | None = None
    current_user_games: list[int] = []
    projected_games: set[int] = set()
    count_batch: dict[int, int] = {}
    pair_batch: dict[tuple[int, int], int] = {}

    with user_game_input.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("user_id", "app_id"))

        for row in reader:
            user_raw = str(row["user_id"]).strip()
            app_raw = str(row["app_id"]).strip()
            if user_raw == "" or app_raw == "":
                raise AssertionError(
                    f"Encontrada aresta invalida em {user_game_input.name}: {row}"
                )

            user_id = int(user_raw)
            app_id = int(app_raw)
            pair = (user_id, app_id)
            n_rows_total += 1

            if previous_pair is not None:
                if pair == previous_pair:
                    raise AssertionError(
                        f"Existe aresta duplicada em {user_game_input.name}: {pair}"
                    )
                if pair < previous_pair:
                    raise AssertionError(
                        "user_game_edges.csv precisa estar ordenado por (user_id, app_id)"
                    )
            previous_pair = pair

            if current_user_id is None:
                current_user_id = user_id
                n_users_total = 1
            elif user_id != current_user_id:
                if current_user_games:
                    n_users_with_common_games += 1
                    raw_pair_incidences += process_user_games(
                        current_user_games,
                        count_batch,
                        pair_batch,
                    )
                    projected_games.update(current_user_games)
                    if connection is not None:
                        if len(count_batch) >= SQLITE_BATCH_SIZE:
                            flush_counts_batch(connection, count_batch)
                        if len(pair_batch) >= SQLITE_BATCH_SIZE:
                            flush_pairs_batch(connection, pair_batch)
                current_user_games = []
                current_user_id = user_id
                n_users_total += 1

            if app_id in common_game_ids:
                current_user_games.append(app_id)

        if current_user_id is None:
            raise AssertionError(f"Nenhuma aresta encontrada em {user_game_input.name}")

        if current_user_games:
            n_users_with_common_games += 1
            raw_pair_incidences += process_user_games(
                current_user_games,
                count_batch,
                pair_batch,
            )
            projected_games.update(current_user_games)

    if connection is not None:
        flush_counts_batch(connection, count_batch)
        flush_pairs_batch(connection, pair_batch)

    missing_games = sorted(common_game_ids.difference(projected_games))
    if missing_games:
        raise AssertionError(
            "Existem jogos da intersecao ausentes na comunidade oficial: "
            f"{missing_games[:10]}"
        )

    return {
        "n_rows_total": n_rows_total,
        "n_users_total": n_users_total,
        "n_users_with_common_games": n_users_with_common_games,
        "n_games_in_common_universe": len(common_game_ids),
        "n_games_projected": len(projected_games),
        "raw_pair_incidences": raw_pair_incidences,
    }


def build_estimate_stats(scan_stats: dict[str, int]) -> dict[str, int | float | str]:
    return {
        "source_graph": "graphs/user_game.graphml",
        "implementation_input": "data/processed/user_game_edges.csv",
        "n_rows_total": scan_stats["n_rows_total"],
        "n_users_total": scan_stats["n_users_total"],
        "n_users_with_common_games": scan_stats["n_users_with_common_games"],
        "n_games_in_common_universe": scan_stats["n_games_in_common_universe"],
        "n_games_projected": scan_stats["n_games_projected"],
        "raw_pair_incidences": scan_stats["raw_pair_incidences"],
        "n_edges": 0,
        "avg_degree": 0.0,
        "min_shared_users": 0,
        "max_shared_users": 0,
        "avg_shared_users": 0.0,
        "min_jaccard_users": 0.0,
        "max_jaccard_users": 0.0,
        "avg_jaccard_users": 0.0,
    }


def export_projection_edges(
    connection: sqlite3.Connection,
    output_path: Path,
) -> dict[str, int | float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            p.source_app_id,
            p.target_app_id,
            p.shared_users,
            c1.user_count,
            c2.user_count
        FROM community_pairs AS p
        JOIN game_user_counts AS c1
            ON c1.app_id = p.source_app_id
        JOIN game_user_counts AS c2
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
        fieldnames = ["source_app_id", "target_app_id", "shared_users", "jaccard_users"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for source_app_id, target_app_id, shared_users, source_user_count, target_user_count in (
            connection.execute(query)
        ):
            if source_app_id == target_app_id:
                raise AssertionError("A projecao da comunidade nao pode conter auto-arestas")

            union_size = source_user_count + target_user_count - shared_users
            if union_size <= 0:
                raise AssertionError(
                    f"Uniao invalida para o par ({source_app_id}, {target_app_id})"
                )

            jaccard_users = round(shared_users / union_size, 6)
            if not 0.0 < jaccard_users <= 1.0:
                raise AssertionError(
                    f"Peso Jaccard invalido no par ({source_app_id}, {target_app_id})"
                )

            writer.writerow(
                {
                    "source_app_id": int(source_app_id),
                    "target_app_id": int(target_app_id),
                    "shared_users": int(shared_users),
                    "jaccard_users": jaccard_users,
                }
            )

            edge_count += 1
            shared_sum += int(shared_users)
            jaccard_sum += jaccard_users
            min_shared = shared_users if min_shared is None else min(min_shared, shared_users)
            max_shared = shared_users if max_shared is None else max(max_shared, shared_users)
            min_jaccard = (
                jaccard_users if min_jaccard is None else min(min_jaccard, jaccard_users)
            )
            max_jaccard = (
                jaccard_users if max_jaccard is None else max(max_jaccard, jaccard_users)
            )

    return {
        "n_edges": edge_count,
        "min_shared_users": int(min_shared) if min_shared is not None else 0,
        "max_shared_users": int(max_shared) if max_shared is not None else 0,
        "avg_shared_users": round(shared_sum / edge_count, 6) if edge_count else 0.0,
        "min_jaccard_users": round(min_jaccard, 6) if min_jaccard is not None else 0.0,
        "max_jaccard_users": round(max_jaccard, 6) if max_jaccard is not None else 0.0,
        "avg_jaccard_users": round(jaccard_sum / edge_count, 6) if edge_count else 0.0,
    }


def build_final_stats(
    scan_stats: dict[str, int],
    edge_stats: dict[str, int | float],
) -> dict[str, int | float | str]:
    n_nodes = scan_stats["n_games_in_common_universe"]
    n_edges = int(edge_stats["n_edges"])
    avg_degree = round((2 * n_edges) / n_nodes, 4) if n_nodes else 0.0

    return {
        "source_graph": "graphs/user_game.graphml",
        "implementation_input": "data/processed/user_game_edges.csv",
        "n_rows_total": scan_stats["n_rows_total"],
        "n_users_total": scan_stats["n_users_total"],
        "n_users_with_common_games": scan_stats["n_users_with_common_games"],
        "n_games_in_common_universe": scan_stats["n_games_in_common_universe"],
        "n_games_projected": scan_stats["n_games_projected"],
        "raw_pair_incidences": scan_stats["raw_pair_incidences"],
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "min_shared_users": int(edge_stats["min_shared_users"]),
        "max_shared_users": int(edge_stats["max_shared_users"]),
        "avg_shared_users": float(edge_stats["avg_shared_users"]),
        "min_jaccard_users": float(edge_stats["min_jaccard_users"]),
        "max_jaccard_users": float(edge_stats["max_jaccard_users"]),
        "avg_jaccard_users": float(edge_stats["avg_jaccard_users"]),
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

    graph_reference = resolve_path(base_dir, args.graph_reference)
    user_game_input = resolve_path(base_dir, args.user_game_input)
    common_ids_input = resolve_path(base_dir, args.common_ids_input)
    edges_output = resolve_path(base_dir, args.edges_output)
    stats_output = resolve_path(base_dir, args.stats_output)
    sqlite_output = resolve_path(base_dir, args.sqlite_output)

    validate_inputs(graph_reference, user_game_input)
    common_game_ids = load_common_game_ids(common_ids_input)

    if args.skip_aggregation:
        connection = open_existing_sqlite_database(sqlite_output)
        validate_existing_sqlite_schema(connection)
        scan_stats = scan_or_aggregate_projection(user_game_input, common_game_ids, None)
    else:
        connection = initialize_sqlite_database(sqlite_output)
        scan_stats = scan_or_aggregate_projection(user_game_input, common_game_ids, connection)

    estimate_stats = build_estimate_stats(scan_stats)

    if args.estimate_only:
        connection.close()
        save_stats(estimate_stats, stats_output)
        print(f"users_total: {estimate_stats['n_users_total']}")
        print(f"common_games: {estimate_stats['n_games_in_common_universe']}")
        print(f"projected_games: {estimate_stats['n_games_projected']}")
        print(f"raw_pair_incidences: {estimate_stats['raw_pair_incidences']}")
        print(f"Arquivos salvos em: {stats_output}")
        return

    if (
        not args.force_large_projection
        and int(estimate_stats["raw_pair_incidences"]) > args.max_raw_pair_incidences
    ):
        connection.close()
        raise AssertionError(
            "A projecao da comunidade excede o limite de incidencias brutas permitido. "
            f"raw_pair_incidences={estimate_stats['raw_pair_incidences']} > "
            f"max_raw_pair_incidences={args.max_raw_pair_incidences}. "
            "Use --estimate-only para diagnostico ou --force-large-projection para executar mesmo assim."
        )

    run_succeeded = False
    try:
        if args.skip_export:
            print(f"users_total: {estimate_stats['n_users_total']}")
            print(f"common_games: {estimate_stats['n_games_in_common_universe']}")
            print(f"projected_games: {estimate_stats['n_games_projected']}")
            print(f"raw_pair_incidences: {estimate_stats['raw_pair_incidences']}")
            print(f"Banco SQLite salvo em: {sqlite_output}")
            run_succeeded = True
            return

        edge_stats = export_projection_edges(connection, edges_output)
        final_stats = build_final_stats(scan_stats, edge_stats)
        save_stats(final_stats, stats_output)
        run_succeeded = True
    finally:
        connection.close()
        if run_succeeded and args.delete_sqlite_after_success and sqlite_output.exists():
            sqlite_output.unlink()

    print(f"users_total: {final_stats['n_users_total']}")
    print(f"common_games: {final_stats['n_games_in_common_universe']}")
    print(f"projected_games: {final_stats['n_games_projected']}")
    print(f"projected_edges: {final_stats['n_edges']}")
    print(f"avg_degree: {final_stats['avg_degree']:.4f}")
    print(f"max_shared_users: {final_stats['max_shared_users']}")
    print(f"max_jaccard_users: {final_stats['max_jaccard_users']:.6f}")
    print("Arquivos salvos em: " f"{edges_output} e {stats_output}")


if __name__ == "__main__":
    main()
