from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/processed/user_game_edges_base.csv")
DEFAULT_EDGES_OUTPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/community_stats.json")
DEFAULT_FILTER_SUMMARY_OUTPUT = Path("data/processed/community_filter_summary.json")
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_MIN_USER_GAMES = 2
DEFAULT_MAX_USER_GAMES = 30
DEFAULT_MIN_GAME_USERS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera a versao final oficial do dataset usuario-jogo."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Caminho relativo ao base-dir para user_game_edges_base.csv.",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=DEFAULT_EDGES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar user_game_edges.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar community_stats.json.",
    )
    parser.add_argument(
        "--filter-summary-output",
        type=Path,
        default=DEFAULT_FILTER_SUMMARY_OUTPUT,
        help="Caminho relativo ao base-dir para salvar community_filter_summary.json.",
    )
    parser.add_argument(
        "--min-user-games",
        type=int,
        default=DEFAULT_MIN_USER_GAMES,
        help="Grau minimo final por usuario.",
    )
    parser.add_argument(
        "--max-user-games",
        type=int,
        default=DEFAULT_MAX_USER_GAMES,
        help="Grau maximo final por usuario.",
    )
    parser.add_argument(
        "--min-game-users",
        type=int,
        default=DEFAULT_MIN_GAME_USERS,
        help="Grau minimo final por jogo.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Numero de linhas por chunk na leitura e escrita.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA cache_size = -200000")
    return conn


def load_base_edges(input_path: Path, conn: sqlite3.Connection, chunk_size: int) -> dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    conn.execute("DROP TABLE IF EXISTS base_edges")
    conn.execute(
        """
        CREATE TABLE base_edges (
            user_id INTEGER NOT NULL,
            app_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, app_id)
        )
        """
    )

    rows = 0
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        expected_columns = ["user_id", "app_id"]
        if list(chunk.columns) != expected_columns:
            raise AssertionError(f"Colunas esperadas em {input_path}: {expected_columns}")
        chunk = chunk.dropna(subset=expected_columns)
        chunk = chunk.astype({"user_id": "int64", "app_id": "int64"})
        rows += int(len(chunk))
        conn.executemany(
            "INSERT OR IGNORE INTO base_edges (user_id, app_id) VALUES (?, ?)",
            chunk.itertuples(index=False, name=None),
        )
        conn.commit()

    conn.execute("CREATE INDEX IF NOT EXISTS idx_base_edges_app_id ON base_edges(app_id)")
    conn.commit()

    n_edges = int(conn.execute("SELECT COUNT(*) FROM base_edges").fetchone()[0])
    n_users = int(conn.execute("SELECT COUNT(DISTINCT user_id) FROM base_edges").fetchone()[0])
    n_games = int(conn.execute("SELECT COUNT(DISTINCT app_id) FROM base_edges").fetchone()[0])

    if n_edges == 0:
        raise AssertionError("Dataset usuario-jogo base esta vazio")

    return {
        "input_rows": rows,
        "base_n_users": n_users,
        "base_n_games": n_games,
        "base_n_edges": n_edges,
    }


def apply_final_filters(
    conn: sqlite3.Connection,
    min_user_games: int,
    max_user_games: int,
    min_game_users: int,
) -> dict[str, int]:
    conn.execute("DROP TABLE IF EXISTS user_counts_initial")
    conn.execute(
        """
        CREATE TABLE user_counts_initial AS
        SELECT user_id, COUNT(*) AS n_games
        FROM base_edges
        GROUP BY user_id
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_counts_initial_user_id "
        "ON user_counts_initial(user_id)"
    )

    conn.execute("DROP TABLE IF EXISTS user_filtered_edges")
    conn.execute(
        """
        CREATE TABLE user_filtered_edges AS
        SELECT e.user_id, e.app_id
        FROM base_edges AS e
        INNER JOIN user_counts_initial AS c
            ON c.user_id = e.user_id
        WHERE c.n_games BETWEEN ? AND ?
        ORDER BY e.user_id, e.app_id
        """,
        (min_user_games, max_user_games),
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_filtered_edges_pair "
        "ON user_filtered_edges(user_id, app_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_filtered_edges_app_id "
        "ON user_filtered_edges(app_id)"
    )

    conn.execute("DROP TABLE IF EXISTS game_counts_after_user_filter")
    conn.execute(
        """
        CREATE TABLE game_counts_after_user_filter AS
        SELECT app_id, COUNT(*) AS n_users
        FROM user_filtered_edges
        GROUP BY app_id
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_counts_after_user_filter_app_id "
        "ON game_counts_after_user_filter(app_id)"
    )

    conn.execute("DROP TABLE IF EXISTS working_edges")
    conn.execute(
        """
        CREATE TABLE working_edges AS
        SELECT user_id, app_id
        FROM user_filtered_edges
        ORDER BY user_id, app_id
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_working_edges_pair "
        "ON working_edges(user_id, app_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_working_edges_app_id "
        "ON working_edges(app_id)"
    )
    conn.commit()

    iterations = 0
    total_removed_low_game_degree_edges = 0
    total_removed_low_user_degree_edges = 0

    while True:
        iterations += 1
        conn.execute("DROP TABLE IF EXISTS low_degree_games")
        conn.execute(
            """
            CREATE TABLE low_degree_games AS
            SELECT app_id
            FROM working_edges
            GROUP BY app_id
            HAVING COUNT(*) < ?
            """,
            (min_game_users,),
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_low_degree_games_app_id "
            "ON low_degree_games(app_id)"
        )
        cursor = conn.execute(
            """
            DELETE FROM working_edges
            WHERE app_id IN (SELECT app_id FROM low_degree_games)
            """
        )
        removed_game_edges = int(cursor.rowcount if cursor.rowcount != -1 else 0)
        total_removed_low_game_degree_edges += removed_game_edges

        conn.execute("DROP TABLE IF EXISTS low_degree_users")
        conn.execute(
            """
            CREATE TABLE low_degree_users AS
            SELECT user_id
            FROM working_edges
            GROUP BY user_id
            HAVING COUNT(*) < ?
            """,
            (min_user_games,),
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_low_degree_users_user_id "
            "ON low_degree_users(user_id)"
        )
        cursor = conn.execute(
            """
            DELETE FROM working_edges
            WHERE user_id IN (SELECT user_id FROM low_degree_users)
            """
        )
        removed_user_edges = int(cursor.rowcount if cursor.rowcount != -1 else 0)
        total_removed_low_user_degree_edges += removed_user_edges
        conn.commit()

        if removed_game_edges == 0 and removed_user_edges == 0:
            break

    conn.execute("DROP TABLE IF EXISTS final_edges")
    conn.execute(
        """
        CREATE TABLE final_edges AS
        SELECT user_id, app_id
        FROM working_edges
        ORDER BY user_id, app_id
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_final_edges_pair "
        "ON final_edges(user_id, app_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_final_edges_app_id ON final_edges(app_id)")
    conn.commit()

    min_final_game_degree = int(
        conn.execute(
            """
            SELECT MIN(n_users)
            FROM (
                SELECT COUNT(*) AS n_users
                FROM final_edges
                GROUP BY app_id
            )
            """
        ).fetchone()[0]
        or 0
    )
    if min_final_game_degree < min_game_users:
        raise AssertionError(
            "Filtro final deixou jogos abaixo do minimo. "
            f"min_final_game_degree={min_final_game_degree}"
        )

    return {
        "n_users_after_user_filter": int(
            conn.execute("SELECT COUNT(DISTINCT user_id) FROM user_filtered_edges").fetchone()[0]
        ),
        "n_games_after_user_filter": int(
            conn.execute("SELECT COUNT(DISTINCT app_id) FROM user_filtered_edges").fetchone()[0]
        ),
        "n_edges_after_user_filter": int(
            conn.execute("SELECT COUNT(*) FROM user_filtered_edges").fetchone()[0]
        ),
        "n_games_after_game_filter": int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM game_counts_after_user_filter
                WHERE n_users >= ?
                """,
                (min_game_users,),
            ).fetchone()[0]
        ),
        "n_edges_after_game_filter": int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM user_filtered_edges AS e
                INNER JOIN game_counts_after_user_filter AS c
                    ON c.app_id = e.app_id
                WHERE c.n_users >= ?
                """,
                (min_game_users,),
            ).fetchone()[0]
        ),
        "core_pruning_iterations": iterations,
        "removed_low_game_degree_edges_in_core": total_removed_low_game_degree_edges,
        "removed_low_user_degree_edges_in_core": total_removed_low_user_degree_edges,
    }


def build_stats(conn: sqlite3.Connection) -> dict[str, int | float]:
    n_edges = int(conn.execute("SELECT COUNT(*) FROM final_edges").fetchone()[0])
    n_users = int(conn.execute("SELECT COUNT(DISTINCT user_id) FROM final_edges").fetchone()[0])
    n_games = int(conn.execute("SELECT COUNT(DISTINCT app_id) FROM final_edges").fetchone()[0])

    min_games_per_user, max_games_per_user, avg_games_per_user = conn.execute(
        """
        SELECT MIN(n_games), MAX(n_games), AVG(n_games)
        FROM (
            SELECT COUNT(*) AS n_games
            FROM final_edges
            GROUP BY user_id
        )
        """
    ).fetchone()

    min_users_per_game, max_users_per_game, avg_users_per_game = conn.execute(
        """
        SELECT MIN(n_users), MAX(n_users), AVG(n_users)
        FROM (
            SELECT COUNT(*) AS n_users
            FROM final_edges
            GROUP BY app_id
        )
        """
    ).fetchone()

    projection_pairs_from_users = int(
        conn.execute(
            """
            SELECT SUM(n_games * (n_games - 1) / 2)
            FROM (
                SELECT COUNT(*) AS n_games
                FROM final_edges
                GROUP BY user_id
            )
            """
        ).fetchone()[0]
        or 0
    )

    return {
        "n_users": n_users,
        "n_games": n_games,
        "n_edges": n_edges,
        "avg_games_per_user": round(float(avg_games_per_user or 0.0), 4),
        "avg_users_per_game": round(float(avg_users_per_game or 0.0), 4),
        "min_games_per_user": int(min_games_per_user or 0),
        "max_games_per_user": int(max_games_per_user or 0),
        "min_users_per_game": int(min_users_per_game or 0),
        "max_users_per_game": int(max_users_per_game or 0),
        "projection_pairs_from_users": projection_pairs_from_users,
    }


def build_filter_summary(
    load_summary: dict[str, int],
    filter_summary: dict[str, int],
    stats: dict[str, int | float],
    min_user_games: int,
    max_user_games: int,
    min_game_users: int,
) -> dict[str, int | float]:
    return {
        **load_summary,
        **filter_summary,
        "filter_name": "final",
        "min_user_games": min_user_games,
        "max_user_games": max_user_games,
        "min_game_users": min_game_users,
        "n_users_after_filter": int(stats["n_users"]),
        "n_games_after_filter": int(stats["n_games"]),
        "n_edges_after_filter": int(stats["n_edges"]),
        "users_preserved_pct": round(
            100 * int(stats["n_users"]) / load_summary["base_n_users"], 2
        ),
        "games_preserved_pct": round(
            100 * int(stats["n_games"]) / load_summary["base_n_games"], 2
        ),
        "edges_preserved_pct": round(
            100 * int(stats["n_edges"]) / load_summary["base_n_edges"], 2
        ),
    }


def save_outputs(
    conn: sqlite3.Connection,
    edges_output: Path,
    stats_output: Path,
    filter_summary_output: Path,
    stats: dict[str, int | float],
    filter_summary: dict[str, int | float],
    chunk_size: int,
) -> None:
    edges_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    filter_summary_output.parent.mkdir(parents=True, exist_ok=True)

    query = "SELECT user_id, app_id FROM final_edges ORDER BY user_id, app_id"
    exported_rows = 0
    wrote_header = False

    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        chunk = chunk.astype({"user_id": "int64", "app_id": "int64"})
        chunk.to_csv(
            edges_output,
            index=False,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
        )
        exported_rows += int(len(chunk))
        wrote_header = True

    if exported_rows != int(stats["n_edges"]):
        raise AssertionError("Numero de linhas exportadas difere das estatisticas")

    stats_output.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )
    filter_summary_output.write_text(
        json.dumps(filter_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def validate_outputs(
    conn: sqlite3.Connection,
    edges_output: Path,
    stats: dict[str, int | float],
    min_user_games: int,
    max_user_games: int,
    min_game_users: int,
) -> None:
    duplicate_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT user_id, app_id, COUNT(*) AS n
                FROM final_edges
                GROUP BY user_id, app_id
                HAVING n > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_count != 0:
        raise AssertionError("Existem duplicatas de (user_id, app_id) na saida final")

    invalid_users = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT user_id, COUNT(*) AS n_games
                FROM final_edges
                GROUP BY user_id
            )
            WHERE n_games < ? OR n_games > ?
            """,
            (min_user_games, max_user_games),
        ).fetchone()[0]
    )
    if invalid_users != 0:
        raise AssertionError("Existem usuarios finais fora do intervalo permitido")

    invalid_games = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT app_id, COUNT(*) AS n_users
                FROM final_edges
                GROUP BY app_id
            )
            WHERE n_users < ?
            """,
            (min_game_users,),
        ).fetchone()[0]
    )
    if invalid_games != 0:
        raise AssertionError("Existem jogos finais abaixo do minimo permitido")

    if not edges_output.exists() or edges_output.stat().st_size == 0:
        raise AssertionError("user_game_edges.csv nao foi gerado corretamente")

    csv_head = pd.read_csv(edges_output, nrows=5)
    expected_columns = ["user_id", "app_id"]
    if list(csv_head.columns) != expected_columns:
        raise AssertionError("Colunas inesperadas em user_game_edges.csv")

    if not pd.api.types.is_integer_dtype(csv_head["user_id"]):
        raise AssertionError("Tipos inconsistentes em user_game_edges.csv")
    if not pd.api.types.is_integer_dtype(csv_head["app_id"]):
        raise AssertionError("Tipos inconsistentes em user_game_edges.csv")

    sql_n_users = int(
        conn.execute("SELECT COUNT(DISTINCT user_id) FROM final_edges").fetchone()[0]
    )
    sql_n_games = int(
        conn.execute("SELECT COUNT(DISTINCT app_id) FROM final_edges").fetchone()[0]
    )
    sql_n_edges = int(conn.execute("SELECT COUNT(*) FROM final_edges").fetchone()[0])

    if (
        sql_n_users != int(stats["n_users"])
        or sql_n_games != int(stats["n_games"])
        or sql_n_edges != int(stats["n_edges"])
    ):
        raise AssertionError("As estatisticas salvas em JSON estao inconsistentes")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    input_path = resolve_path(base_dir, args.input)
    edges_output = resolve_path(base_dir, args.edges_output)
    stats_output = resolve_path(base_dir, args.stats_output)
    filter_summary_output = resolve_path(base_dir, args.filter_summary_output)

    with tempfile.TemporaryDirectory(prefix="mc859-user-game-final-") as temp_dir:
        conn = connect_sqlite(Path(temp_dir) / "final.sqlite")
        try:
            load_summary = load_base_edges(input_path, conn, args.chunk_size)
            filter_summary_base = apply_final_filters(
                conn=conn,
                min_user_games=args.min_user_games,
                max_user_games=args.max_user_games,
                min_game_users=args.min_game_users,
            )
            stats = build_stats(conn)
            filter_summary = build_filter_summary(
                load_summary=load_summary,
                filter_summary=filter_summary_base,
                stats=stats,
                min_user_games=args.min_user_games,
                max_user_games=args.max_user_games,
                min_game_users=args.min_game_users,
            )
            save_outputs(
                conn=conn,
                edges_output=edges_output,
                stats_output=stats_output,
                filter_summary_output=filter_summary_output,
                stats=stats,
                filter_summary=filter_summary,
                chunk_size=args.chunk_size,
            )
            validate_outputs(
                conn=conn,
                edges_output=edges_output,
                stats=stats,
                min_user_games=args.min_user_games,
                max_user_games=args.max_user_games,
                min_game_users=args.min_game_users,
            )
        finally:
            conn.close()


if __name__ == "__main__":
    main()
