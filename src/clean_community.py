from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_RECOMMENDATIONS_INPUT = Path("data/raw/recommendations.csv")
DEFAULT_EDGES_OUTPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/community_stats.json")
DEFAULT_FILTER_SUMMARY_OUTPUT = Path("data/processed/community_filter_summary.json")
DEFAULT_CHUNK_SIZE = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materializa o dataset final usuario-jogo da comunidade."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--recommendations-input",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_INPUT,
        help="Caminho relativo ao base-dir para recommendations.csv.",
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
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Numero de linhas por chunk na leitura do CSV bruto.",
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


def load_interactions(
    recommendations_input: Path,
    conn: sqlite3.Connection,
    chunk_size: int,
) -> dict[str, int]:
    raw_rows = 0

    conn.execute("DROP TABLE IF EXISTS raw_edges")
    conn.execute(
        """
        CREATE TABLE raw_edges (
            user_id INTEGER NOT NULL,
            app_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, app_id)
        )
        """
    )

    for chunk in pd.read_csv(
        recommendations_input,
        usecols=["user_id", "app_id"],
        chunksize=chunk_size,
    ):
        raw_rows += int(len(chunk))
        chunk = chunk.dropna(subset=["user_id", "app_id"])
        if chunk.empty:
            continue

        chunk = chunk.astype({"user_id": "int64", "app_id": "int64"})
        chunk = chunk[["user_id", "app_id"]]
        chunk = chunk.drop_duplicates(subset=["user_id", "app_id"])

        conn.executemany(
            "INSERT OR IGNORE INTO raw_edges (user_id, app_id) VALUES (?, ?)",
            chunk.itertuples(index=False, name=None),
        )
        conn.commit()

    deduplicated_edges = int(
        conn.execute("SELECT COUNT(*) FROM raw_edges").fetchone()[0]
    )
    n_users_before = int(
        conn.execute("SELECT COUNT(DISTINCT user_id) FROM raw_edges").fetchone()[0]
    )
    n_games_before = int(
        conn.execute("SELECT COUNT(DISTINCT app_id) FROM raw_edges").fetchone()[0]
    )

    if deduplicated_edges == 0:
        raise AssertionError("Nenhuma interacao usuario-jogo foi carregada")

    return {
        "raw_rows": raw_rows,
        "deduplicated_edges": deduplicated_edges,
        "n_users_before_filter": n_users_before,
        "n_games_before_filter": n_games_before,
    }


def apply_user_filters(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS user_counts")
    conn.execute(
        """
        CREATE TABLE user_counts AS
        SELECT user_id, COUNT(*) AS n_games
        FROM raw_edges
        GROUP BY user_id
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_counts_user_id ON user_counts(user_id)")

    conn.execute("DROP TABLE IF EXISTS filtered_users")
    conn.execute(
        """
        CREATE TABLE filtered_users AS
        SELECT user_id, n_games
        FROM user_counts
        WHERE n_games BETWEEN 2 AND 100
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_filtered_users_user_id ON filtered_users(user_id)"
    )

    conn.execute("DROP TABLE IF EXISTS final_edges")
    conn.execute(
        """
        CREATE TABLE final_edges AS
        SELECT r.user_id, r.app_id
        FROM raw_edges AS r
        INNER JOIN filtered_users AS u
            ON u.user_id = r.user_id
        ORDER BY r.user_id, r.app_id
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_final_edges_pair ON final_edges(user_id, app_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_final_edges_app_id ON final_edges(app_id)"
    )
    conn.commit()


def build_stats(conn: sqlite3.Connection) -> dict[str, int | float]:
    n_edges = int(conn.execute("SELECT COUNT(*) FROM final_edges").fetchone()[0])
    n_users = int(conn.execute("SELECT COUNT(*) FROM filtered_users").fetchone()[0])
    n_games = int(
        conn.execute("SELECT COUNT(DISTINCT app_id) FROM final_edges").fetchone()[0]
    )

    min_games_per_user, max_games_per_user, avg_games_per_user = conn.execute(
        """
        SELECT MIN(n_games), MAX(n_games), AVG(n_games)
        FROM filtered_users
        """
    ).fetchone()

    avg_users_per_game = conn.execute(
        """
        SELECT AVG(user_count)
        FROM (
            SELECT COUNT(*) AS user_count
            FROM final_edges
            GROUP BY app_id
        )
        """
    ).fetchone()[0]

    stats = {
        "n_users": n_users,
        "n_games": n_games,
        "n_edges": n_edges,
        "avg_games_per_user": round(float(avg_games_per_user or 0.0), 4),
        "avg_users_per_game": round(float(avg_users_per_game or 0.0), 4),
        "min_games_per_user": int(min_games_per_user or 0),
        "max_games_per_user": int(max_games_per_user or 0),
    }

    if stats["n_edges"] == 0:
        raise AssertionError("user_game_edges.csv nao pode ficar vazio")

    return stats


def build_filter_summary(
    conn: sqlite3.Connection, load_summary: dict[str, int], stats: dict[str, int | float]
) -> dict[str, int | float]:
    removed_single_game_users = int(
        conn.execute("SELECT COUNT(*) FROM user_counts WHERE n_games < 2").fetchone()[0]
    )
    removed_super_active_users = int(
        conn.execute("SELECT COUNT(*) FROM user_counts WHERE n_games > 100").fetchone()[0]
    )

    return {
        **load_summary,
        "n_users_after_filter": int(stats["n_users"]),
        "n_games_after_filter": int(stats["n_games"]),
        "n_edges_after_filter": int(stats["n_edges"]),
        "removed_single_game_users": removed_single_game_users,
        "removed_super_active_users": removed_super_active_users,
        "users_preserved_pct": round(
            100 * int(stats["n_users"]) / load_summary["n_users_before_filter"], 2
        ),
        "games_preserved_pct": round(
            100 * int(stats["n_games"]) / load_summary["n_games_before_filter"], 2
        ),
        "edges_preserved_pct": round(
            100 * int(stats["n_edges"]) / load_summary["deduplicated_edges"], 2
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
        chunk.to_csv(edges_output, index=False, mode="a" if wrote_header else "w", header=not wrote_header)
        exported_rows += int(len(chunk))
        wrote_header = True

    if exported_rows != int(stats["n_edges"]):
        raise AssertionError(
            "Numero de linhas exportadas difere das estatisticas calculadas"
        )

    stats_output.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )
    filter_summary_output.write_text(
        json.dumps(filter_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def validate_outputs(
    conn: sqlite3.Connection, edges_output: Path, stats: dict[str, int | float]
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
            FROM filtered_users
            WHERE n_games < 2 OR n_games > 100
            """
        ).fetchone()[0]
    )
    if invalid_users != 0:
        raise AssertionError("Existem usuarios finais fora do intervalo [2, 100]")

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
        conn.execute("SELECT COUNT(*) FROM filtered_users").fetchone()[0]
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
    recommendations_input = resolve_path(base_dir, args.recommendations_input)
    edges_output = resolve_path(base_dir, args.edges_output)
    stats_output = resolve_path(base_dir, args.stats_output)
    filter_summary_output = resolve_path(base_dir, args.filter_summary_output)

    if not recommendations_input.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {recommendations_input}")

    with tempfile.TemporaryDirectory(prefix="mc859-community-") as temp_dir:
        db_path = Path(temp_dir) / "community.sqlite"
        conn = connect_sqlite(db_path)
        try:
            load_summary = load_interactions(
                recommendations_input=recommendations_input,
                conn=conn,
                chunk_size=args.chunk_size,
            )
            apply_user_filters(conn)
            stats = build_stats(conn)
            filter_summary = build_filter_summary(conn, load_summary, stats)
            save_outputs(
                conn=conn,
                edges_output=edges_output,
                stats_output=stats_output,
                filter_summary_output=filter_summary_output,
                stats=stats,
                filter_summary=filter_summary,
                chunk_size=args.chunk_size,
            )
            validate_outputs(conn, edges_output, stats)
        finally:
            conn.close()

    print(f"usuarios: {stats['n_users']}")
    print(f"jogos: {stats['n_games']}")
    print(f"arestas: {stats['n_edges']}")
    print(f"avg_games_per_user: {stats['avg_games_per_user']:.4f}")
    print(
        "Arquivos salvos em: "
        f"{edges_output}, {stats_output} e {filter_summary_output}"
    )


if __name__ == "__main__":
    main()
