from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tag_cleaning import run_analysis


DEFAULT_GAMES_INPUT = Path("data/raw/games.csv")
DEFAULT_EDGES_OUTPUT = Path("data/processed/game_tag_edges_filtered.csv")
DEFAULT_GAMES_OUTPUT = Path("data/processed/games_with_tags.csv")
DEFAULT_GAME_ID_MAP_OUTPUT = Path("data/processed/game_id_map.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/catalog_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera os datasets finais do catalogo com tags limpas."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--games-input",
        type=Path,
        default=DEFAULT_GAMES_INPUT,
        help="Caminho relativo ao base-dir para o catalogo bruto de jogos.",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=DEFAULT_EDGES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar as arestas jogo-tag filtradas.",
    )
    parser.add_argument(
        "--games-output",
        type=Path,
        default=DEFAULT_GAMES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar os jogos com tags finais.",
    )
    parser.add_argument(
        "--game-id-map-output",
        type=Path,
        default=DEFAULT_GAME_ID_MAP_OUTPUT,
        help="Caminho relativo ao base-dir para salvar o mapeamento app_id -> nome.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar as estatisticas do catalogo.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def build_outputs(
    base_dir: Path, games_input: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = run_analysis(base_dir)

    edges_df = (
        results["final_pairs_df"][["app_id", "tag"]]
        .dropna(subset=["app_id", "tag"])
        .drop_duplicates(subset=["app_id", "tag"])
        .sort_values(["app_id", "tag"], kind="stable")
        .reset_index(drop=True)
    )
    edges_df["app_id"] = edges_df["app_id"].astype("int64")
    edges_df["tag"] = edges_df["tag"].astype(str).str.strip()

    if edges_df.empty:
        raise AssertionError("Nenhuma aresta jogo-tag foi gerada")
    if edges_df["tag"].eq("").any():
        raise AssertionError("Existem tags vazias na saida final")
    if edges_df.duplicated(subset=["app_id", "tag"]).any():
        raise AssertionError("Existem arestas duplicadas na saida final")

    games_df = pd.read_csv(games_input)
    if "app_id" not in games_df.columns:
        raise AssertionError("O catalogo de jogos precisa conter a coluna app_id")
    if "name" in games_df.columns:
        name_column = "name"
    elif "title" in games_df.columns:
        name_column = "title"
    else:
        raise AssertionError("O catalogo de jogos precisa conter a coluna name ou title")

    games_df = games_df.copy()
    games_df["app_id"] = games_df["app_id"].astype("int64")

    edge_app_ids = pd.Index(edges_df["app_id"].unique())
    missing_in_games = edge_app_ids.difference(games_df["app_id"])
    if not missing_in_games.empty:
        sample = missing_in_games[:10].tolist()
        raise AssertionError(
            "Existem jogos nas arestas filtradas ausentes de games.csv: "
            f"{sample}"
        )

    games_with_tags_df = (
        games_df[games_df["app_id"].isin(edge_app_ids)]
        .drop_duplicates(subset=["app_id"])
        .sort_values("app_id", kind="stable")
        .reset_index(drop=True)
    )

    if games_with_tags_df.empty:
        raise AssertionError("Nenhum jogo com tags finais foi encontrado")
    if games_with_tags_df["app_id"].duplicated().any():
        raise AssertionError("games_with_tags.csv contem app_id duplicado")

    games_with_tags_ids = pd.Index(games_with_tags_df["app_id"])
    if not edge_app_ids.equals(games_with_tags_ids):
        missing_in_games_with_tags = edge_app_ids.difference(games_with_tags_ids)
        extra_in_games_with_tags = games_with_tags_ids.difference(edge_app_ids)
        raise AssertionError(
            "Inconsistencia entre jogos filtrados e arestas finais: "
            f"faltando={missing_in_games_with_tags[:10].tolist()} "
            f"extras={extra_in_games_with_tags[:10].tolist()}"
        )

    game_id_map_df = (
        games_with_tags_df[["app_id", name_column]]
        .rename(columns={name_column: "name"})
        .dropna(subset=["app_id", "name"])
        .copy()
    )
    game_id_map_df["name"] = game_id_map_df["name"].astype(str).str.strip()
    game_id_map_df = (
        game_id_map_df[game_id_map_df["name"] != ""]
        .drop_duplicates(subset=["app_id"])
        .sort_values("app_id", kind="stable")
        .reset_index(drop=True)
    )

    if game_id_map_df.empty:
        raise AssertionError("game_id_map.csv ficou vazio")
    if game_id_map_df["app_id"].duplicated().any():
        raise AssertionError("game_id_map.csv contem app_id duplicado")
    if game_id_map_df["name"].isna().any() or game_id_map_df["name"].eq("").any():
        raise AssertionError("game_id_map.csv contem nomes nulos ou vazios")

    game_id_map_ids = pd.Index(game_id_map_df["app_id"])
    if not games_with_tags_ids.equals(game_id_map_ids):
        missing_in_map = games_with_tags_ids.difference(game_id_map_ids)
        extra_in_map = game_id_map_ids.difference(games_with_tags_ids)
        raise AssertionError(
            "Inconsistencia entre games_with_tags.csv e game_id_map.csv: "
            f"faltando={missing_in_map[:10].tolist()} "
            f"extras={extra_in_map[:10].tolist()}"
        )

    return edges_df, games_with_tags_df, game_id_map_df


def build_catalog_stats(
    edges_df: pd.DataFrame, games_with_tags_df: pd.DataFrame
) -> dict[str, int | float]:
    n_games = int(len(games_with_tags_df))
    n_edges = int(len(edges_df))
    avg_tags_per_game = round(n_edges / n_games, 4) if n_games else 0.0
    return {
        "n_games": n_games,
        "n_edges": n_edges,
        "avg_tags_per_game": avg_tags_per_game,
    }


def save_outputs(
    edges_df: pd.DataFrame,
    games_with_tags_df: pd.DataFrame,
    game_id_map_df: pd.DataFrame,
    stats: dict[str, int | float],
    edges_output: Path,
    games_output: Path,
    game_id_map_output: Path,
    stats_output: Path,
) -> None:
    edges_output.parent.mkdir(parents=True, exist_ok=True)
    games_output.parent.mkdir(parents=True, exist_ok=True)
    game_id_map_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)

    edges_df.to_csv(edges_output, index=False)
    games_with_tags_df.to_csv(games_output, index=False)
    game_id_map_df.to_csv(game_id_map_output, index=False)
    stats_output.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    games_input = resolve_path(base_dir, args.games_input)
    edges_output = resolve_path(base_dir, args.edges_output)
    games_output = resolve_path(base_dir, args.games_output)
    game_id_map_output = resolve_path(base_dir, args.game_id_map_output)
    stats_output = resolve_path(base_dir, args.stats_output)

    edges_df, games_with_tags_df, game_id_map_df = build_outputs(base_dir, games_input)
    stats = build_catalog_stats(edges_df, games_with_tags_df)
    save_outputs(
        edges_df,
        games_with_tags_df,
        game_id_map_df,
        stats,
        edges_output,
        games_output,
        game_id_map_output,
        stats_output,
    )

    print(f"games_with_tags.csv: {stats['n_games']} jogos")
    print(f"game_tag_edges_filtered.csv: {stats['n_edges']} arestas")
    print(f"game_id_map.csv: {len(game_id_map_df)} jogos")
    print(f"tags distintas: {edges_df['tag'].nunique()}")
    print(f"avg_tags_per_game: {stats['avg_tags_per_game']:.4f}")
    print(
        "Arquivos salvos em: "
        f"{edges_output}, {games_output}, {game_id_map_output} e {stats_output}"
    )


if __name__ == "__main__":
    main()
