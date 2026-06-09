from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STATS_INPUT = Path("data/processed/bipartite_analysis_stats.json")
DEFAULT_GAME_TAG_GLOBAL_OUTPUT = Path("figures/game_tag_degree_distribution_loglog.png")
DEFAULT_GAME_TAG_GAMES_OUTPUT = Path("figures/game_tag_games_degree_loglog.png")
DEFAULT_GAME_TAG_TAGS_OUTPUT = Path("figures/game_tag_tags_degree_loglog.png")
DEFAULT_USER_GAME_GLOBAL_OUTPUT = Path("figures/user_game_degree_distribution_loglog.png")
DEFAULT_USER_GAME_USERS_OUTPUT = Path("figures/user_game_users_degree_loglog.png")
DEFAULT_USER_GAME_GAMES_OUTPUT = Path("figures/user_game_games_degree_loglog.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera versoes log-log das distribuicoes de grau para relatorio."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--stats-input",
        type=Path,
        default=DEFAULT_STATS_INPUT,
        help="Caminho relativo ao base-dir para bipartite_analysis_stats.json.",
    )
    parser.add_argument(
        "--game-tag-global-output",
        type=Path,
        default=DEFAULT_GAME_TAG_GLOBAL_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao global de game_tag.",
    )
    parser.add_argument(
        "--game-tag-games-output",
        type=Path,
        default=DEFAULT_GAME_TAG_GAMES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao jogos de game_tag.",
    )
    parser.add_argument(
        "--game-tag-tags-output",
        type=Path,
        default=DEFAULT_GAME_TAG_TAGS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao tags de game_tag.",
    )
    parser.add_argument(
        "--user-game-global-output",
        type=Path,
        default=DEFAULT_USER_GAME_GLOBAL_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao global de user_game.",
    )
    parser.add_argument(
        "--user-game-users-output",
        type=Path,
        default=DEFAULT_USER_GAME_USERS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao usuarios de user_game.",
    )
    parser.add_argument(
        "--user-game-games-output",
        type=Path,
        default=DEFAULT_USER_GAME_GAMES_OUTPUT,
        help="Caminho relativo ao base-dir para salvar a versao jogos de user_game.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def load_stats(input_path: Path) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def distribution_to_arrays(distribution: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    degree_values = np.array([int(key) for key in distribution], dtype=np.int64)
    frequencies = np.array([int(value) for value in distribution.values()], dtype=np.int64)

    positive_mask = degree_values > 0
    degree_values = degree_values[positive_mask]
    frequencies = frequencies[positive_mask]

    if degree_values.size == 0:
        raise AssertionError("Distribuicao sem graus positivos para plot em escala log")

    order = np.argsort(degree_values)
    return degree_values[order], frequencies[order]


def build_log_bins(degree_values: np.ndarray, target_bins: int = 24) -> np.ndarray:
    min_degree = int(degree_values.min())
    max_degree = int(degree_values.max())

    if min_degree <= 0:
        raise AssertionError("Escala log requer graus estritamente positivos")

    if min_degree == max_degree:
        upper_edge = max_degree + 1
        return np.array([min_degree, upper_edge], dtype=float)

    return np.logspace(
        np.log10(min_degree),
        np.log10(max_degree),
        num=target_bins,
        endpoint=True,
    )


def save_loglog_histogram(
    distribution: dict[str, int],
    output_path: Path,
    title: str,
    color: str,
) -> None:
    degree_values, frequencies = distribution_to_arrays(distribution)
    bins = build_log_bins(degree_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(
        degree_values,
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


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    stats_input = resolve_path(base_dir, args.stats_input)
    game_tag_global_output = resolve_path(base_dir, args.game_tag_global_output)
    game_tag_games_output = resolve_path(base_dir, args.game_tag_games_output)
    game_tag_tags_output = resolve_path(base_dir, args.game_tag_tags_output)
    user_game_global_output = resolve_path(base_dir, args.user_game_global_output)
    user_game_users_output = resolve_path(base_dir, args.user_game_users_output)
    user_game_games_output = resolve_path(base_dir, args.user_game_games_output)

    stats = load_stats(stats_input)

    game_tag_stats = stats["game_tag"]
    user_game_stats = stats["user_game"]

    save_loglog_histogram(
        distribution=game_tag_stats["degree_distribution"],
        output_path=game_tag_global_output,
        title="Distribuicao global de graus do grafo bipartido jogo-tag",
        color="#2a6f97",
    )
    save_loglog_histogram(
        distribution=game_tag_stats["degree_distribution_by_partition"]["game"],
        output_path=game_tag_games_output,
        title="Distribuicao do numero de tags por jogo",
        color="#468faf",
    )
    save_loglog_histogram(
        distribution=game_tag_stats["degree_distribution_by_partition"]["tag"],
        output_path=game_tag_tags_output,
        title="Distribuicao do numero de jogos por tag",
        color="#014f86",
    )

    save_loglog_histogram(
        distribution=user_game_stats["degree_distribution"],
        output_path=user_game_global_output,
        title="Distribuicao global de graus do grafo bipartido usuario-jogo",
        color="#6a994e",
    )
    save_loglog_histogram(
        distribution=user_game_stats["degree_distribution_by_partition"]["user"],
        output_path=user_game_users_output,
        title="Distribuicao do numero de jogos por usuario",
        color="#90a955",
    )
    save_loglog_histogram(
        distribution=user_game_stats["degree_distribution_by_partition"]["game"],
        output_path=user_game_games_output,
        title="Distribuicao do numero de usuarios por jogo",
        color="#386641",
    )


if __name__ == "__main__":
    main()
