from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_CATALOG_INPUT = Path("data/processed/game_tag_edges_filtered.csv")
DEFAULT_COMMUNITY_INPUT = Path("data/processed/user_game_edges.csv")
DEFAULT_GAME_ID_MAP_INPUT = Path("data/processed/game_id_map.csv")
DEFAULT_COMMON_IDS_OUTPUT = Path("data/processed/common_game_ids.csv")
DEFAULT_COMMON_ID_MAP_OUTPUT = Path("data/processed/common_game_id_map.csv")
DEFAULT_STATS_OUTPUT = Path("data/processed/common_game_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constroi a intersecao oficial de jogos entre catalogo e comunidade."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Diretorio raiz do projeto.",
    )
    parser.add_argument(
        "--catalog-input",
        type=Path,
        default=DEFAULT_CATALOG_INPUT,
        help="Caminho relativo ao base-dir para game_tag_edges_filtered.csv.",
    )
    parser.add_argument(
        "--community-input",
        type=Path,
        default=DEFAULT_COMMUNITY_INPUT,
        help="Caminho relativo ao base-dir para user_game_edges.csv.",
    )
    parser.add_argument(
        "--game-id-map-input",
        type=Path,
        default=DEFAULT_GAME_ID_MAP_INPUT,
        help="Caminho relativo ao base-dir para game_id_map.csv.",
    )
    parser.add_argument(
        "--common-ids-output",
        type=Path,
        default=DEFAULT_COMMON_IDS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar common_game_ids.csv.",
    )
    parser.add_argument(
        "--common-id-map-output",
        type=Path,
        default=DEFAULT_COMMON_ID_MAP_OUTPUT,
        help="Caminho relativo ao base-dir para salvar common_game_id_map.csv.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT,
        help="Caminho relativo ao base-dir para salvar common_game_stats.json.",
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


def load_distinct_game_ids(input_path: Path, required_column: str) -> set[int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    game_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, (required_column,))

        for row in reader:
            raw_value = str(row[required_column]).strip()
            if raw_value == "":
                raise AssertionError(
                    f"Encontrado {required_column} vazio em {input_path.name}: {row}"
                )
            game_ids.add(int(raw_value))

    if not game_ids:
        raise AssertionError(f"Nenhum jogo encontrado em {input_path.name}")

    return game_ids


def load_game_names(input_path: Path) -> dict[int, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    names_by_id: dict[int, str] = {}
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_required_columns(reader.fieldnames, ("app_id", "name"))

        for row in reader:
            app_id_raw = str(row["app_id"]).strip()
            name = str(row["name"]).strip()
            if app_id_raw == "" or name == "":
                raise AssertionError(
                    f"Encontrado registro invalido em {input_path.name}: {row}"
                )

            app_id = int(app_id_raw)
            if app_id in names_by_id and names_by_id[app_id] != name:
                raise AssertionError(
                    f"Nome inconsistente para app_id {app_id} em {input_path.name}"
                )
            names_by_id[app_id] = name

    if not names_by_id:
        raise AssertionError(f"Nenhum mapeamento encontrado em {input_path.name}")

    return names_by_id


def build_stats(
    catalog_game_ids: set[int],
    community_game_ids: set[int],
    common_game_ids: list[int],
) -> dict[str, int | float]:
    n_catalog = len(catalog_game_ids)
    n_community = len(community_game_ids)
    n_common = len(common_game_ids)
    n_catalog_only = len(catalog_game_ids.difference(community_game_ids))
    n_community_only = len(community_game_ids.difference(catalog_game_ids))

    return {
        "n_catalog_games": n_catalog,
        "n_community_games": n_community,
        "n_common_games": n_common,
        "n_catalog_only_games": n_catalog_only,
        "n_community_only_games": n_community_only,
        "catalog_overlap_pct": round(100 * n_common / n_catalog, 2),
        "community_overlap_pct": round(100 * n_common / n_community, 2),
    }


def save_common_ids(common_game_ids: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["app_id"])
        for app_id in common_game_ids:
            writer.writerow([app_id])


def save_common_id_map(
    common_game_ids: list[int],
    names_by_id: dict[int, str],
    output_path: Path,
) -> None:
    missing_names = [app_id for app_id in common_game_ids if app_id not in names_by_id]
    if missing_names:
        raise AssertionError(
            "Existem jogos da intersecao ausentes em game_id_map.csv: "
            f"{missing_names[:10]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["app_id", "name"])
        for app_id in common_game_ids:
            writer.writerow([app_id, names_by_id[app_id]])


def save_stats(stats: dict[str, int | float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    catalog_input = resolve_path(base_dir, args.catalog_input)
    community_input = resolve_path(base_dir, args.community_input)
    game_id_map_input = resolve_path(base_dir, args.game_id_map_input)
    common_ids_output = resolve_path(base_dir, args.common_ids_output)
    common_id_map_output = resolve_path(base_dir, args.common_id_map_output)
    stats_output = resolve_path(base_dir, args.stats_output)

    catalog_game_ids = load_distinct_game_ids(catalog_input, "app_id")
    community_game_ids = load_distinct_game_ids(community_input, "app_id")
    common_game_ids = sorted(catalog_game_ids.intersection(community_game_ids))

    if not common_game_ids:
        raise AssertionError("A intersecao entre catalogo e comunidade nao pode ser vazia")

    names_by_id = load_game_names(game_id_map_input)
    stats = build_stats(catalog_game_ids, community_game_ids, common_game_ids)

    save_common_ids(common_game_ids, common_ids_output)
    save_common_id_map(common_game_ids, names_by_id, common_id_map_output)
    save_stats(stats, stats_output)

    print(f"catalog_games: {stats['n_catalog_games']}")
    print(f"community_games: {stats['n_community_games']}")
    print(f"common_games: {stats['n_common_games']}")
    print(f"catalog_overlap_pct: {stats['catalog_overlap_pct']:.2f}")
    print(f"community_overlap_pct: {stats['community_overlap_pct']:.2f}")
    print(
        "Arquivos salvos em: "
        f"{common_ids_output}, {common_id_map_output} e {stats_output}"
    )


if __name__ == "__main__":
    main()
