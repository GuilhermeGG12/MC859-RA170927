from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pandas as pd


BASELINE_RESULTS = {
    "final_tag_count": 334,
    "avg_tags_per_game": 7.2438,
    "games_with_final_tags": 42464,
    "pairs_final": 312255,
    "top_tags": [
        "Story Rich",
        "Pixel Graphics",
        "Exploration",
        "Early Access",
        "Fantasy",
        "Roguelike",
        "Anime",
        "Survival",
        "Horror",
        "Arcade",
    ],
}

KEY_STRUCTURAL_TAGS = [
    "Action",
    "Adventure",
    "RPG",
    "Strategy",
    "Simulation",
    "Shooter",
    "Platformer",
    "Puzzle",
    "JRPG",
    "4X",
    "Souls-like",
    "Metroidvania",
    "Roguelike",
    "Roguelite",
    "Action RPG",
    "Turn-Based Strategy",
    "Real-Time Strategy",
    "Survival Horror",
    "Deckbuilder",
    "MOBA",
    "Visual Novel",
]


def load_tag_config(base_dir: Path):
    config_path = base_dir / "src" / "config" / "tags.py"
    spec = importlib.util.spec_from_file_location("tag_config", config_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {config_path}")
    spec.loader.exec_module(module)
    return module


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "app_id": row.get("app_id"),
                    "description": row.get("description", ""),
                    "tags": row.get("tags") or [],
                }
            )
    metadata_df = pd.DataFrame(records)
    if metadata_df.empty:
        raise AssertionError("O arquivo de metadata nao pode estar vazio")
    if not {"app_id", "tags"}.issubset(metadata_df.columns):
        raise AssertionError("Colunas esperadas ausentes")
    if not metadata_df["app_id"].notna().all():
        raise AssertionError("Existem app_id nulos")
    if not metadata_df["tags"].map(lambda value: isinstance(value, list)).all():
        raise AssertionError("Todas as tags devem ser listas")
    return metadata_df


def run_analysis(base_dir: Path) -> dict:
    tag_config = load_tag_config(base_dir)
    metadata_path = base_dir / "data" / "raw" / "games_metadata.json"

    metadata_df = load_metadata(metadata_path)

    tag_pairs_df = (
        metadata_df[["app_id", "tags"]]
        .explode("tags")
        .rename(columns={"tags": "tag"})
        .dropna(subset=["tag"])
    )
    tag_pairs_df["tag"] = tag_pairs_df["tag"].astype(str).str.strip()
    tag_pairs_df = tag_pairs_df[tag_pairs_df["tag"] != ""].reset_index(drop=True)

    normalized_pairs_df = tag_pairs_df.copy()
    normalized_pairs_df["tag_original"] = normalized_pairs_df["tag"]
    normalized_pairs_df["tag"] = normalized_pairs_df["tag"].map(
        lambda value: tag_config.TAG_NORMALIZATION.get(value, value)
    )

    without_unwanted_df = normalized_pairs_df[
        ~normalized_pairs_df["tag"].isin(tag_config.TAGS_TO_REMOVE)
    ].copy()
    game_count_after_category_filter = without_unwanted_df["app_id"].nunique()

    tag_freq_after_category_filter = Counter(without_unwanted_df["tag"])
    too_frequent_tags = {
        tag
        for tag, freq in tag_freq_after_category_filter.items()
        if tag not in tag_config.ALWAYS_KEEP_TAGS
        and game_count_after_category_filter > 0
        and (freq / game_count_after_category_filter) > tag_config.MAX_TAG_PERCENT
    }
    rare_tags = {
        tag
        for tag, freq in tag_freq_after_category_filter.items()
        if tag not in tag_config.ALWAYS_KEEP_TAGS and freq < tag_config.MIN_TAG_FREQ
    }

    final_pairs_df = without_unwanted_df[
        ~without_unwanted_df["tag"].isin(too_frequent_tags | rare_tags)
    ].copy()

    raw_tag_freq = (
        tag_pairs_df["tag"].value_counts().rename_axis("tag").reset_index(name="freq")
    )
    normalized_tag_freq = (
        normalized_pairs_df["tag"]
        .value_counts()
        .rename_axis("tag")
        .reset_index(name="freq")
    )
    final_tag_freq = (
        final_pairs_df["tag"]
        .value_counts()
        .rename_axis("tag")
        .reset_index(name="freq")
    )

    normalization_changes = normalized_pairs_df[
        normalized_pairs_df["tag_original"] != normalized_pairs_df["tag"]
    ].copy()
    normalization_summary = (
        normalization_changes.groupby(["tag_original", "tag"])
        .size()
        .reset_index(name="freq")
        .sort_values(["freq", "tag_original"], ascending=[False, True])
        .reset_index(drop=True)
    )

    tags_per_game = (
        final_pairs_df.groupby("app_id")["tag"].nunique().rename("num_tags")
    )
    raw_tags_per_game = tag_pairs_df.groupby("app_id")["tag"].count()

    too_frequent_df = pd.DataFrame(
        {
            "tag": sorted(too_frequent_tags),
            "freq": [tag_freq_after_category_filter[tag] for tag in sorted(too_frequent_tags)],
            "percentual_sobre_jogos": [
                round(tag_freq_after_category_filter[tag] / game_count_after_category_filter, 4)
                for tag in sorted(too_frequent_tags)
            ],
        }
    )
    rare_tags_df = pd.DataFrame(
        {
            "tag": sorted(rare_tags),
            "freq": [tag_freq_after_category_filter[tag] for tag in sorted(rare_tags)],
        }
    )
    filter_summary_df = pd.DataFrame(
        {
            "etapa": [
                "pares brutos",
                "apos normalizacao",
                "apos remover blacklist semantica",
                "apos filtros de frequencia/raridade",
            ],
            "pares": [
                len(tag_pairs_df),
                len(normalized_pairs_df),
                len(without_unwanted_df),
                len(final_pairs_df),
            ],
        }
    )

    final_tag_count = final_pairs_df["tag"].nunique()
    avg_tags_per_game = float(tags_per_game.mean()) if not tags_per_game.empty else 0.0
    games_with_final_tags = int(tags_per_game.shape[0])
    top_30_final_tags = final_tag_freq.head(30).reset_index(drop=True)

    suspicious_survivors_df = (
        final_tag_freq[final_tag_freq["tag"].isin(tag_config.NON_STRUCTURAL_AUDIT_TAGS)]
        .reset_index(drop=True)
    )
    preserved_structural_df = pd.DataFrame(
        {
            "tag": [tag for tag in KEY_STRUCTURAL_TAGS if tag in set(final_tag_freq["tag"])],
            "freq": [
                int(final_tag_freq.loc[final_tag_freq["tag"] == tag, "freq"].iloc[0])
                for tag in KEY_STRUCTURAL_TAGS
                if tag in set(final_tag_freq["tag"])
            ],
        }
    )

    summary_df = pd.DataFrame(
        {
            "metrica": [
                "numero_final_de_tags",
                "media_tags_por_jogo",
                "jogos_com_tags_finais",
                "pares_finais",
            ],
            "valor": [
                int(final_tag_count),
                round(avg_tags_per_game, 4),
                games_with_final_tags,
                int(len(final_pairs_df)),
            ],
        }
    )
    comparison_df = pd.DataFrame(
        {
            "metrica": [
                "numero_final_de_tags",
                "media_tags_por_jogo",
                "jogos_com_tags_finais",
                "pares_finais",
            ],
            "antes": [
                BASELINE_RESULTS["final_tag_count"],
                BASELINE_RESULTS["avg_tags_per_game"],
                BASELINE_RESULTS["games_with_final_tags"],
                BASELINE_RESULTS["pairs_final"],
            ],
            "depois": [
                int(final_tag_count),
                round(avg_tags_per_game, 4),
                games_with_final_tags,
                int(len(final_pairs_df)),
            ],
        }
    )

    non_protected_final_freq = final_tag_freq[
        ~final_tag_freq["tag"].isin(tag_config.ALWAYS_KEEP_TAGS)
    ].copy()

    assert raw_tag_freq["freq"].sum() == len(tag_pairs_df), "Frequencia bruta inconsistente"
    assert normalized_tag_freq["freq"].sum() == len(normalized_pairs_df), "Frequencia normalizada inconsistente"
    assert final_tag_count == len(final_tag_freq), "Contagem final de tags inconsistente"
    assert set(final_pairs_df["tag"]).issubset(set(normalized_pairs_df["tag"])), "Tags finais devem vir da base normalizada"
    assert final_pairs_df["tag"].isin(tag_config.TAGS_TO_REMOVE).sum() == 0, "Ainda existem tags da blacklist na saida final"
    assert final_pairs_df["tag"].isin(too_frequent_tags).sum() == 0, "Ainda existem tags removiveis por alta frequencia na saida final"
    assert final_pairs_df["tag"].isin(rare_tags).sum() == 0, "Ainda existem tags removiveis por raridade na saida final"
    if not non_protected_final_freq.empty:
        assert non_protected_final_freq["freq"].min() >= tag_config.MIN_TAG_FREQ, "Ainda ha tags nao protegidas abaixo da frequencia minima"
        assert (
            non_protected_final_freq["freq"].max() / game_count_after_category_filter
        ) <= tag_config.MAX_TAG_PERCENT, "Ainda ha tags nao protegidas acima da frequencia maxima"

    return {
        "tag_config": tag_config,
        "metadata_df": metadata_df,
        "tag_pairs_df": tag_pairs_df,
        "normalized_pairs_df": normalized_pairs_df,
        "without_unwanted_df": without_unwanted_df,
        "final_pairs_df": final_pairs_df,
        "raw_tag_freq": raw_tag_freq,
        "normalized_tag_freq": normalized_tag_freq,
        "final_tag_freq": final_tag_freq,
        "normalization_summary": normalization_summary,
        "too_frequent_df": too_frequent_df,
        "rare_tags_df": rare_tags_df,
        "filter_summary_df": filter_summary_df,
        "summary_df": summary_df,
        "comparison_df": comparison_df,
        "top_30_final_tags": top_30_final_tags,
        "preserved_structural_df": preserved_structural_df,
        "suspicious_survivors_df": suspicious_survivors_df,
        "stats": {
            "total_games": int(len(metadata_df)),
            "games_with_tags": int((metadata_df["tags"].str.len() > 0).sum()),
            "games_without_tags": int((metadata_df["tags"].str.len() == 0).sum()),
            "empty_descriptions": int(metadata_df["description"].eq("").sum()),
            "raw_pairs": int(len(tag_pairs_df)),
            "raw_distinct_tags": int(tag_pairs_df["tag"].nunique()),
            "normalized_distinct_tags": int(normalized_pairs_df["tag"].nunique()),
            "pairs_after_blacklist": int(len(without_unwanted_df)),
            "pairs_final": int(len(final_pairs_df)),
            "final_tag_count": int(final_tag_count),
            "games_with_final_tags": games_with_final_tags,
            "avg_tags_per_game": avg_tags_per_game,
            "raw_avg_tags_per_game": float(raw_tags_per_game.mean()),
            "raw_median_tags_per_game": float(raw_tags_per_game.median()),
            "normalization_changes": int(len(normalization_changes)),
            "protected_tag_count": int(len(tag_config.ALWAYS_KEEP_TAGS)),
            "blacklist_size": int(len(tag_config.TAGS_TO_REMOVE)),
            "too_frequent_count": int(len(too_frequent_tags)),
            "rare_count": int(len(rare_tags)),
        },
    }
