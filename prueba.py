from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class PipelineConfig:
    source_csv: Path
    data_dir: Path


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "bronze_raw": base_dir / "bronze" / "raw",
        "bronze_partitioned": base_dir / "bronze" / "partitioned",
        "silver": base_dir / "silver",
        "gold": base_dir / "gold",
        "logs": base_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def file_sha256(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def read_source_csv(source_csv: Path) -> tuple[pd.DataFrame, dict]:
    metadata_lines: list[str] = []
    with source_csv.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(4):
            metadata_lines.append(f.readline().strip())

    df = pd.read_csv(source_csv, skiprows=4)

    total_match = re.search(r"Total:\s*(\d+)\s*eventos", " ".join(metadata_lines), flags=re.I)
    total_reported = int(total_match.group(1)) if total_match else None

    source_metadata = {
        "header_lines": metadata_lines,
        "total_reported_events": total_reported,
    }
    return df, source_metadata


def bronze_layer(config: PipelineConfig, dirs: dict[str, Path], run_id: str) -> tuple[pd.DataFrame, dict]:
    source_hash = file_sha256(config.source_csv)
    raw_copy_path = dirs["bronze_raw"] / f"ssn_raw_{run_id}.csv"
    shutil.copy2(config.source_csv, raw_copy_path)

    df_raw, source_metadata = read_source_csv(config.source_csv)
    df_raw.to_csv(dirs["bronze_raw"] / f"ssn_raw_table_{run_id}.csv", index=False, encoding="utf-8")

    temp = df_raw.copy()
    temp["Fecha"] = pd.to_datetime(temp["Fecha"], errors="coerce")
    temp["year"] = temp["Fecha"].dt.year

    partition_rows = 0
    for year, group in temp.dropna(subset=["year"]).groupby("year"):
        year_dir = dirs["bronze_partitioned"] / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year"]).to_csv(year_dir / f"ssn_{int(year)}.csv", index=False, encoding="utf-8")
        partition_rows += len(group)

    bronze_meta = {
        "layer": "bronze",
        "run_id": run_id,
        "source_file": str(config.source_csv.name),
        "source_sha256": source_hash,
        "ingest_timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_rows_loaded": int(len(df_raw)),
        "rows_partitioned_by_year": int(partition_rows),
        "source_metadata": source_metadata,
    }

    return df_raw, bronze_meta


def extract_state(reference_text: str) -> str | None:
    if not isinstance(reference_text, str):
        return None
    match = re.search(r",\s*([A-Z]{2,4})\s*$", reference_text.strip())
    if match:
        return match.group(1)
    return None


def extract_distance_km(reference_text: str) -> float | None:
    if not isinstance(reference_text, str):
        return None
    match = re.search(r"^\s*(\d+(?:\.\d+)?)\s*km\b", reference_text.strip(), flags=re.I)
    if match:
        return float(match.group(1))
    return None


def classify_magnitude(mag: float) -> str:
    if mag >= 8.0:
        return "great"
    if mag >= 7.0:
        return "major"
    if mag >= 6.0:
        return "strong"
    if mag >= 5.0:
        return "moderate"
    return "light"


def silver_layer(df_bronze: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, dict]:
    required_columns = [
        "Fecha",
        "Hora",
        "Magnitud",
        "Latitud",
        "Longitud",
        "Profundidad",
        "Referencia de localizacion",
        "Fecha UTC",
        "Hora UTC",
        "Estatus",
    ]
    missing_columns = [col for col in required_columns if col not in df_bronze.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df_bronze.copy()
    initial_rows = len(df)

    df = df.drop_duplicates()
    after_dedup_rows = len(df)

    df["fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["fecha_utc"] = pd.to_datetime(df["Fecha UTC"], errors="coerce")
    df["magnitud"] = pd.to_numeric(df["Magnitud"], errors="coerce")
    df["latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    df["profundidad"] = pd.to_numeric(df["Profundidad"], errors="coerce")

    df["hora_local"] = pd.to_datetime(df["Hora"], format="%H:%M:%S", errors="coerce").dt.time
    df["hora_utc"] = pd.to_datetime(df["Hora UTC"], format="%H:%M:%S", errors="coerce").dt.time

    df["estado"] = df["Referencia de localizacion"].apply(extract_state)
    df["distancia_km"] = df["Referencia de localizacion"].apply(extract_distance_km)

    null_mag_rows = int(df["magnitud"].isna().sum())
    df = df.dropna(subset=["magnitud", "latitud", "longitud", "profundidad", "fecha"])

    geo_valid = (
        df["latitud"].between(-90, 90)
        & df["longitud"].between(-180, 180)
        & df["profundidad"].between(0, 700)
    )
    invalid_geo_rows = int((~geo_valid).sum())
    df = df[geo_valid].copy()

    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["hora"] = pd.to_datetime(df["Hora"], format="%H:%M:%S", errors="coerce").dt.hour
    df["estacion"] = df["mes"].map(
        {
            12: "invierno",
            1: "invierno",
            2: "invierno",
            3: "primavera",
            4: "primavera",
            5: "primavera",
            6: "verano",
            7: "verano",
            8: "verano",
            9: "otono",
            10: "otono",
            11: "otono",
        }
    )

    df["clasificacion_magnitud"] = df["magnitud"].apply(classify_magnitude)

    silver_meta = {
        "layer": "silver",
        "run_id": run_id,
        "rows_initial": int(initial_rows),
        "rows_after_dedup": int(after_dedup_rows),
        "null_magnitude_rows_removed": null_mag_rows,
        "invalid_geo_rows_removed": invalid_geo_rows,
        "rows_final": int(len(df)),
    }

    return df, silver_meta


def gold_layer(df_silver: pd.DataFrame, run_id: str) -> tuple[dict[str, pd.DataFrame], dict]:
    gold_sismicidad_regional = (
        df_silver.groupby(["estado", "anio"], dropna=False)
        .agg(
            total_sismos=("magnitud", "size"),
            magnitud_promedio=("magnitud", "mean"),
            magnitud_maxima=("magnitud", "max"),
        )
        .reset_index()
        .sort_values(["anio", "total_sismos"], ascending=[True, False])
    )

    gold_patrones_temporales = (
        df_silver.groupby(["hora", "dia_semana", "mes", "estacion"], dropna=False)
        .agg(
            total_sismos=("magnitud", "size"),
            magnitud_promedio=("magnitud", "mean"),
        )
        .reset_index()
        .sort_values("total_sismos", ascending=False)
    )

    gold_sismos_significativos = (
        df_silver[df_silver["magnitud"] > 5.0][
            [
                "fecha",
                "Hora",
                "magnitud",
                "latitud",
                "longitud",
                "profundidad",
                "estado",
                "distancia_km",
                "clasificacion_magnitud",
                "Referencia de localizacion",
            ]
        ]
        .sort_values("magnitud", ascending=False)
        .reset_index(drop=True)
    )

    yearly = (
        df_silver.groupby("anio")
        .agg(total_sismos=("magnitud", "size"), magnitud_maxima=("magnitud", "max"))
        .reset_index()
    )
    monthly = (
        df_silver.groupby(["anio", "mes"])
        .agg(total_sismos=("magnitud", "size"), magnitud_maxima=("magnitud", "max"))
        .reset_index()
    )
    yearly["nivel_tiempo"] = "anual"
    yearly["subperiodo"] = 0
    monthly["nivel_tiempo"] = "mensual"
    monthly = monthly.rename(columns={"mes": "subperiodo"})

    gold_evolucion_historica = pd.concat(
        [
            yearly[["nivel_tiempo", "anio", "subperiodo", "total_sismos", "magnitud_maxima"]],
            monthly[["nivel_tiempo", "anio", "subperiodo", "total_sismos", "magnitud_maxima"]],
        ],
        ignore_index=True,
    ).sort_values(["nivel_tiempo", "anio", "subperiodo"])

    gold_tables = {
        "gold_sismicidad_regional": gold_sismicidad_regional,
        "gold_patrones_temporales": gold_patrones_temporales,
        "gold_sismos_significativos": gold_sismos_significativos,
        "gold_evolucion_historica": gold_evolucion_historica,
    }

    gold_meta = {
        "layer": "gold",
        "run_id": run_id,
        "tables": {name: int(len(df)) for name, df in gold_tables.items()},
    }
    return gold_tables, gold_meta


def write_dual_format(df: pd.DataFrame, target_base: Path) -> None:
    df.to_csv(target_base.with_suffix(".csv"), index=False, encoding="utf-8")
    parquet_df = df.copy()
    object_columns = parquet_df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        parquet_df[column] = parquet_df[column].astype("string")
    parquet_df.to_parquet(target_base.with_suffix(".parquet"), index=False)


def save_metadata(dirs: dict[str, Path], run_id: str, payload: dict) -> None:
    metadata_path = dirs["logs"] / f"run_metadata_{run_id}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def validate_outputs(data_dir: Path) -> None:
    required_outputs = [
        data_dir / "silver" / "silver_sismos_limpios.csv",
        data_dir / "silver" / "silver_sismos_limpios.parquet",
        data_dir / "gold" / "gold_sismicidad_regional.csv",
        data_dir / "gold" / "gold_sismicidad_regional.parquet",
        data_dir / "gold" / "gold_patrones_temporales.csv",
        data_dir / "gold" / "gold_patrones_temporales.parquet",
        data_dir / "gold" / "gold_sismos_significativos.csv",
        data_dir / "gold" / "gold_sismos_significativos.parquet",
        data_dir / "gold" / "gold_evolucion_historica.csv",
        data_dir / "gold" / "gold_evolucion_historica.parquet",
    ]

    missing = [str(path) for path in required_outputs if not path.exists()]
    empty = [str(path) for path in required_outputs if path.exists() and path.stat().st_size == 0]

    if missing or empty:
        detail = {
            "missing": missing,
            "empty": empty,
        }
        raise RuntimeError(f"Validacion de outputs fallida: {json.dumps(detail, ensure_ascii=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Medallion para SSN")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("SSNMX_catalogo_19000101_20260407.csv"),
        help="Ruta al CSV de entrada",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Carpeta base de salida para Bronze/Silver/Gold",
    )
    return parser.parse_args()


def run_pipeline(config: PipelineConfig) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = ensure_dirs(config.data_dir)

    df_bronze, bronze_meta = bronze_layer(config, dirs, run_id)
    df_silver, silver_meta = silver_layer(df_bronze, run_id)

    write_dual_format(df_silver, dirs["silver"] / "silver_sismos_limpios")

    gold_tables, gold_meta = gold_layer(df_silver, run_id)
    for table_name, df in gold_tables.items():
        write_dual_format(df, dirs["gold"] / table_name)

    metadata_payload = {
        "run_id": run_id,
        "source": str(config.source_csv),
        "outputs": {
            "bronze": str(dirs["bronze_raw"]),
            "silver": str(dirs["silver"]),
            "gold": str(dirs["gold"]),
        },
        "bronze": bronze_meta,
        "silver": silver_meta,
        "gold": gold_meta,
    }
    save_metadata(dirs, run_id, metadata_payload)

    print("Pipeline Medallion completado")
    print(f"Run ID: {run_id}")
    print(f"Silver rows: {silver_meta['rows_final']}")
    print("Gold tables:")
    for name, count in gold_meta["tables"].items():
        print(f"  - {name}: {count}")

    validate_outputs(config.data_dir)
    print("Validacion de archivos generados: OK")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    args = parse_args()
    source_csv = (root / args.input).resolve() if not args.input.is_absolute() else args.input
    data_dir = (root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir

    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    cfg = PipelineConfig(source_csv=source_csv, data_dir=data_dir)
    run_pipeline(cfg)
