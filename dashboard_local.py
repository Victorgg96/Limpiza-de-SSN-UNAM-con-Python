from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="SSN Medallion Dashboard", layout="wide")
st.title("🌍 SSN-UNAM - Dashboard de Datos Limpios (Gold)")
st.markdown("**Pipeline Medallion**: Bronze → Silver → Gold | Datos del Servicio Sismológico Nacional (UNAM)")

root = Path(__file__).resolve().parent
gold_dir = root / "data" / "gold"
logs_dir = root / "data" / "logs"


def load_gold_table(table_name: str) -> pd.DataFrame:
    parquet_path = gold_dir / f"{table_name}.parquet"
    csv_path = gold_dir / f"{table_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No se encontro {table_name} en Parquet ni CSV")


def read_latest_run_metadata() -> dict:
    if not logs_dir.exists():
        return {}
    candidates = sorted(logs_dir.glob("run_metadata_*.json"), reverse=True)
    if not candidates:
        return {}
    with candidates[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_from_app(script_path: Path) -> tuple[int, str, str]:
    """Ejecuta `prueba.py` con el mismo intérprete y retorna (code, stdout, stderr)."""
    proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


@st.cache_data
def try_load_gold_data() -> tuple[dict, list]:
    """Intenta cargar todas las tablas Gold. Retorna (loaded_dict, missing_list)."""
    required_files = {
        "regional": "gold_sismicidad_regional",
        "temporal": "gold_patrones_temporales",
        "significativos": "gold_sismos_significativos",
        "evolucion": "gold_evolucion_historica",
    }
    loaded = {}
    missing = []
    for name, base_name in required_files.items():
        try:
            loaded[name] = load_gold_table(base_name)
        except FileNotFoundError:
            missing.append(name)
    return loaded, missing


# Intenta cargar datos
gold_data, missing_files = try_load_gold_data()

# Sidebar: Pipeline control
st.sidebar.header("🚀 Pipeline")
if st.sidebar.button("▶ Ejecutar pipeline (prueba.py)"):
    script_file = Path(__file__).resolve().parent / "prueba.py"
    with st.spinner("⏳ Ejecutando pipeline... esto puede tardar varios segundos"):
        code, out, err = run_pipeline_from_app(script_file)
    if code == 0:
        st.sidebar.success("✅ Pipeline finalizado correctamente")
        st.cache_data.clear()
        gold_data, missing_files = try_load_gold_data()
    else:
        st.sidebar.error(f"❌ Pipeline finalizó con código {code}")
    with st.expander("📋 Salida del pipeline (stdout)"):
        st.code(out or "(sin salida)", language="text")
    if err:
        with st.expander("🔴 Errores (stderr)"):
            st.code(err, language="text")
    rerun_app()

if missing_files:
    st.sidebar.warning(f"⚠️ Faltan {len(missing_files)} tabla(s) Gold")
else:
    st.sidebar.success("✅ Datos cargados correctamente")

# Estado inicial si no hay datos
if not gold_data or all(df.empty if isinstance(df, pd.DataFrame) else not df for df in gold_data.values()):
    st.warning("📊 No hay datos cargados aún.")
    st.info("Haz clic en el botón '▶ Ejecutar pipeline' en la barra lateral para procesar los datos del SSN.")
    st.stop()

# Extraer datos con seguridad
regional = gold_data.get("regional", pd.DataFrame())
temporal = gold_data.get("temporal", pd.DataFrame())
significativos = gold_data.get("significativos", pd.DataFrame())
evolucion = gold_data.get("evolucion", pd.DataFrame())

# Convertir tipos numéricos
for col in ["anio", "total_sismos", "magnitud_promedio", "magnitud_maxima"]:
    if col in regional.columns:
        regional[col] = pd.to_numeric(regional[col], errors="coerce")

if "magnitud" in significativos.columns:
    significativos["magnitud"] = pd.to_numeric(significativos["magnitud"], errors="coerce")

# Mostrar metadatos del último run
metadata = read_latest_run_metadata()
if metadata:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Run ID",
        metadata.get("run_id", "N/D")[-10:],
        help="Últimos 10 caracteres del ID de la corrida",
    )
    col2.metric(
        "Filas Bronze",
        f"{metadata.get('bronze', {}).get('raw_rows_loaded', 'N/D'):,}",
    )
    col3.metric(
        "Filas Silver",
        f"{metadata.get('silver', {}).get('rows_final', 'N/D'):,}",
    )
    col4.metric(
        "Calidad (Silver/Bronze)",
        f"{round(metadata.get('silver', {}).get('rows_final', 0) / max(metadata.get('bronze', {}).get('raw_rows_loaded', 1), 1) * 100, 1)}%",
    )

st.divider()

# Sidebar: Filtros
st.sidebar.header("📈 Filtros")

# Filtro de años
available_years = sorted(regional["anio"].dropna().astype(int).unique().tolist())
year_option = st.sidebar.radio("Mostrar años:", ("Todos", "Rango", "Año específico"), horizontal=False)

if year_option == "Todos":
    selected_years = available_years
    year_label = "Todos los años"
elif year_option == "Rango":
    year_range = st.sidebar.slider(
        "Rango de años",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(min(available_years), max(available_years)),
    )
    selected_years = [y for y in available_years if year_range[0] <= y <= year_range[1]]
    year_label = f"{year_range[0]}-{year_range[1]}"
else:  # Año específico
    selected_year_single = st.sidebar.selectbox(
        "Selecciona un año",
        available_years,
        index=len(available_years) - 1,
    )
    selected_years = [selected_year_single]
    year_label = str(selected_year_single)

# Filtro de estados
available_states = sorted(regional["estado"].dropna().astype(str).unique().tolist())
selected_states = st.sidebar.multiselect("Estados (vacío = todos)", available_states)

# Filtro de magnitud
min_mag = float(significativos["magnitud"].dropna().min()) if "magnitud" in significativos.columns else 0.0
max_mag = float(significativos["magnitud"].dropna().max()) if "magnitud" in significativos.columns else 10.0
mag_range = st.sidebar.slider(
    "Rango de magnitud",
    min_mag,
    max_mag,
    (max(5.0, min_mag), max_mag),
)

st.divider()

# Resumen
st.subheader("📊 Resumen")
col1, col2, col3 = st.columns(3)
col1.metric("Filas regionales", f"{len(regional):,}")
col2.metric("Sismos significativos > 5.0", f"{len(significativos):,}")
col3.metric("Filas evolucion historica", f"{len(evolucion):,}")

# Sismicidad regional
st.subheader(f"🗺️ Sismicidad regional ({year_label})")
regional_filtered = regional[regional["anio"].isin(selected_years)].sort_values("total_sismos", ascending=False).head(20)
if selected_states:
    regional_filtered = regional_filtered[regional_filtered["estado"].isin(selected_states)]

if not regional_filtered.empty:
    fig_regional = px.bar(
        regional_filtered,
        x="estado",
        y="total_sismos",
        color="magnitud_promedio",
        title=f"Top estados por sismos ({year_label})",
        labels={"total_sismos": "Total sismos", "magnitud_promedio": "Magnitud promedio", "estado": "Estado"},
    )
    st.plotly_chart(fig_regional, use_container_width=True)
else:
    st.info("Sin datos para los filtros seleccionados.")

# Patrones temporales
st.subheader("⏰ Patrones temporales (por hora del día)")
if not temporal.empty:
    temporal_hora = temporal.groupby("hora", as_index=False)["total_sismos"].sum().sort_values("hora")
    fig_hora = px.line(
        temporal_hora,
        x="hora",
        y="total_sismos",
        markers=True,
        title="Frecuencia total por hora del día",
        labels={"hora": "Hora (local)", "total_sismos": "Total sismos"},
    )
    st.plotly_chart(fig_hora, use_container_width=True)
else:
    st.info("Sin datos de patrones temporales.")

# Evolución histórica
st.subheader("📈 Evolución histórica (anual)")
if not evolucion.empty:
    evo_anual = evolucion[evolucion["nivel_tiempo"] == "anual"].sort_values("anio")
    evo_filtered = evo_anual[evo_anual["anio"].isin(selected_years)]
    if not evo_filtered.empty:
        fig_evo = px.line(
            evo_filtered,
            x="anio",
            y="total_sismos",
            markers=True,
            title=f"Conteo anual de sismos ({year_label})",
            labels={"anio": "Año", "total_sismos": "Total sismos"},
        )
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("Sin datos para el rango de años seleccionado.")
else:
    st.info("Sin datos de evolución histórica.")

# Top sismos significativos
st.subheader("🔥 Top sismos significativos (Magnitud > 5.0)")
significativos_view = significativos[
    (significativos["magnitud"] >= mag_range[0])
    & (significativos["magnitud"] <= mag_range[1])
]
if selected_states:
    significativos_view = significativos_view[significativos_view["estado"].isin(selected_states)]

view_cols = [
    "fecha",
    "Hora",
    "magnitud",
    "estado",
    "profundidad",
    "distancia_km",
    "clasificacion_magnitud",
    "Referencia de localizacion",
]
available_cols = [col for col in view_cols if col in significativos_view.columns]
if available_cols:
    st.dataframe(
        significativos_view[available_cols].sort_values("magnitud", ascending=False).head(200),
        use_container_width=True,
    )
else:
    st.info("Sin columnas disponibles para mostrar.")

# Reporte de calidad / Metadatos
st.divider()
st.subheader("📋 Reporte de calidad / Metadatos del último run")
if metadata:
    with st.expander("Ver JSON completo de metadatos", expanded=False):
        st.json(metadata)
    gold_tables = metadata.get("gold", {}).get("tables", {})
    if gold_tables:
        df_tables = pd.DataFrame(list(gold_tables.items()), columns=["Tabla", "Filas"]).sort_values("Filas", ascending=False)
        fig_tables = px.bar(
            df_tables,
            x="Tabla",
            y="Filas",
            title="Filas por tabla Gold",
            labels={"Tabla": "Tabla Gold", "Filas": "Cantidad de filas"},
        )
        st.plotly_chart(fig_tables, use_container_width=True)
else:
    st.info("No hay metadatos de corrida registrados aún. Ejecuta el pipeline primero.")
