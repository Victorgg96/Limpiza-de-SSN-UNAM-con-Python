from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="SSN Medallion Dashboard", layout="wide")
st.title("SSN-UNAM - Dashboard de Datos Limpios (Gold)")

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

required_files = {
    "regional": "gold_sismicidad_regional",
    "temporal": "gold_patrones_temporales",
    "significativos": "gold_sismos_significativos",
    "evolucion": "gold_evolucion_historica",
}

missing = []
for name, base_name in required_files.items():
    if not (gold_dir / f"{base_name}.parquet").exists() and not (gold_dir / f"{base_name}.csv").exists():
        missing.append(name)

if missing:
    st.error(
        "Faltan archivos Gold. Ejecuta primero: python prueba.py\\n"
        f"No encontrados: {', '.join(missing)}"
    )
    st.stop()

regional = load_gold_table(required_files["regional"])
temporal = load_gold_table(required_files["temporal"])
significativos = load_gold_table(required_files["significativos"])
evolucion = load_gold_table(required_files["evolucion"])

for col in ["anio", "total_sismos", "magnitud_promedio", "magnitud_maxima"]:
    if col in regional.columns:
        regional[col] = pd.to_numeric(regional[col], errors="coerce")

if "magnitud" in significativos.columns:
    significativos["magnitud"] = pd.to_numeric(significativos["magnitud"], errors="coerce")

metadata = read_latest_run_metadata()
if metadata:
    st.caption(
        f"Ultimo run_id: {metadata.get('run_id', 'N/D')} | "
        f"Silver rows: {metadata.get('silver', {}).get('rows_final', 'N/D')}"
    )
# Sidebar: Pipeline control
st.sidebar.header("Pipeline")
if st.sidebar.button("Ejecutar pipeline (prueba.py)"):
    script_file = Path(__file__).resolve().parent / "prueba.py"
    with st.spinner("Ejecutando pipeline... esto puede tardar varios segundos"):
        code, out, err = run_pipeline_from_app(script_file)
    if code == 0:
        st.sidebar.success("Pipeline finalizado correctamente")
    else:
        st.sidebar.error(f"Pipeline finalizó con código {code}")
    with st.expander("Salida del pipeline (stdout)"):
        st.text(out or "(sin salida)")
    if err:
        with st.expander("Errores (stderr)"):
            st.text(err)
    time.sleep(0.5)
    st.experimental_rerun()

st.subheader("Resumen")
col1, col2, col3 = st.columns(3)
col1.metric("Filas regionales", f"{len(regional):,}")
col2.metric("Sismos significativos > 5.0", f"{len(significativos):,}")
col3.metric("Filas evolucion historica", f"{len(evolucion):,}")

st.sidebar.header("Filtros")
available_years = sorted(regional["anio"].dropna().astype(int).unique().tolist())
selected_year = st.sidebar.selectbox("Anio", available_years, index=len(available_years) - 1)

available_states = sorted(regional["estado"].dropna().astype(str).unique().tolist())
selected_states = st.sidebar.multiselect("Estados", available_states)

min_mag = float(significativos["magnitud"].dropna().min()) if "magnitud" in significativos.columns else 0.0
max_mag = float(significativos["magnitud"].dropna().max()) if "magnitud" in significativos.columns else 10.0
mag_range = st.sidebar.slider("Rango de magnitud", min_mag, max_mag, (max(5.0, min_mag), max_mag))

st.divider()

st.subheader("Sismicidad regional")
regional_year = regional[regional["anio"] == selected_year].sort_values("total_sismos", ascending=False).head(20)
if selected_states:
    regional_year = regional_year[regional_year["estado"].isin(selected_states)]

fig_regional = px.bar(
    regional_year,
    x="estado",
    y="total_sismos",
    color="magnitud_promedio",
    title=f"Top estados por sismos en {selected_year}",
)
st.plotly_chart(fig_regional, use_container_width=True)

st.subheader("Patrones temporales")
fig_hora = px.line(
    temporal.groupby("hora", as_index=False)["total_sismos"].sum().sort_values("hora"),
    x="hora",
    y="total_sismos",
    markers=True,
    title="Frecuencia total por hora del dia",
)
st.plotly_chart(fig_hora, use_container_width=True)

st.subheader("Evolucion historica (anual)")
evo_anual = evolucion[evolucion["nivel_tiempo"] == "anual"].sort_values("anio")
fig_evo = px.line(
    evo_anual,
    x="anio",
    y="total_sismos",
    title="Conteo anual de sismos",
)
st.plotly_chart(fig_evo, use_container_width=True)

st.subheader("Top sismos significativos")
significativos_view = significativos[(significativos["magnitud"] >= mag_range[0]) & (significativos["magnitud"] <= mag_range[1])]
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
st.dataframe(
    significativos_view[view_cols].sort_values("magnitud", ascending=False).head(200),
    use_container_width=True,
)

st.subheader("Reporte de calidad / Metadatos del último run")
if metadata:
    with st.expander("Ver metadatos de la última corrida", expanded=False):
        st.json(metadata)
    gold_tables = metadata.get("gold", {}).get("tables", {})
    if gold_tables:
        df_tables = pd.DataFrame(list(gold_tables.items()), columns=["tabla", "filas"]).sort_values("filas", ascending=False)
        fig_tables = px.bar(df_tables, x="tabla", y="filas", title="Filas por tabla Gold")
        st.plotly_chart(fig_tables, use_container_width=True)
else:
    st.write("No hay metadatos de corrida registrados aún.")
