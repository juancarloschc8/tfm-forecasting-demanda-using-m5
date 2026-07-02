"""
dashboard.py — Cuadro de mando interactivo del prototipo M5 Forecasting.

Aplicación Streamlit que permite explorar de forma interactiva los resultados
del pipeline de pronóstico: comparar modelos por serie, visualizar predicción
vs. valor real en el horizonte de validación y consultar la tabla de métricas.

Es el componente interactivo del prototipo del Entregable 4. Lee los datos
desde la base SQLite generada por src/forecast_pipeline.py.

Uso
---
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Permite importar el módulo de almacenamiento desde src/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
import storage  # noqa: E402

st.set_page_config(page_title="M5 Forecasting — Prototipo",
                   layout="wide")

# Ajustes CSS mínimos que el config.toml no cubre
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #d1d9e6;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)

MODEL_COLORS = {
    "Naïve": "#2563eb",
    "SARIMA": "#d97706",
    "LightGBM": "#16a34a",
    "Real": "#64748b",
}


@st.cache_data(show_spinner=False)
def load_run_data(run_id: int):
    """Carga métricas y predicciones de una ejecución desde SQLite."""
    conn = storage.connect()
    metrics = storage.load_metrics(conn, run_id)
    preds = storage.load_predictions(conn, run_id)
    conn.close()
    return metrics, preds


def get_latest_run() -> int | None:
    conn = storage.connect()
    rid = storage.latest_run_id(conn)
    conn.close()
    return rid


# Cabecera
st.title("Prototipo M5 Forecasting — Pronostico de demanda jerarquica")
st.caption(
    "Cuadro de mando interactivo del Trabajo Final de Máster. "
    "Datos generados por el pipeline reproducible `src/forecast_pipeline.py`."
)

run_id = get_latest_run()
if run_id is None:
    st.warning(
        "No hay resultados almacenados todavía. Ejecuta primero el pipeline:\n\n"
        "```\npython src/forecast_pipeline.py\n```"
    )
    st.stop()

metrics_df, preds_df = load_run_data(run_id)

# Barra lateral
st.sidebar.header("Controles")
st.sidebar.metric("Ejecucion activa", f"#{run_id}")

series_names = sorted(metrics_df["name"].unique())
sel_series = st.sidebar.selectbox("Serie representativa", series_names)

all_models = ["Naïve", "SARIMA", "LightGBM"]
present_models = [m for m in all_models if m in metrics_df["model"].unique()]
sel_models = st.sidebar.multiselect(
    "Modelos a comparar", present_models, default=present_models
)

# KPIs de la serie seleccionada
serie_metrics = metrics_df[metrics_df["name"] == sel_series].copy()
st.subheader(f"Resultados — {sel_series}")

if not serie_metrics.empty:
    best = serie_metrics.sort_values("rmse").iloc[0]
    cols = st.columns(len(present_models) + 1)
    cols[0].metric("Mejor modelo (RMSE)", best["model"], f"{best['rmse']:.2f}")
    for i, model in enumerate(present_models, start=1):
        row = serie_metrics[serie_metrics["model"] == model]
        if not row.empty:
            cols[i].metric(f"RMSE — {model}", f"{row.iloc[0]['rmse']:.2f}")

# Gráfico: predicción vs. real 
st.markdown("#### Predicción vs. valor real (validación de 28 días)")
serie_preds = preds_df[(preds_df["name"] == sel_series) &
                       (preds_df["model"].isin(sel_models))]

if serie_preds.empty:
    st.info("Selecciona al menos un modelo para visualizar las predicciones.")
else:
    fig = go.Figure()
    real = (serie_preds[serie_preds["model"] == serie_preds["model"].iloc[0]]
            .sort_values("day_num"))
    fig.add_trace(go.Scatter(
        x=real["day_num"], y=real["y_true"], name="Real",
        mode="lines+markers", line=dict(color=MODEL_COLORS["Real"], width=3),
    ))
    for model in sel_models:
        mp = serie_preds[serie_preds["model"] == model].sort_values("day_num")
        if not mp.empty:
            fig.add_trace(go.Scatter(
                x=mp["day_num"], y=mp["y_pred"], name=model, mode="lines+markers",
                line=dict(color=MODEL_COLORS.get(model), dash="dash"),
            ))
    fig.update_layout(
        xaxis_title="Dia (day_num)", yaxis_title="Ventas (unidades)",
        legend_title="Serie", height=450, hovermode="x unified",
        plot_bgcolor="#ffffff", paper_bgcolor="#f4f6f9",
        font=dict(color="#2d3748"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e0"),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e0"),
    )
    st.plotly_chart(fig, width="stretch")

# Comparativa de métricas
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Comparativa de RMSE por modelo")
    bar = serie_metrics[serie_metrics["model"].isin(sel_models)]
    if not bar.empty:
        fig_bar = px.bar(
            bar.sort_values("rmse"), x="model", y="rmse", color="model",
            color_discrete_map=MODEL_COLORS, text_auto=".2f",
        )
        fig_bar.update_layout(
            showlegend=False, xaxis_title="", yaxis_title="RMSE", height=380,
            plot_bgcolor="#ffffff", paper_bgcolor="#f4f6f9",
            font=dict(color="#2d3748"),
            xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e0"),
            yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e0"),
        )
        st.plotly_chart(fig_bar, width="stretch")

with right:
    st.markdown("#### Tabla de métricas")
    st.dataframe(
        serie_metrics[["model", "rmse", "mae", "mape"]]
        .rename(columns={"model": "Modelo", "rmse": "RMSE",
                         "mae": "MAE", "mape": "MAPE (%)"})
        .sort_values("RMSE").reset_index(drop=True),
        width="stretch", hide_index=True,
    )

# Resumen global de todas las series
st.markdown("---")
st.subheader("Resumen global — todas las series")
pivot = metrics_df.pivot_table(index="name", columns="model",
                               values="rmse").round(2)
st.dataframe(pivot, width="stretch")
st.caption(
    "RMSE por serie y modelo. Menor es mejor. "
    "Fuente: outputs/forecast.db (modelo de datos del prototipo)."
)
