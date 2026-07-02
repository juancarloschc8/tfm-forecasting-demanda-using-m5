"""
forecast_pipeline.py — Pipeline reproducible del prototipo M5 Forecasting.

Recrea, de extremo a extremo, los resultados de pronóstico del TFM sobre un
subconjunto representativo de series del dataset M5 de Walmart. Es la
"herramienta que permite recrear los resultados" del Entregable 4.

Flujo
-----
1. Carga de datos (sales_train_validation.csv, calendar.csv).
2. Selección de 4 series representativas (FOODS, HOUSEHOLD, HOBBIES + agregado).
3. Entrenamiento de tres modelos: Naïve estacional, SARIMA y LightGBM.
4. Cálculo de métricas (RMSE, MAE, MAPE) en validación de 28 días.
5. Persistencia de modelos (joblib), predicciones y métricas (CSV + SQLite).

Uso
---
    python src/forecast_pipeline.py
    python src/forecast_pipeline.py --no-sarima   # omite SARIMA (más rápido)

Las salidas se escriben en outputs/ y models/. El dashboard (app/dashboard.py)
consume la base de datos outputs/forecast.db generada por este script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Permite ejecutar el script desde la raíz del repositorio.
sys.path.append(str(Path(__file__).resolve().parent))
import storage  # noqa: E402

DATA_DIR = Path("data/m5")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")
H = 28  # horizonte de predicción (días)

FEATURE_COLS = [
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
    "roll_mean_28", "roll_mean_7",
    "wday", "month", "snap_CA", "has_event",
]

LGB_PARAMS = {
    "objective": "regression_l2",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


# Carga de datos
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, int]:
    """Carga ventas y calendario; devuelve también day_cols y los splits."""
    if not (DATA_DIR / "sales_train_validation.csv").exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_DIR/'sales_train_validation.csv'}. "
            "Descarga el dataset M5 en data/m5/ antes de ejecutar."
        )
    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_days = len(day_cols)
    train_days = n_days - H
    return sales, calendar, day_cols, n_days, train_days


def select_representative_series(sales: pd.DataFrame,
                                 day_cols: list[str]) -> dict[str, str]:
    """Serie con menor % de ceros por categoría en CA_1 + agregado FOODS CA_1."""
    pct_ceros = (sales[day_cols] == 0).mean(axis=1)
    rep = {}
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        mask = (sales["cat_id"] == cat) & (sales["store_id"] == "CA_1")
        best = sales[mask].loc[pct_ceros[mask].idxmin()]
        rep[cat] = best["id"]
    rep["FOODS_CA1_AGG"] = "FOODS_CA1_AGG"
    return rep


def get_series(sales: pd.DataFrame, day_cols: list[str], series_id: str) -> np.ndarray:
    """Extrae una serie temporal como array float."""
    if series_id == "FOODS_CA1_AGG":
        y = sales[(sales["cat_id"] == "FOODS") &
                  (sales["store_id"] == "CA_1")][day_cols].sum().values
    else:
        y = sales[sales["id"] == series_id][day_cols].values.flatten()
    return y.astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE y MAPE (MAPE solo sobre posiciones con y_true > 0)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mask = y_true > 0
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.sum() > 0 else float("nan"))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# Feature engineering para LightGBM
def make_lgb_dataset(y_arr: np.ndarray, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Dataset tabular con lags desplazados >= 28 días (sin fuga de información)."""
    cal = calendar_df[["d", "wday", "month", "snap_CA", "event_type_1"]].copy()
    cal["day_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal.set_index("day_num")

    n = len(y_arr)
    records = []
    for t in range(56, n):
        dn = t + 1
        rec = {
            "day_num": dn,
            "target": y_arr[t],
            "lag_28": y_arr[t - 28],
            "lag_35": y_arr[t - 35],
            "lag_42": y_arr[t - 42],
            "lag_49": y_arr[t - 49],
            "lag_56": y_arr[t - 56],
            "roll_mean_28": float(np.mean(y_arr[t - 56: t - 28])),
            "roll_mean_7": float(np.mean(y_arr[t - 35: t - 28])),
        }
        if dn in cal.index:
            cr = cal.loc[dn]
            rec["wday"] = int(cr["wday"])
            rec["month"] = int(cr["month"])
            rec["snap_CA"] = int(cr["snap_CA"])
            rec["has_event"] = 0 if pd.isna(cr["event_type_1"]) else 1
        else:
            rec["wday"] = rec["month"] = rec["snap_CA"] = rec["has_event"] = 0
        records.append(rec)
    return pd.DataFrame(records)


# Modelos
def run_naive(y_train: np.ndarray) -> np.ndarray:
    """Naïve estacional s=7: repite la última semana de entrenamiento."""
    return np.tile(y_train[-7:], (H // 7) + 1)[:H]


def run_sarima(y_train: np.ndarray) -> np.ndarray:
    """SARIMA con orden seleccionado por auto_arima (m=7)."""
    from pmdarima import auto_arima
    model = auto_arima(
        y_train, start_p=1, start_q=1, max_p=2, max_q=2,
        m=7, seasonal=True, start_P=0, start_Q=0, max_P=1, max_Q=1,
        d=1, D=1, information_criterion="aic", stepwise=True,
        error_action="ignore", suppress_warnings=True,
    )
    return np.clip(model.predict(n_periods=H), 0, None)


def run_lightgbm(y: np.ndarray, calendar: pd.DataFrame,
                 train_days: int, n_days: int):
    """Entrena LightGBM por serie y devuelve (modelo, predicciones, y_val)."""
    df = make_lgb_dataset(y, calendar)
    df_train = df[df["day_num"] <= train_days].dropna(subset=FEATURE_COLS)
    df_val = df[(df["day_num"] > train_days) &
                (df["day_num"] <= n_days)].dropna(subset=FEATURE_COLS)
    dtrain = lgb.Dataset(df_train[FEATURE_COLS], label=df_train["target"])
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200,
                      callbacks=[lgb.log_evaluation(0)])
    y_pred = np.clip(model.predict(df_val[FEATURE_COLS]), 0, None)
    return model, y_pred, df_val["target"].values


# Orquestación
def main(run_sarima_flag: bool = True) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    sales, calendar, day_cols, n_days, train_days = load_data()
    rep = select_representative_series(sales, day_cols)
    pct_ceros = (sales[day_cols] == 0).mean(axis=1)
    print(f"Series: {len(sales):,} | train: d_1–d_{train_days} | val: {H} días\n")

    conn = storage.connect()
    run_id = storage.create_run(conn, H, train_days, H)

    # Metadata de series
    series_meta = []
    for name, sid in rep.items():
        if sid == "FOODS_CA1_AGG":
            meta = {"series_id": sid, "name": name, "category": "FOODS",
                    "store_id": "CA_1", "pct_zeros": None}
        else:
            row = sales[sales["id"] == sid].iloc[0]
            meta = {"series_id": sid, "name": name, "category": row["cat_id"],
                    "store_id": row["store_id"],
                    "pct_zeros": float(pct_ceros[row.name] * 100)}
        series_meta.append(meta)
    storage.upsert_series(conn, series_meta)

    pred_rows, metric_rows = [], []
    day_index = list(range(train_days + 1, n_days + 1))  # días de validación

    for name, sid in rep.items():
        print(f"── {name} ({sid}) ─────────────────────────────")
        y = get_series(sales, day_cols, sid)
        y_train, y_val = y[:train_days], y[train_days:]

        # 1. Naïve estacional
        models_preds = {"Naïve": run_naive(y_train)}

        # 2. SARIMA (opcional)
        if run_sarima_flag:
            print("   Ajustando SARIMA...", flush=True)
            models_preds["SARIMA"] = run_sarima(y_train)

        # 3. LightGBM
        print("   Entrenando LightGBM...", flush=True)
        lgb_model, lgb_pred, lgb_yval = run_lightgbm(y, calendar, train_days, n_days)
        models_preds["LightGBM"] = lgb_pred
        joblib.dump(lgb_model, MODELS_DIR / f"lgb_{name}.joblib")

        # Métricas y predicciones por modelo
        for model_name, y_pred in models_preds.items():
            y_ref = lgb_yval if model_name == "LightGBM" else y_val
            m = compute_metrics(y_ref, y_pred)
            metric_rows.append({"run_id": run_id, "series_id": sid,
                                "model": model_name, "rmse": m["RMSE"],
                                "mae": m["MAE"], "mape": m["MAPE"]})
            for dn, yt, yp in zip(day_index, y_ref, y_pred):
                pred_rows.append({"run_id": run_id, "series_id": sid,
                                  "model": model_name, "day_num": int(dn),
                                  "y_true": float(yt), "y_pred": float(yp)})
            print(f"      {model_name:<10} RMSE={m['RMSE']:7.3f} "
                  f"MAE={m['MAE']:7.3f} MAPE={m['MAPE']:5.1f}%")

    df_pred = pd.DataFrame(pred_rows)
    df_metrics = pd.DataFrame(metric_rows)

    # Persistencia: SQLite + CSV
    storage.save_predictions(conn, df_pred)
    storage.save_metrics(conn, df_metrics)
    conn.close()
    df_pred.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    df_metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    print(f"\nEjecución #{run_id} completada.")
    print(f"  Base de datos : {storage.DB_PATH}")
    print(f"  Predicciones  : {OUTPUTS_DIR/'predictions.csv'}")
    print(f"  Métricas      : {OUTPUTS_DIR/'metrics.csv'}")
    print(f"  Modelos       : {MODELS_DIR}/lgb_*.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline M5 Forecasting")
    parser.add_argument("--no-sarima", action="store_true",
                        help="Omite SARIMA para una ejecución más rápida.")
    args = parser.parse_args()
    main(run_sarima_flag=not args.no_sarima)
