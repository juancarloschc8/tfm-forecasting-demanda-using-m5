"""
storage.py — Capa de almacenamiento del prototipo M5 Forecasting.

Define el modelo de datos relacional (SQLite) que soporta la persistencia de
los resultados del pipeline de pronóstico. El esquema permite registrar cada
ejecución (run), las series modeladas, las predicciones puntuales por día y
las métricas agregadas por modelo y serie.

Modelo de datos
---------------
    series      (series_id PK, name, category, store_id, pct_zeros)
    runs        (run_id PK, created_at, horizon, n_train_days, n_val_days)
    predictions (run_id FK, series_id FK, model, day_num, y_true, y_pred)
    metrics     (run_id FK, series_id FK, model, rmse, mae, mape)

Las claves foráneas garantizan la coherencia entre las tablas y permiten
reconstruir cualquier ejecución histórica del prototipo.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path("outputs/forecast.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    series_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT,
    store_id   TEXT,
    pct_zeros  REAL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    horizon       INTEGER NOT NULL,
    n_train_days  INTEGER NOT NULL,
    n_val_days    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id     INTEGER NOT NULL,
    series_id  TEXT NOT NULL,
    model      TEXT NOT NULL,
    day_num    INTEGER NOT NULL,
    y_true     REAL,
    y_pred     REAL,
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (series_id) REFERENCES series(series_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id     INTEGER NOT NULL,
    series_id  TEXT NOT NULL,
    model      TEXT NOT NULL,
    rmse       REAL,
    mae        REAL,
    mape       REAL,
    FOREIGN KEY (run_id)    REFERENCES runs(run_id),
    FOREIGN KEY (series_id) REFERENCES series(series_id)
);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Abre una conexión SQLite, creando el directorio y el esquema si faltan."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)
    return conn


def create_run(conn: sqlite3.Connection, horizon: int,
               n_train_days: int, n_val_days: int) -> int:
    """Registra una nueva ejecución y devuelve su run_id."""
    cur = conn.execute(
        "INSERT INTO runs (created_at, horizon, n_train_days, n_val_days) "
        "VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"),
         horizon, n_train_days, n_val_days),
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_series(conn: sqlite3.Connection, series_meta: list[dict]) -> None:
    """Inserta o actualiza la metadata de las series modeladas."""
    conn.executemany(
        "INSERT INTO series (series_id, name, category, store_id, pct_zeros) "
        "VALUES (:series_id, :name, :category, :store_id, :pct_zeros) "
        "ON CONFLICT(series_id) DO UPDATE SET "
        "name=excluded.name, category=excluded.category, "
        "store_id=excluded.store_id, pct_zeros=excluded.pct_zeros",
        series_meta,
    )
    conn.commit()


def save_predictions(conn: sqlite3.Connection, df_pred: pd.DataFrame) -> None:
    """Persiste predicciones. Columnas: run_id, series_id, model, day_num,
    y_true, y_pred."""
    df_pred.to_sql("predictions", conn, if_exists="append", index=False)


def save_metrics(conn: sqlite3.Connection, df_metrics: pd.DataFrame) -> None:
    """Persiste métricas. Columnas: run_id, series_id, model, rmse, mae, mape."""
    df_metrics.to_sql("metrics", conn, if_exists="append", index=False)


def latest_run_id(conn: sqlite3.Connection) -> int | None:
    """Devuelve el run_id más reciente, o None si no hay ejecuciones."""
    row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
    return row[0] if row and row[0] is not None else None


def load_metrics(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Carga la tabla de métricas de una ejecución como DataFrame."""
    return pd.read_sql_query(
        "SELECT m.series_id, s.name, s.category, m.model, m.rmse, m.mae, m.mape "
        "FROM metrics m JOIN series s ON m.series_id = s.series_id "
        "WHERE m.run_id = ? ORDER BY s.name, m.rmse",
        conn, params=(run_id,),
    )


def load_predictions(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Carga las predicciones de una ejecución como DataFrame."""
    return pd.read_sql_query(
        "SELECT p.series_id, s.name, p.model, p.day_num, p.y_true, p.y_pred "
        "FROM predictions p JOIN series s ON p.series_id = s.series_id "
        "WHERE p.run_id = ? ORDER BY s.name, p.model, p.day_num",
        conn, params=(run_id,),
    )
