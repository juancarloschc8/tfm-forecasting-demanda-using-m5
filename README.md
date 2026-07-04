# Prototipo de pronóstico de demanda jerárquica — Dataset M5 (Walmart)

Prototipo del **Trabajo Final de Máster** (Maestría en Análisis y Visualización
de Datos Masivos, UNIR). Implementa, de forma reproducible, un sistema de
pronóstico de demanda sobre el dataset
[M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
de Walmart, comparando modelos estadísticos y de *machine learning* y
exponiendo los resultados en un cuadro de mando interactivo.

> **Autores:**
> - Juan Carlos Chávez Cruz
> - José Carlos Castillo Villa
> - Marco Antonio Martínez Parada
> - Marcos José Nunez Estevez  
> 

**Asignatura:** Trabajo Final de Máster

**Repositorio:** https://github.com/juancarloschc8/tfm-forecasting-demanda-using-m5

---

## 1. Objetivo

Construir una **herramienta que recrea los resultados** del proyecto de análisis
de datos: a partir de los datos crudos del dataset M5, entrena varios modelos de
pronóstico, calcula las métricas de error en validación y persiste los resultados
en un modelo de datos relacional consultable desde un dashboard.

El prototipo sigue un **enfoque simplificado** (4 series representativas y 3
modelos núcleo) que permite ejecutar el flujo completo en pocos minutos, frente
a la evaluación exhaustiva de las 30.490 series documentada en los *notebooks*.

---

## 2. Estructura del repositorio

```
trabajo_final/
├── data/m5/                       # Datos crudos M5 (no versionados; ver §3)
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
├── src/
│   ├── forecast_pipeline.py       # Pipeline reproducible (entrena y persiste)
│   └── storage.py                 # Modelo de datos SQLite (capa de almacenamiento)
├── app/
│   └── dashboard.py               # Cuadro de mando interactivo (Streamlit)
├── outputs/                       # Resultados generados (CSV + SQLite)
├── models/                        # Modelos entrenados (.joblib)
├── EDA_M5_Forecasting.ipynb               # Análisis exploratorio
├── Modelo_M5_Forecasting.ipynb            # Modelado preliminar (E2)
├── Evaluacion_Resultados_M5_Forecasting.ipynb  # Evaluación completa (E3)
├── requirements.txt
└── README.md
```

---

## 3. Datos

El prototipo usa los ficheros del dataset M5 ubicados en `data/m5/`:

| Fichero | Descripción |
|---|---|
| `sales_train_validation.csv` | 30.490 series × 1.913 días de ventas diarias |
| `calendar.csv` | 1.969 días con eventos, SNAP y festivos |
| `sell_prices.csv` | Precios semanales por tienda y producto |

Por su tamaño (~450 MB) **no se versionan** (ver `.gitignore`). Descárgalos desde
la [competición M5 en Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
y colócalos en `data/m5/`.

---

## 4. Instalación

```powershell
# 1. Crear y activar entorno virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 5. Uso

### 5.1. Ejecutar el pipeline (recrea los resultados)

```powershell
python src/forecast_pipeline.py            # incluye SARIMA
python src/forecast_pipeline.py --no-sarima  # ejecución más rápida
```

El pipeline:
1. Carga los datos y selecciona 4 series representativas.
2. Entrena **Naïve estacional**, **SARIMA** y **LightGBM**.
3. Calcula **RMSE, MAE y MAPE** en validación de 28 días.
4. Persiste modelos (`models/*.joblib`), predicciones y métricas
   (`outputs/*.csv` y la base SQLite `outputs/forecast.db`).

### 5.2. Lanzar el cuadro de mando interactivo

```powershell
streamlit run app/dashboard.py
```

El dashboard permite seleccionar la serie, comparar modelos, visualizar
predicción vs. valor real y consultar la tabla de métricas, leyendo los datos
desde `outputs/forecast.db`.

---

## 6. Modelo de datos

La capa de almacenamiento (`src/storage.py`) define un esquema relacional SQLite
que **soporta la persistencia de los resultados** del prototipo:

| Tabla | Contenido |
|---|---|
| `series` | Metadata de cada serie (categoría, tienda, % de ceros) |
| `runs` | Registro de cada ejecución (fecha, horizonte, splits) |
| `predictions` | Predicción puntual por modelo, serie y día |
| `metrics` | RMSE, MAE y MAPE por modelo y serie |

Las claves foráneas garantizan la coherencia entre tablas y permiten reconstruir
cualquier ejecución histórica.

---

## 7. Modelos implementados

| Modelo | Tipo | Descripción |
|---|---|---|
| **Naïve estacional** | Estadístico (baseline) | Repite la última semana (s=7) |
| **SARIMA** | Estadístico | Orden seleccionado por `auto_arima` (m=7) |
| **LightGBM** | Machine learning | Gradient boosting con *features* de lag y calendario |

La evaluación completa de los *notebooks* incluye además SARIMAX, Prophet,
XGBoost, Random Forest y LightGBM global, con la métrica oficial **WRMSSE** y
**reconciliación jerárquica MinT**.

---

## 8. Reproducibilidad

Los resultados son reproducibles ejecutando el pipeline sobre los mismos datos.
Los *notebooks* del repositorio documentan el análisis exploratorio (EDA), el
modelado preliminar y la evaluación exhaustiva que respaldan la memoria del TFM.

---

## 9. Licencia

Proyecto académico desarrollado en el marco del TFM de la UNIR. El dataset M5 es
propiedad de Walmart y se distribuye bajo los términos de la competición de Kaggle.
