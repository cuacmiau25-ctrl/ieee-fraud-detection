# Detección de Fraude — IEEE-CIS Fraud Detection

Proyecto de clasificación binaria para detectar transacciones fraudulentas usando el dataset
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) de Kaggle. La variable objetivo es
`isFraud` y el dataset está fuertemente desbalanceado: solo ~3.5% de las transacciones son fraude.

El pipeline completo —desde la exploración hasta el modelado— vive en `notebooks/`, se apoya en `src/` para
utilidades compartidas, y produce artefactos intermedios en `data/processed/` y el modelo final en `outputs/`.

## Estructura de carpetas

```
fraud/
├── data/
│   ├── raw/            # CSVs originales de Kaggle (no versionado, ver "Descarga de datos")
│   └── processed/      # Datasets y encoders generados por los notebooks (no versionado)
│       └── artifacts/  # Imputadores, encoders, tablas de agregación (joblib) y feature_columns.json
├── notebooks/
│   ├── 01_exploracion.ipynb          # Carga, dimensiones, missing, desbalance, EDA de variables clave
│   ├── 02_limpieza.ipynb             # Split train/test, imputación y codificación (sin fugas)
│   ├── 03_feature_engineering.ipynb  # Features nuevas (hora/día, agregaciones, email, reducción de V-cols)
│   ├── 04_balanceo.ipynb             # Comparación SMOTE vs. class_weight y decisión final
│   └── 05_modelo.ipynb               # LogisticRegression y RandomForest, CV, overfitting, interpretabilidad
├── outputs/             # Modelos entrenados (.joblib) y métricas finales (no versionado el .joblib)
├── src/
│   └── utils.py         # reduce_mem_usage() y utilidades compartidas
├── requirements.txt
└── README.md
```

## Descarga de datos

Los datos no están incluidos en el repositorio (ver `.gitignore`). Para obtenerlos necesitas una cuenta de Kaggle
con la API configurada (`~/.kaggle/kaggle.json`) y haber aceptado las reglas de la competencia:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Luego descomprime el `.zip` dentro de `data/raw/` de modo que queden estos 5 archivos directamente en esa carpeta:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
data/raw/test_transaction.csv
data/raw/test_identity.csv
data/raw/sample_submission.csv
```

> Nota: `test_transaction.csv`/`test_identity.csv` de Kaggle **no tienen la columna `isFraud`** (son el set de
> submission de la competencia). Todo el pipeline de evaluación de este proyecto usa en cambio un split
> estratificado 80/20 interno, generado a partir de `train_transaction.csv` + `train_identity.csv`, para poder medir
> métricas supervisadas de forma honesta.

## Instalación de dependencias

El proyecto usa un entorno virtual en `.venv/`. Para recrearlo:

```bash
python -m venv .venv
.venv/Scripts/activate      # En Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

## Cómo ejecutar el proyecto

Los notebooks deben correrse **en orden** (01 → 05): cada uno lee artefactos que el anterior dejó en
`data/processed/`, y no hay fugas de información porque el split train/test se hace en `02_limpieza.ipynb` antes de
cualquier imputación o feature engineering, y todo parámetro (medianas, categorías frecuentes, encoders, tablas de
agregación) se aprende solo del train y se aplica igual al test.

```bash
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/01_exploracion.ipynb
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/02_limpieza.ipynb
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/04_balanceo.ipynb
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/05_modelo.ipynb
```

(o simplemente ábrelos en Jupyter/VS Code y ejecuta todas las celdas en orden).

`data/raw/`, `data/processed/` y los modelos pesados en `outputs/` (`*.pkl`, `*.joblib`) **no están versionados en
git** — se regeneran automáticamente corriendo los notebooks en el orden de arriba una vez descargados los datos
crudos.

## Resumen del pipeline

- **01_exploracion**: dimensiones, tipos de datos, % de faltantes por columna, distribución de `isFraud` (~3.5% de
  fraude), EDA de `TransactionAmt`, `ProductCD`, `card4`/`card6` y patrones temporales vía `TransactionDT`.
- **02_limpieza**: split estratificado 80/20 por `isFraud` (`random_state=42`) inmediatamente después del merge y
  antes de imputar nada; eliminación de columnas con >90% de faltantes (12 columnas); imputación por mediana
  (numéricas) o categoría `"missing"` explícita (categóricas), ajustada solo en train; agrupación de categorías
  raras (<0.1% de frecuencia en train) como `"rare"`; codificación con `OrdinalEncoder` (`unknown_value=-1` para
  categorías nuevas en test).
- **03_feature_engineering**: hora del día / día de semana desde `TransactionDT`; parte decimal y log1p de
  `TransactionAmt`; frecuencia y media de `TransactionAmt` por `card1`, `card2`, `addr1` y por el UID
  `card1+addr1+D1n` (técnica de "magic feature" documentada por Chris Deotte para esta competencia); features de
  dominio de email (proveedor agrupado, coincidencia P/R); reducción de las columnas `V1-V339` por correlación >0.90
  (161 eliminadas de 339, técnica también documentada en soluciones top de esta competencia). Feature set final:
  274 columnas, guardadas en `data/processed/train_fe.parquet` (472,432 filas) y `test_fe.parquet` (118,108 filas).
- **04_balanceo**: comparación empírica de `SMOTE` vs. `class_weight='balanced'` sobre el train. Se decide usar
  **`class_weight='balanced'`** para ambos modelos: la mejora de recall/AUPRC de SMOTE no compensa su costo
  computacional ni el riesgo de generar transacciones sintéticas poco realistas en un espacio de 274 dimensiones
  altamente correlacionadas, y `class_weight` está soportado nativamente por los dos modelos usados en el paso
  siguiente.
- **05_modelo**: `LogisticRegression` y `RandomForestClassifier` (scikit-learn, sin gradient boosting), validación
  cruzada estratificada de 5 folds, diagnóstico explícito de overfitting train-vs-validación y train-vs-test, y
  mitigación aplicada donde hizo falta.

## Diagnóstico y mitigación de overfitting

Con hiperparámetros por defecto / poco restringidos, Random Forest memorizaba el train (CV, 5 folds):

| Modelo | Métrica | Gap train−val (antes) | Gap train−val (después) |
|---|---|---:|---:|
| LogisticRegression (C=100 → C=0.01) | ROC-AUC | 0.0036 | 0.0033 |
| LogisticRegression (C=100 → C=0.01) | AUPRC | 0.0042 | 0.0041 |
| RandomForest (sin límite → `max_depth=12, min_samples_leaf=50`) | ROC-AUC | 0.0450 | 0.0175 |
| RandomForest (sin límite → `max_depth=12, min_samples_leaf=50`) | AUPRC | **0.2127** | **0.0412** |
| RandomForest (sin límite → `max_depth=12, min_samples_leaf=50`) | F1 | **0.2806** | **0.0139** |

Random Forest sin restricciones llegaba a memorizar casi perfectamente el train (F1 de train ≈ 0.998, recall de
train = 1.0), con una caída brusca en validación. Al limitar `max_depth=12`, `min_samples_leaf=50` y
`max_features='sqrt'` (y regularizar Logistic Regression con `C=0.01`), la brecha train-validación se redujo
drásticamente sin sacrificar el desempeño en validación.

## Resultados finales (holdout de test, 118,108 filas)

| Modelo | AUC-ROC | AUPRC | F1 | Precisión | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (C=0.01, class_weight='balanced') | 0.863 | 0.424 | 0.230 | 0.136 | 0.737 |
| **Random Forest** (max_depth=12, min_samples_leaf=50, class_weight='balanced') | **0.909** | **0.574** | **0.323** | **0.204** | **0.776** |

**Modelo recomendado: Random Forest.** Domina a Logistic Regression en las 5 métricas sobre el holdout de test, con
una brecha train-test controlada tras la mitigación de overfitting (ROC-AUC: 0.925 train vs. 0.909 test; AUPRC:
0.607 train vs. 0.574 test). En un caso de fraude bancario el costo de un falso negativo (fraude no detectado) suele
ser mucho mayor que el de un falso positivo (transacción legítima marcada para revisión manual), por lo que
priorizamos recall y AUPRC (más informativo que ROC-AUC bajo desbalance fuerte) por encima de la precisión bruta:
Random Forest ofrece el mejor recall (77.6% de los fraudes detectados en test) manteniendo además mejor precisión
que Logistic Regression, gracias a que puede capturar interacciones no lineales entre las features de agregación
(`card`/`addr`/UID) que un modelo lineal no modela directamente. Logistic Regression sigue siendo útil como modelo
base interpretable y mucho más barato de reentrenar/servir.

Los modelos entrenados (`outputs/logistic_regression_final.joblib`, `outputs/random_forest_final.joblib`,
`outputs/scaler.joblib`) y el resumen de métricas (`outputs/metrics_summary.json`) se generan al ejecutar
`05_modelo.ipynb`; no están versionados en git por su peso — corre el notebook para regenerarlos.
