# Analítica de Clientes — Predicción de Churn y Score Crediticio

Proyecto de machine learning para predecir el abandono de clientes (abandono) y modelar
el puntaje crediticio (score_crediticio) de una empresa de servicios digitales por suscripción.

**Curso:** SCY1101 — Programación para la Ciencia de Datos  
**Autores:** Felipe Ahumada Silva · Francisca Carrasco Lozano

---

## Resultados Clave

| Problema | Modelo seleccionado | Métrica principal |
|---|---|---|
| Clasificación (churn) | DecisionTreeClassifier optimizado | **Recall = 0.782** |
| Regresión (score) | — (sin señal predictiva) | R² ≈ 0.001 |

El árbol de decisión con class_weight='balanced' y max_depth=2 detecta el **78% de los
clientes que abandonan**, priorizando recall porque el costo de perder un cliente sin
detectarlo supera al de activar una retención innecesaria.

---

## Dataset

| Atributo | Valor |
|---|---|
| Archivo | data/dataset_clientes.csv |
| Filas (tras deduplicación) | 20.000 |
| Variables | 22 (8 numéricas · 6 categóricas · 2 targets · metadatos) |
| Target clasificación | abandono (0 = activo · 1 = abandona) · desbalance 67/33 |
| Target regresión | `score_crediticio` (continuo) |

---

## Estructura del Proyecto

```
.
├── data/
│   └── dataset_clientes.csv
├── docs/
│   ├── informe_tecnico.tex       # Informe LaTeX completo
│   ├── Caso_de_estudio.pdf
│   └── SCY1101_E2.pdf
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA: distribuciones, nulos, correlaciones
│   ├── 02_supervised_modeling.ipynb    # Entrenamiento base + validación cruzada
│   ├── 03_model_evaluation.ipynb       # Evaluación comparativa con métricas y plots
│   ├── 04_hyperparameter_optimization.ipynb  # RandomizedSearchCV + GridSearchCV
│   └── 05_final_analysis.ipynb         # Resumen, comparación final y conclusiones
├── results/
│   ├── plots/                    # Figuras guardadas (formato: NN_descripcion.png)
│   ├── metrics/                  # CSVs de métricas
│   └── reports/                  # Resúmenes escritos
├── models/
│   └── trained_models/           # Artefactos serializados
└── src/                          # Módulos auxiliares (referencia)
```

---

## Secuencia de Notebooks

Ejecutar en orden. Cada notebook es **autocontenido** (no depende de artefactos de otros).

| # | Notebook | Contenido |
|---|---|---|
| 01 | 01_exploratory_analysis | Distribuciones, valores nulos, atípicos, correlaciones |
| 02 | 02_supervised_modeling | Pipelines base · validación cruzada clasificación |
| 03 | 03_model_evaluation | Métricas test · plots de comparación |
| 04 | 04_hyperparameter_optimization | RandomizedSearchCV (amplio) → GridSearchCV (fino), scoring=recall |
| 05 | 05_final_analysis | Tabla comparativa base vs tuned · matriz de confusión · ROC · conclusiones |

---

## Arquitectura del Pipeline

Todos los modelos comparten la misma cadena de preprocesamiento:

```
FunctionTransformer(tratar_duplicados)
→ ColumnTransformer
    numéricas:    Winsorizer([p5, p95]) → SimpleImputer(mean)
    categóricas:  SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
→ DataFrameConverter        # restaura nombres de columnas
→ CorrelationFilter(τ=0.9)  # elimina features con |r| > 0.9
→ Estimador
```

> El pipeline SVM añade `StandardScaler` en la rama numérica (distancias sensibles a escala).

---

## Optimización de Hiperparámetros

Búsqueda en dos etapas sobre `StratifiedKFold(n_splits=5)`, métrica: **recall**.

| Etapa | Método | Descripción |
|---|---|---|
| 1 | `RandomizedSearchCV` | Exploración amplia con `loguniform` / `randint` |
| 2 | `GridSearchCV` | Refinamiento local alrededor del óptimo de Etapa 1 |

**Mejores configuraciones encontradas:**

| Modelo | Parámetros | Recall CV | Recall test |
|---|---|---|---|
| **DecisionTreeClassifier** | `class_weight=balanced`, `max_depth=2`, `min_samples_leaf=4` | 0.8037 | **0.782** |
| SVM (RBF) | `C=0.2154`, `class_weight=balanced` | 0.6578 | 0.655 |
| LogisticRegression | `C=0.6549`, `class_weight=balanced` | 0.5474 | 0.572 |

---

## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**Python:** 3.12+

Para abrir los notebooks:

```bash
jupyter notebook
```

---

## Métricas completas — Clasificación (conjunto de prueba)

### Modelos base

| Modelo | Accuracy | F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| DecisionTreeClassifier | 0.569 | 0.451 | 0.456 | 0.446 | 0.548 |
| LogisticRegression | 0.649 | 0.452 | 0.593 | 0.365 | 0.667 |
| SVM (RBF) | 0.650 | 0.412 | 0.616 | 0.309 | 0.657 |

### Modelos optimizados

| Modelo | Accuracy | F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| **DecisionTreeClassifier** | 0.584 | **0.599** | 0.485 | **0.782** | 0.653 |
| LogisticRegression | 0.509 | 0.480 | 0.414 | 0.572 | 0.530 |
| SVM (RBF) | 0.621 | 0.578 | 0.518 | 0.655 | **0.672** |
