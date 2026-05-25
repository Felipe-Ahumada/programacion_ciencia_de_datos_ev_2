"""
Módulo de optimización de hiperparámetros para el proyecto de predicción de churn.

Implementa funciones de búsqueda con GridSearchCV y RandomizedSearchCV,
con espacios de búsqueda predefinidos para los tres modelos de clasificación.

Referencia: Bergstra & Bengio (2012) - Random Search for Hyper-Parameter Optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import loguniform, randint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 29
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def grid_search(pipeline, param_grid: dict, X_train, y_train,
                scoring='recall', n_jobs=-1) -> GridSearchCV:
    """
    Ejecuta GridSearchCV sobre el pipeline dado.

    Realiza búsqueda exhaustiva sobre todas las combinaciones de param_grid,
    evaluando con validación cruzada estratificada de 5 folds.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Pipeline completo (preprocesamiento + modelo).
    param_grid : dict
        Diccionario con nombre de hiperparámetro (prefijado con 'modelo__') y valores.
        Ej: {'modelo__max_depth': [3, 5, 10]}
    X_train, y_train : array-like
        Datos de entrenamiento.
    scoring : str, default='recall'
        Métrica de optimización. Se usa 'recall' dado el costo asimétrico del churn.
    n_jobs : int, default=-1
        Número de workers paralelos (-1 = todos los disponibles).

    Returns
    -------
    GridSearchCV
        Objeto ajustado con atributos best_params_, best_score_, best_estimator_.
    """
    gs = GridSearchCV(
        pipeline, param_grid,
        cv=CV, scoring=scoring, n_jobs=n_jobs,
        return_train_score=True
    )
    gs.fit(X_train, y_train)
    print(f"[GridSearchCV] Mejor {scoring} CV: {gs.best_score_:.4f}")
    print(f"  Mejores parámetros: {gs.best_params_}")
    return gs


def random_search(pipeline, param_distributions: dict, X_train, y_train,
                  n_iter=60, scoring='recall', n_jobs=-1) -> RandomizedSearchCV:
    """
    Ejecuta RandomizedSearchCV sobre el pipeline dado.

    Muestrea aleatoriamente n_iter combinaciones del espacio de búsqueda,
    siendo más eficiente que GridSearch para espacios de alta dimensionalidad
    y permite usar distribuciones continuas (loguniform, randint).

    Parameters
    ----------
    pipeline : sklearn Pipeline
    param_distributions : dict
        Diccionario con distribuciones o listas de valores para cada hiperparámetro.
        Ej: {'modelo__C': loguniform(1e-3, 1e2), 'modelo__max_depth': randint(2, 20)}
    X_train, y_train : array-like
    n_iter : int, default=60
        Número de combinaciones a evaluar.
    scoring : str, default='recall'
    n_jobs : int, default=-1

    Returns
    -------
    RandomizedSearchCV
        Objeto ajustado con atributos best_params_, best_score_, best_estimator_.
    """
    rs = RandomizedSearchCV(
        pipeline, param_distributions,
        n_iter=n_iter, cv=CV, scoring=scoring,
        random_state=SEED, n_jobs=n_jobs,
        return_train_score=True
    )
    rs.fit(X_train, y_train)
    print(f"[RandomizedSearchCV] Mejor {scoring} CV: {rs.best_score_:.4f}")
    print(f"  Mejores parámetros: {rs.best_params_}")
    return rs


def get_param_grid_dtc() -> dict:
    """
    Retorna la grilla de hiperparámetros para DecisionTreeClassifier.

    Incluye max_depth (complejidad), min_samples_split y min_samples_leaf
    (regularización suave) y class_weight (desbalance de clases).
    """
    return {
        'modelo__max_depth':         [3, 5, 10],
        'modelo__min_samples_split': [2, 5, 10],
        'modelo__min_samples_leaf':  [1, 2, 4],
        'modelo__class_weight':      [None, 'balanced']
    }


def get_param_dist_dtc() -> dict:
    """
    Retorna las distribuciones para RandomizedSearchCV sobre DecisionTreeClassifier.

    Usa randint para explorar rangos continuos de profundidad y tamaños de hoja,
    cubriendo un espacio de búsqueda más amplio que la grilla discreta.
    """
    return {
        'modelo__max_depth':         randint(2, 20),
        'modelo__min_samples_split': randint(2, 30),
        'modelo__min_samples_leaf':  randint(1, 20),
        'modelo__class_weight':      ['balanced', None]
    }


def get_param_grid_logreg() -> dict:
    """
    Retorna la grilla de hiperparámetros para LogisticRegression.

    C controla la regularización L2: valores bajos → mayor regularización.
    """
    return {
        'modelo__C':            [0.01, 0.1, 1, 10],
        'modelo__class_weight': [None, 'balanced']
    }


def get_param_dist_logreg() -> dict:
    """
    Retorna las distribuciones para RandomizedSearchCV sobre LogisticRegression.

    loguniform permite muestrear C en escala logarítmica, apropiado para
    parámetros de regularización que varían en órdenes de magnitud.
    """
    return {
        'modelo__C':            loguniform(1e-3, 1e2),
        'modelo__class_weight': ['balanced', None]
    }


def get_param_grid_svm() -> dict:
    """Retorna la grilla de hiperparámetros para SVC."""
    return {
        'modelo__C':            [0.1, 1, 10],
        'modelo__kernel':       ['rbf', 'linear'],
        'modelo__class_weight': [None, 'balanced']
    }


def get_param_dist_svm() -> dict:
    """
    Retorna las distribuciones para RandomizedSearchCV sobre SVC.

    loguniform para C permite explorar ampliamente el espacio de penalización.
    Se incluye gamma como hiperparámetro adicional del kernel RBF.
    """
    return {
        'modelo__C':            loguniform(1e-2, 1e2),
        'modelo__kernel':       ['rbf', 'linear'],
        'modelo__gamma':        ['scale', 'auto'],
        'modelo__class_weight': ['balanced', None]
    }


def graficar_impacto_optimizacion(comparacion_df: pd.DataFrame,
                                   metricas=('Recall', 'F1'),
                                   save_path=None):
    """
    Visualiza el impacto de la optimización de hiperparámetros comparando
    Baseline vs. GridSearchCV vs. RandomizedSearchCV.

    Parameters
    ----------
    comparacion_df : pd.DataFrame
        DataFrame con columnas: 'Modelo', 'Método', y las métricas a graficar.
    metricas : tuple, default=('Recall', 'F1')
        Métricas a visualizar.
    save_path : str or None
    """
    colores_metodo = {
        'Baseline': '#888888',
        'GridSearchCV': '#4C72B0',
        'RandomizedSearchCV': '#DD8452'
    }
    fig, axes = plt.subplots(1, len(metricas), figsize=(7 * len(metricas), 6))
    if len(metricas) == 1:
        axes = [axes]

    for ax, metrica in zip(axes, metricas):
        for metodo, color in colores_metodo.items():
            subset = comparacion_df[comparacion_df['Método'] == metodo]
            ax.scatter(subset['Modelo'], subset[metrica],
                       label=metodo, color=color, s=100, zorder=5)
        ax.set_title(f'Impacto en {metrica}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metrica, fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=9)

    plt.suptitle('Análisis del Impacto de la Optimización de Hiperparámetros',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
