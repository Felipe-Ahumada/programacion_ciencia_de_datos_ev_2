"""
Módulo de entrenamiento de modelos supervisados para el proyecto de predicción de churn.

Define las funciones de construcción y entrenamiento de los cinco modelos
requeridos por la rúbrica: LinearRegression, DecisionTreeRegressor (regresión)
y LogisticRegression, DecisionTreeClassifier, SVC (clasificación).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from data_preprocessing import (
    Winsorizer, CorrelationFilter, DataFrameConverter,
    tratar_duplicados, construir_preprocesador
)

SEED = 29

FEATURES_NUM = [
    'edad', 'ingreso_mensual', 'gasto_mensual', 'deuda_total',
    'antiguedad_meses', 'frecuencia_compra', 'ultima_compra_dias', 'num_productos'
]
FEATURES_CAT = [
    'genero', 'region', 'estado_civil', 'uso_app', 'tipo_plan', 'canal_registro'
]


def construir_pipeline_regresion_lineal(features_num=FEATURES_NUM,
                                         features_cat=FEATURES_CAT) -> Pipeline:
    """
    Construye el pipeline completo para LinearRegression.

    Incluye: eliminación de duplicados → preprocesamiento → filtro de colinealidad
    → LinearRegression.

    Returns
    -------
    Pipeline
        Pipeline de Scikit-learn listo para fit/predict.
    """
    preprocesador = construir_preprocesador(features_num, features_cat, escalar=True)
    return Pipeline(steps=[
        ('duplicados', FunctionTransformer(tratar_duplicados, kw_args={'drop': False})),
        ('preprocesador', preprocesador),
        ('conversion', DataFrameConverter(preprocesador)),
        ('colinealidad', CorrelationFilter(threshold=0.9)),
        ('modelo', LinearRegression())
    ])


def construir_pipeline_arbol_regresion(max_depth=2, min_samples_leaf=200,
                                        min_samples_split=100,
                                        features_num=FEATURES_NUM,
                                        features_cat=FEATURES_CAT) -> Pipeline:
    """
    Construye el pipeline completo para DecisionTreeRegressor.

    Parameters
    ----------
    max_depth : int, default=2
        Profundidad máxima del árbol (regularización).
    min_samples_leaf : int, default=200
        Mínimo de muestras en nodo hoja.
    min_samples_split : int, default=100
        Mínimo de muestras para dividir un nodo.

    Returns
    -------
    Pipeline
    """
    preprocesador = construir_preprocesador(features_num, features_cat, escalar=True)
    return Pipeline(steps=[
        ('duplicados', FunctionTransformer(tratar_duplicados, kw_args={'drop': False})),
        ('preprocesador', preprocesador),
        ('conversion', DataFrameConverter(preprocesador)),
        ('colinealidad', CorrelationFilter(threshold=0.9)),
        ('modelo', DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=SEED
        ))
    ])


def construir_pipeline_regresion_logistica(C=1.0, class_weight='balanced',
                                            features_num=FEATURES_NUM,
                                            features_cat=FEATURES_CAT) -> Pipeline:
    """
    Construye el pipeline completo para LogisticRegression.

    Parameters
    ----------
    C : float, default=1.0
        Inverso de la fuerza de regularización L2.
    class_weight : str or dict, default='balanced'
        Tratamiento del desbalance de clases.

    Returns
    -------
    Pipeline
    """
    preprocesador = construir_preprocesador(features_num, features_cat, escalar=True)
    return Pipeline(steps=[
        ('duplicados', FunctionTransformer(tratar_duplicados, kw_args={'drop': False})),
        ('preprocesador', preprocesador),
        ('colinealidad', CorrelationFilter(threshold=0.9)),
        ('modelo', LogisticRegression(
            C=C, class_weight=class_weight,
            max_iter=10000, random_state=SEED
        ))
    ])


def construir_pipeline_arbol_clasificacion(max_depth=None, class_weight='balanced',
                                            features_num=FEATURES_NUM,
                                            features_cat=FEATURES_CAT) -> Pipeline:
    """
    Construye el pipeline completo para DecisionTreeClassifier.

    Parameters
    ----------
    max_depth : int or None, default=None
        Profundidad máxima del árbol (None = árbol completo).
    class_weight : str or dict, default='balanced'
        Tratamiento del desbalance de clases.

    Returns
    -------
    Pipeline
    """
    preprocesador = construir_preprocesador(features_num, features_cat, escalar=False)
    return Pipeline(steps=[
        ('duplicados', FunctionTransformer(tratar_duplicados, kw_args={'drop': False})),
        ('preprocesador', preprocesador),
        ('colinealidad', CorrelationFilter(threshold=0.9)),
        ('modelo', DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=SEED
        ))
    ])


def construir_pipeline_svm(C=1.0, kernel='rbf', class_weight='balanced',
                            features_num=FEATURES_NUM,
                            features_cat=FEATURES_CAT) -> Pipeline:
    """
    Construye el pipeline completo para SVC (SVM).

    Incluye StandardScaler obligatorio, ya que SVM es sensible a la escala.

    Parameters
    ----------
    C : float, default=1.0
        Parámetro de penalización.
    kernel : str, default='rbf'
        Tipo de kernel ('rbf', 'linear', 'poly', 'sigmoid').
    class_weight : str or dict, default='balanced'
        Tratamiento del desbalance de clases.

    Returns
    -------
    Pipeline
    """
    preprocesador = construir_preprocesador(features_num, features_cat, escalar=True)
    return Pipeline(steps=[
        ('duplicados', FunctionTransformer(tratar_duplicados, kw_args={'drop': False})),
        ('preprocesador', preprocesador),
        ('colinealidad', CorrelationFilter(threshold=0.9)),
        ('modelo', SVC(
            C=C, kernel=kernel, class_weight=class_weight,
            probability=True, random_state=SEED
        ))
    ])
