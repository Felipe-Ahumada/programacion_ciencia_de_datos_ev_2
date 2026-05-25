"""
Módulo de preprocesamiento de datos para el proyecto de predicción de churn.

Contiene transformadores personalizados de Scikit-learn y funciones de utilidad
para la limpieza, transformación y preparación del dataset de clientes.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica winsorización por percentil para tratar valores atípicos.

    Recorta los valores de cada columna al rango [percentil_inf, percentil_sup],
    preservando el número de filas y reduciendo la influencia de outliers extremos
    sobre modelos lineales y SVM.

    Parameters
    ----------
    limits : tuple of float, default=(0.05, 0.05)
        Percentiles inferior y superior para el recorte (ej: 0.05 = percentil 5%).
    """

    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        """Aprende los percentiles de recorte sobre el conjunto de entrenamiento."""
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        """Aplica el recorte usando los percentiles aprendidos en fit."""
        X = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X = X.astype('float64')
            X[col] = np.clip(X[col], lower, upper)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.columns_)
        return np.array(input_features)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Transformador que elimina variables con alta correlación para reducir
    la multicolinealidad, que infla la varianza de los coeficientes en regresión lineal.

    Parameters
    ----------
    threshold : float, default=0.9
        Umbral de correlación de Pearson por encima del cual se elimina una variable.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_drop_ = None

    def fit(self, X, y=None):
        """Identifica las columnas a eliminar por alta correlación."""
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.columns_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        """Elimina las columnas identificadas en fit."""
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=self.columns_to_drop_, errors='ignore').values


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """
    Convierte la salida de ColumnTransformer (array numpy) en un DataFrame
    con nombres de columna, necesario para que CorrelationFilter opere correctamente.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        El preprocesador del que se extraen los nombres de columna.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.feature_names_ = self.preprocessor.get_feature_names_out()
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_)


def tratar_duplicados(X: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    """
    Tratamiento de registros duplicados.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de entrada.
    drop : bool, default=True
        Si True, elimina filas completamente duplicadas.

    Returns
    -------
    pd.DataFrame
        DataFrame sin duplicados (si drop=True).
    """
    return X.drop_duplicates() if drop else X


def construir_preprocesador(features_num: list, features_cat: list,
                             escalar: bool = True) -> ColumnTransformer:
    """
    Construye un ColumnTransformer con preprocesamiento estándar para el proyecto.

    Aplica la siguiente secuencia para variables numéricas:
        1. Winsorización (percentil 5%-95%)
        2. Imputación por media
        3. Escalado estándar (opcional, requerido para SVM y LogReg)

    Y para variables categóricas:
        1. Imputación por moda
        2. One-Hot Encoding con drop='first' (evita dummy variable trap)

    Parameters
    ----------
    features_num : list
        Lista de nombres de columnas numéricas.
    features_cat : list
        Lista de nombres de columnas categóricas.
    escalar : bool, default=True
        Si True, incluye StandardScaler en el pipeline numérico.

    Returns
    -------
    ColumnTransformer
        Preprocesador configurado.
    """
    pasos_num = [
        ('winsorizer', Winsorizer()),
        ('imputer', SimpleImputer(strategy='mean')),
    ]
    if escalar:
        pasos_num.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=pasos_num)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_num),
            ('cat', categorical_transformer, features_cat)
        ],
        remainder='drop',
        force_int_remainder_cols=False
    )
