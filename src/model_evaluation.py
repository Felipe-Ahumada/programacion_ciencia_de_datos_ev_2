"""
Módulo de evaluación y comparación de modelos para el proyecto de predicción de churn.

Proporciona funciones para calcular métricas de clasificación y regresión,
generar visualizaciones comparativas y exportar resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, r2_score, mean_squared_error,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)


def evaluar_clasificacion(modelo, X_train, X_test, y_train, y_test) -> dict:
    """
    Entrena el modelo y calcula métricas de clasificación sobre el conjunto de prueba.

    Parameters
    ----------
    modelo : sklearn estimator
        Modelo o pipeline de Scikit-learn con interfaz fit/predict.
    X_train, X_test : array-like
        Features de entrenamiento y prueba.
    y_train, y_test : array-like
        Etiquetas de entrenamiento y prueba.

    Returns
    -------
    dict
        Diccionario con métricas: accuracy, f1, precision, recall, roc_auc.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob)
    }


def evaluar_regresion(modelo, X_train, X_test, y_train, y_test) -> dict:
    """
    Entrena el modelo y calcula métricas de regresión sobre el conjunto de prueba.

    Parameters
    ----------
    modelo : sklearn estimator
        Modelo o pipeline de Scikit-learn con interfaz fit/predict.
    X_train, X_test : array-like
    y_train, y_test : array-like

    Returns
    -------
    dict
        Diccionario con métricas: r2, mae, rmse.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return {
        'r2':   r2_score(y_test, y_pred),
        'mae':  mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def tabla_comparativa(resultados: dict) -> pd.DataFrame:
    """
    Construye un DataFrame comparativo de métricas para múltiples modelos.

    Parameters
    ----------
    resultados : dict
        Diccionario con nombre de modelo como clave y dict de métricas como valor.
        Ej: {'LogReg': {'accuracy': 0.65, 'f1': 0.57, ...}}

    Returns
    -------
    pd.DataFrame
        Tabla con modelos como índice y métricas como columnas.
    """
    return pd.DataFrame(resultados).T.round(4)


def graficar_matrices_confusion(modelos_dict: dict, X_test, y_test,
                                 figsize=(15, 5), save_path=None):
    """
    Genera matrices de confusión comparativas para múltiples modelos.

    Parameters
    ----------
    modelos_dict : dict
        Diccionario {nombre_modelo: pipeline_entrenado}.
    X_test : array-like
    y_test : array-like
    figsize : tuple, default=(15, 5)
    save_path : str or None
        Ruta para guardar la figura (opcional).
    """
    fig, axes = plt.subplots(1, len(modelos_dict), figsize=figsize)
    if len(modelos_dict) == 1:
        axes = [axes]

    for ax, (nombre, pipeline) in zip(axes, modelos_dict.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Permanece (0)', 'Abandona (1)']
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(nombre, fontsize=11, fontweight='bold')

    plt.suptitle('Matrices de Confusión — Modelos de Clasificación', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def graficar_curvas_roc(modelos_dict: dict, X_test, y_test,
                         figsize=(8, 7), save_path=None):
    """
    Genera curvas ROC comparativas para múltiples clasificadores.

    Parameters
    ----------
    modelos_dict : dict
        Diccionario {nombre_modelo: pipeline_entrenado}.
    X_test : array-like
    y_test : array-like
    figsize : tuple, default=(8, 7)
    save_path : str or None
    """
    colores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    fig, ax = plt.subplots(figsize=figsize)

    for (nombre, pipeline), color in zip(modelos_dict.items(), colores):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.3f})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador aleatorio (AUC = 0.500)')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=12)
    ax.set_title('Curvas ROC Comparativas — Predicción de Churn', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
