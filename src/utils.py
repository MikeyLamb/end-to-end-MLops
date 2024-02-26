import os
import sys

import numpy as np 
import pandas as pd
import dill #to load pickle file
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
     Save a Python object to a file using pickle serialization.

    Parameters:
    - file_path (str): The path to the file where the object will be saved.
    - obj: The Python object to be saved.

    Raises:
    - CustomException: If any exception occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    Evaluate machine learning models using grid search for hyperparameter tuning.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_test: Testing data features.
    - y_test: Testing data labels.
    - models (dict): A dictionary containing models to be evaluated.
    - param (dict): A dictionary containing hyperparameter grids for corresponding models.

    Returns:
    - dict: A dictionary containing R^2 scores for each evaluated model.

    Raises:
    - CustomException: If any exception occurs during the evaluation process.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization.

    Parameters:
    - file_path (str): The path to the file from which the object will be loaded.

    Returns:
    - The Python object loaded from the file.

    Raises:
    - CustomException: If any exception occurs during the loading process.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
