import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, X_test, y_train, y_test = (
                train_array[:,:-1], # remove last column
                test_array[:,:-1], # remove the last column
                train_array[:, -1], #only keep the last column
                test_array[:, -1] # only keep the last column
            )
            model = {'CatBoosting Regressor': CatBoostRegressor(verbose=False)} #Only using one model but can use others
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=model)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]
            
            # Set a threshold
            if best_model_score<0.7:
                raise CustomException("No best model found", e, sys)
            
            logging.info(f"Best found model on testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_squared_score = r2_score(y_test, predicted)
            return r2_squared_score

        except Exception as e:
            raise CustomException(e, sys)