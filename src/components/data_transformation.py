import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = Path('artifacts') / "preprocessor.pkl"

class DataTransformation:
    """
    Class responsible for data transformation using preprocessing techniques.

    Methods:
    - get_data_transformer_object(): Get the data transformer object for preprocessing.
    - initiate_data_transformation(train_path, test_path): Apply data transformation to training 
        and testing datasets.

    Attributes:
    - data_transformation_config (DataTransformationConfig): Configuration object for DataTransformation.
    """

    IMPUTATION_STRATEGY_MEDIAN = "median"
    IMPUTATION_STRATEGY_MOST_FREQUENT = "most_frequent"

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _load_csv(self, file_path):
        """
        Load CSV file and return DataFrame.

        Parameters:
        - file_path (Path): Path to the CSV file.

        Returns:
        - pd.DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(file_path)

    def get_data_transformer_object(self):
        """
        Get the data transformer object for preprocessing.

        Returns:
        - ColumnTransformer: Preprocessor object for numerical and categorical data.

        Raises:
        - CustomException: If any exception occurs during the process.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.IMPUTATION_STRATEGY_MEDIAN)),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.IMPUTATION_STRATEGY_MOST_FREQUENT)),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply data transformation to training and testing datasets.

        Parameters:
        - train_path (Path): Path to the training dataset.
        - test_path (Path): Path to the testing dataset.

        Returns:
        - Tuple of transformed training array, transformed testing array, and preprocessor object file path.

        Raises:
        - CustomException: If any exception occurs during the process.
        """
        try:
            train_df = self._load_csv(train_path)
            test_df = self._load_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
