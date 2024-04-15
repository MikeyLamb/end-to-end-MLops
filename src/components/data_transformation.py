import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # I am not using this now, but could be useful in the future
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# Any paths that I will require for data transformation
# Again data class is a simplter way to store data. There is no need to write the init function
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Function that transforms the data based on numerical and categorical data
        """
        try:
            cat_columns = ['Suburb']
            num_columns = ['Bedrooms','Bathrooms', 'Parking_spaces']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                ]
            )

            logging.info('Categorical columns encoding complete')
            logging.info('Numerical columns scaler complete')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
            
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Read data, split and save data
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()

            # Find the relevant columns
            target_column_name='Price'
            num_column = ['Bedrooms', 'Bathrooms', 'Parking_spaces']
            
            # Get features for train
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            # Get features for test
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            

            # Transform features but not including Price, returns array
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f'Input features have been transformed successfully')


            target_arr = np.array(target_feature_train_df)           

            # Concat the arrays column wise
            # Have to use toarray() because the OHE caused many zeros and thus a sparse matrix 
            #to be created instead of a dense matrix
            train_arr = np.c_[input_feature_train_arr.toarray(), target_arr]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]
            

            logging.info(f'Train_array successfully created')

            logging.info(f"Saved preprocessing object.")

            # Save objects
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