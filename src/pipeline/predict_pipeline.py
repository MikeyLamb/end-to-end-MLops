import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    A class representing a prediction pipeline.

    Methods:
    - __init__(): Initializes the PredictPipeline object.
    - predict(features): Predicts outcomes using the provided features.

    Attributes:
    - None
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts outcomes using the provided features.

        Parameters:
        - features (numpy array): The input features for prediction.

        Returns:
        - numpy array: Predicted outcomes.

        Raises:
        - CustomException: If any exception occurs during the prediction process.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A class representing custom data for prediction.

    Methods:
    - __init__(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course,
                reading_score, writing_score): Initializes the CustomData object.
    - get_data_as_data_frame(): Converts the custom data to a pandas DataFrame.

    Attributes:
    - gender (str): Gender of the individual.
    - race_ethnicity (str): Race or ethnicity of the individual.
    - parental_level_of_education: Level of education of the parent.
    - lunch (str): Type of lunch.
    - test_preparation_course (str): Test preparation course status.
    - reading_score (int): Reading score.
    - writing_score (int): Writing score.
    """

    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch,
                 test_preparation_course, reading_score, writing_score):
        """
        Initializes the CustomData object.

        Parameters:
        - gender (str): Gender of the individual.
        - race_ethnicity (str): Race or ethnicity of the individual.
        - parental_level_of_education: Level of education of the parent.
        - lunch (str): Type of lunch.
        - test_preparation_course (str): Test preparation course status.
        - reading_score (int): Reading score.
        - writing_score (int): Writing score.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the custom data to a pandas DataFrame.

        Returns:
        - pandas DataFrame: Custom data as a DataFrame.

        Raises:
        - CustomException: If any exception occurs during the conversion process.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
