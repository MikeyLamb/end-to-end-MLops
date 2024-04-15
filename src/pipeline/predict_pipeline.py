import sys
import pandas as pd
import numpy as np


from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scale=preprocessor.transform(features)
            pred = model.predict(data_scale)
            answer = np.exp(pred)
            return answer
        
        except Exception as e:
             raise CustomException(e, sys)


# Map inputs from html to backend
class CustomData:
    def __init__(self, 
                Bedrooms: int,
                Bathrooms: int,
                Suburb: str,
                Parking_spaces: int,):
                
                self.Bedrooms = Bedrooms
                self.Bathrooms = Bathrooms
                self.Suburb = Suburb
                self.Parking_spaces=Parking_spaces

    def get_data_as_data_frame(self):
        """
        Returns a dataframe
        """
        try:
            data_dict_input = {
                      'Suburb': [self.Suburb],
                      'Bedrooms': [self.Bedrooms],
                      'Bathrooms': [self.Bathrooms],
                      'Parking_spaces': [self.Parking_spaces]
                }
            df = pd.DataFrame(data_dict_input)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    

    