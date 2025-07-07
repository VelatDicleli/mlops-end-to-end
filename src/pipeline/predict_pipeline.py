import os
from typing import Literal
import joblib
import numpy as np
from src.exception import exception_handler_decorator
from src.logger import logging
from pydantic import BaseModel, Field
import pandas as pd


class PredictPipeline:
    def __init__(self, preprocessor_path: str, model_path: str):
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)

    @exception_handler_decorator
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        logging.info("Starting prediction...")
        
        
        processed_data = self.preprocessor.transform(input_data)
        
        
        predictions = self.model.predict(processed_data)
        
        logging.info("Prediction completed.")
        
        return predictions
    
    

class CustomData(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Family_Income: float
    Teacher_Quality: float
    Peer_Influence: float
    Physical_Activity: float

    Parental_Involvement: Literal["Low", "Medium", "High"]
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Extracurricular_Activities: Literal["Yes", "No"]
    Motivation_Level: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["Yes", "No"]
    School_Type: Literal["Public", "Private"]
    Learning_Disabilities: Literal["Yes", "No"]
    Parental_Education_Level: Literal["High School", "College", "Postgraduate"]
    Distance_from_Home: Literal["Near", "Moderate", "Far"]
    Gender: Literal["Male", "Female", "Other"]

    def get_features(self) -> np.ndarray:
        data_dict = {
        "Hours_Studied": self.Hours_Studied,
        "Attendance": self.Attendance,
        "Parental_Involvement": self.Parental_Involvement,
        "Access_to_Resources": self.Access_to_Resources,
        "Extracurricular_Activities": self.Extracurricular_Activities,
        "Sleep_Hours": self.Sleep_Hours,
        "Previous_Scores": self.Previous_Scores,
        "Motivation_Level": self.Motivation_Level,
        "Internet_Access": self.Internet_Access,
        "Tutoring_Sessions": self.Tutoring_Sessions,
        "Family_Income": self.Family_Income,
        "Teacher_Quality": self.Teacher_Quality,
        "School_Type": self.School_Type,
        "Peer_Influence": self.Peer_Influence,
        "Physical_Activity": self.Physical_Activity,
        "Learning_Disabilities": self.Learning_Disabilities,
        "Parental_Education_Level": self.Parental_Education_Level,
        "Distance_from_Home": self.Distance_from_Home,
        "Gender": self.Gender
    }

        df = pd.DataFrame([data_dict])
        return df