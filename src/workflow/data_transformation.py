import sys
import os
from dataclasses import dataclass
from src.exception import exception_handler_decorator
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    
    
    
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @exception_handler_decorator
    def get_transformed_data(self, train_data_file: str, test_data_file: str):
        
        logging.info("Starting data transformation...")

     
        train_df = pd.read_csv(train_data_file)
        test_df = pd.read_csv(test_data_file)

        
        target_column = 'Exam_Score'
        
     
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
    
        
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_train .select_dtypes(include=['object','category']).columns.tolist()
        
          
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        
        logging.info("Transforming training and test data...")
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        logging.info("Data transformation completed successfully.")
        
        train_arr = np.c_[X_train_transformed, y_train.values]
        
        test_arr = np.c_[X_test_transformed, y_test.values]  
        
        logging.info("Saving preprocessor to file...")
        
        joblib.dump(preprocessor, self.config.preprocessor_file_path)
        
        logging.info(f"Preprocessor saved to {self.config.preprocessor_file_path}")
        
        logging.info("Data transformation process completed.")
        
        return train_arr, test_arr, self.config.preprocessor_file_path
        
          

        