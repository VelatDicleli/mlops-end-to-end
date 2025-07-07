

from src.exception import exception_handler_decorator
from src.logger import logging

from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split
import pandas as pd

from src.workflow.data_transformation import DataTransformationConfig, DataTransformation



@dataclass
class DataIngestionConfig:

    raw_data_dir: str = os.path.join("data","StudentPerformanceFactors.csv")
    train_data_file: str = os.path.join("data", "processed", "train.csv")
    test_data_file: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @exception_handler_decorator
    def get_data(self):

        logging .info(f"Getting dataset from {self.config.raw_data_dir}...")

        df = pd.read_csv(self.config.raw_data_dir)
        
        logging.info("Data retrieved successfully.")
        
        os.makedirs(os.path.dirname(self.config.train_data_file), exist_ok=True)
        df.to_csv(self.config.raw_data_dir, index=False, header=True)

        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        train_data.to_csv(self.config.train_data_file, index=False, header=True)
        test_data.to_csv(self.config.test_data_file, index=False, header=True)

        logging.info("Data split into train and test sets.")
        
        return self.config.train_data_file, self.config.test_data_file
        

   
