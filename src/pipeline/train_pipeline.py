from src.workflow.data_transformation import DataTransformationConfig, DataTransformation
from src.workflow.model_trainer import ModelTrainerConfig, ModelTrainer
from src.workflow.data_ingestion import DataIngestion, DataIngestionConfig
from src.logger import logging

def main():

    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)

    train_file, test_file = data_ingestion.get_data()

    logging.info(f"Train file: {train_file}")
    logging.info(f"Test file: {test_file}")


    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_transformation_config)
    train_array, test_array = data_transformation.get_transformed_data(train_file, test_file)

    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)
    model_trainer.train_model(train_array, test_array)
    logging.info("Model training completed successfully.")


if __name__ == "__main__":
    main()