import os
from dataclasses import dataclass

import mlflow.artifacts
from src.exception import exception_handler_decorator
from src.logger import logging
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib
from src.utils import evaluate_models
import mlflow
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @exception_handler_decorator
    def train_model(self, train_array, test_array):
        logging.info("Starting model training...")

        X_train = train_array[:, :-1]
        y_train = train_array[:, -1]    
        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        models = {
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "XGBRegressor": xgb.XGBRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor()
        }

        best_model_name = None
        best_model_score = float('-inf')
        best_model = None

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_registry_uri("http://127.0.0.1:5000")

        mlflow.set_experiment("student-score-local")

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                logging.info(f"{name} modeli eğitildi.")

                results = evaluate_models({name: model}, X_train, y_train, X_test, y_test)
                test_r2 = results[name]["test"]["r2"]
                train_r2 = results[name]["train"]["r2"]

                mlflow.log_param("model_name", name)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)

                if name == "CatBoostRegressor":
                    # mlflow.catboost.log_model(model, artifact_path="model", registered_model_name=registered_name)
                    mlflow.log_artifact(self.config.model_file_path, artifact_path="model")
                    
                    
                elif name == "XGBRegressor":
                    # mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=registered_name)
                    mlflow.log_artifact(self.config.model_file_path, artifact_path="model")
                else:
                    # mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=registered_name)
                    mlflow.log_artifact(self.config.model_file_path, artifact_path="model")
                    mlflow.reg

                if test_r2 > best_model_score:
                    best_model_score = test_r2
                    best_model_name = name
                    best_model = model

        logging.info(f"En iyi model: {best_model_name} (R²: {best_model_score:.4f})")
        joblib.dump(best_model, self.config.model_file_path)
        logging.info(f"Model kaydedildi: {self.config.model_file_path}")

        return {
            "best_model_name": best_model_name,
            "best_model_score": best_model_score,
            "model_path": self.config.model_file_path
        }
