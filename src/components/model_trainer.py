import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Example model (Linear Regression)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            logging.info(f"Model R2 Score: {score}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            return score

        except Exception as e:
            raise CustomException(e, sys)
