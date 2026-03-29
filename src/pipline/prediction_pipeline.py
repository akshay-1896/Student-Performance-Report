"""
Prediction Pipeline for Student Performance Predictor
======================================================
Handles prediction requests and returns Pass/Fail results with probabilities.
"""

import sys
import os
from src.exception import MyException
from src.logger import logging
import pandas as pd
from src.entity.config_entity import StudentPerformancePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.utils.main_utils import load_object


class StudentData:
    """Student Data constructor for prediction"""
    def __init__(self,
                Study_Hours,
                Sleep_Hours,
                Attendance_Percentage,
                Previous_Score,
                Internet_Usage,
                Social_Activity_Level
                ):
        try:
            self.Study_Hours = Study_Hours
            self.Sleep_Hours = Sleep_Hours
            self.Attendance_Percentage = Attendance_Percentage
            self.Previous_Score = Previous_Score
            self.Internet_Usage = Internet_Usage
            self.Social_Activity_Level = Social_Activity_Level

        except Exception as e:
            raise MyException(e, sys)

    def get_student_input_data_frame(self) -> pd.DataFrame:
        """This function returns a DataFrame from StudentData class input"""
        try:
            student_input_dict = self.get_student_data_as_dict()
            return pd.DataFrame(student_input_dict)

        except Exception as e:
            raise MyException(e, sys)

    def get_student_data_as_dict(self):
        """This function returns a dictionary from StudentData class input"""
        logging.info("Entered get_student_data_as_dict method as StudentData class")

        try:
            input_data = {
                "Study_Hours": [self.Study_Hours],
                "Sleep_Hours": [self.Sleep_Hours],
                "Attendance_Percentage": [self.Attendance_Percentage],
                "Previous_Score": [self.Previous_Score],
                "Internet_Usage": [self.Internet_Usage],
                "Social_Activity_Level": [self.Social_Activity_Level]
            }

            logging.info("Created student data dict")
            logging.info("Exited get_student_data_as_dict method as StudentData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys)


class StudentPerformanceClassifier:
    """Main prediction class for student performance"""
    def __init__(self, prediction_pipeline_config: StudentPerformancePredictorConfig = StudentPerformancePredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model = None
            self.load_model()
        except Exception as e:
            raise MyException(e, sys)

    def load_model(self):
        """Load the trained model"""
        try:
            # Try loading from local path first (for development)
            local_model_path = "artifact/model_trainer/trained_model/model.pkl"

            if os.path.exists(local_model_path):
                logging.info(f"Loading model from local path: {local_model_path}")
                self.model = load_object(local_model_path)
            else:
                # Try loading from S3
                logging.info("Loading model from S3")
                self.model = Proj1Estimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path,
                )
                self.model = self.model.load_model()

            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise MyException(e, sys)

    def predict(self, dataframe) -> list:
        """
        This is the method of StudentPerformanceClassifier
        Returns: Prediction as list [0] for Fail, [1] for Pass
        """
        try:
            logging.info("Entered predict method of StudentPerformanceClassifier class")

            # Make prediction
            predictions = self.model.predict(dataframe)

            # Convert to list if single value
            if hasattr(predictions, 'tolist'):
                predictions = predictions.tolist()

            logging.info(f"Predictions: {predictions}")
            return predictions

        except Exception as e:
            raise MyException(e, sys)

    def predict_proba(self, dataframe) -> list:
        """
        This method returns prediction probabilities
        Returns: List of [prob_fail, prob_pass]
        """
        try:
            logging.info("Entered predict_proba method of StudentPerformanceClassifier class")

            # Get the trained model from the MyModel wrapper
            if hasattr(self.model, 'trained_model_object'):
                # Get probabilities from the underlying model
                transformed_features = self.model.preprocessing_object.transform(dataframe)
                probabilities = self.model.trained_model_object.predict_proba(transformed_features)

                # Return as list
                if hasattr(probabilities, 'tolist'):
                    probabilities = probabilities.tolist()

                return probabilities

            return [[0.5, 0.5]]  # Default if can't get probabilities

        except Exception as e:
            logging.warning(f"Could not get probabilities: {e}")
            return [[0.5, 0.5]]  # Default on error