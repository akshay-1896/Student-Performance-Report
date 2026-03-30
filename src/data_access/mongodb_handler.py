"""
MongoDB Handler for Student Performance Predictor
===================================================
Manages prediction history and student data storage in MongoDB.
"""

import os
import sys
from datetime import datetime
import pymongo
import certifi
from typing import List, Dict, Any, Optional
from bson import ObjectId

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY


class MongoDBHandler:
    """Handles all MongoDB operations for the application."""

    ca = certifi.where()
    client = None

    def __init__(self, database_name: str = DATABASE_NAME):
        """
        Initialize MongoDB connection.

        Args:
            database_name: Name of the MongoDB database
        """
        try:
            if MongoDBHandler.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    logging.warning("MongoDB URL not set. Using local file storage.")
                    self.client = None
                else:
                    MongoDBHandler.client = pymongo.MongoClient(
                        mongo_db_url,
                        tlsCAFile=self.ca
                    )

            self.database_name = database_name
            if self.client:
                self.database = self.client[database_name]
                self.predictions_collection = self.database["predictions"]
                self.students_collection = self.database["students"]
                logging.info(f"MongoDB connected to database: {database_name}")
            else:
                self.database = None

        except Exception as e:
            logging.warning(f"MongoDB connection failed: {e}. Using local storage.")
            self.client = None
            self.database = None

    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Save a prediction to MongoDB.

        Args:
            prediction_data: Dictionary containing prediction details

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.client is None:
                return False

            # Add timestamp
            prediction_data["timestamp"] = datetime.now()

            # Insert into predictions collection
            result = self.predictions_collection.insert_one(prediction_data)
            logging.info(f"Prediction saved to MongoDB: {result.inserted_id}")
            return True

        except Exception as e:
            logging.error(f"Error saving prediction: {e}")
            return False

    def get_prediction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get prediction history from MongoDB.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of prediction records
        """
        try:
            if self.client is None:
                return []

            # Get most recent predictions
            predictions = list(self.predictions_collection.find()
                             .sort("timestamp", -1)
                             .limit(limit))

            # Convert ObjectId and datetime to string for JSON serialization
            for pred in predictions:
                if "_id" in pred:
                    pred["_id"] = str(pred["_id"])
                # Convert datetime to string
                if "timestamp" in pred and isinstance(pred["timestamp"], datetime):
                    pred["timestamp"] = pred["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

            return predictions

        except Exception as e:
            logging.error(f"Error fetching prediction history: {e}")
            return []

    def save_student_data(self, student_data: Dict[str, Any]) -> bool:
        """
        Save student data to MongoDB.

        Args:
            student_data: Dictionary containing student details

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.client is None:
                return False

            result = self.students_collection.insert_one(student_data)
            logging.info(f"Student data saved to MongoDB: {result.inserted_id}")
            return True

        except Exception as e:
            logging.error(f"Error saving student data: {e}")
            return False

    def get_all_students(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all student records from MongoDB.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of student records
        """
        try:
            if self.client is None:
                return []

            students = list(self.students_collection.find().limit(limit))

            for student in students:
                if "_id" in student:
                    student["_id"] = str(student["_id"])

            return students

        except Exception as e:
            logging.error(f"Error fetching students: {e}")
            return []

    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        return self.client is not None


# Singleton instance
_mongodb_handler = None


def get_mongodb_handler() -> MongoDBHandler:
    """Get or create MongoDB handler singleton."""
    global _mongodb_handler
    if _mongodb_handler is None:
        _mongodb_handler = MongoDBHandler()
    return _mongodb_handler