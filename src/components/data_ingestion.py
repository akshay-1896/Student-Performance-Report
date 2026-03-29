import os
import sys
import pandas as pd
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.student_data import StudentData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file

        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            my_data = StudentData()
            dataframe = my_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def export_data_from_csv(self, file_path: str = None) -> pd.DataFrame:
        """
        Method Name :   export_data_from_csv
        Description :   This method exports data from csv file

        Output      :   data is returned as DataFrame
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if file_path is None:
                file_path = self.data_ingestion_config.feature_store_file_path

            logging.info(f"Reading data from CSV: {file_path}")

            # Try direct path first
            if os.path.exists(file_path):
                dataframe = pd.read_csv(file_path)
            else:
                # Try relative to project root
                base_path = os.path.join(os.getcwd(), file_path)
                if os.path.exists(base_path):
                    dataframe = pd.read_csv(base_path)
                else:
                    # Try data folder
                    data_path = os.path.join(os.getcwd(), 'data', os.path.basename(file_path))
                    if os.path.exists(data_path):
                        dataframe = pd.read_csv(data_path)
                    else:
                        raise Exception(f"Data file not found: {file_path}")

            logging.info(f"Shape of dataframe: {dataframe.shape}")
            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio

        Output      :   Folder is created
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline

        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            # Try to get data from MongoDB first, fallback to CSV
            try:
                dataframe = self.export_data_into_feature_store()
            except Exception as mongo_err:
                logging.warning(f"MongoDB fetch failed: {mongo_err}. Trying CSV...")
                dataframe = self.export_data_from_csv()

            logging.info("Got the data from source")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                           test_file_path=self.data_ingestion_config.testing_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)