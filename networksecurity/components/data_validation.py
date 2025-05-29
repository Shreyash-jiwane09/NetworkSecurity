from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod 
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns =[list(col.keys())[0] for col in self._schema_config["columns"]]  # Ensure schema defines columns
            actual_columns = list(dataframe.columns)

            logging.info(f"Expected columns: {expected_columns}")
            logging.info(f"Actual columns: {actual_columns}")

            if set(expected_columns) == set(actual_columns):  # Ensure the columns match
                return True

            missing_columns = set(expected_columns) - set(actual_columns)
            extra_columns = set(actual_columns) - set(expected_columns)

            if missing_columns:
                logging.error(f"Missing columns in dataframe: {missing_columns}")
            if extra_columns:
                logging.error(f"Unexpected extra columns in dataframe: {extra_columns}")

            return False  # Columns do not match
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_numerical_columns(self,dataframe:pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config.get("numerical_columns",[])
            # Validate each specified numerical column
            for col in numerical_columns:
                if col not in dataframe.columns:
                    logging.error(f"Numerical column '{col}' not found in dataframe")
                    return False
                
                if not pd.api.types.is_numeric_dtype(dataframe[col]):
                    logging.error(f"Column '{col}' is not numerical. Detected dtype: {dataframe[col].dtype}")
                    return False
        
            logging.info("All numerical columns validated successfully")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the data
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate number of columns
            status_columns_train = self.validate_number_of_columns(train_dataframe)
            status_columns_test = self.validate_number_of_columns(test_dataframe)

            # Validate numerical columns
            status_numerical_train = self.validate_numerical_columns(train_dataframe)
            status_numerical_test = self.validate_numerical_columns(test_dataframe)

            # Handle errors
            error_message = ""
            if not status_columns_train:
                error_message += "Train dataframe does not contain all columns.\n"
            if not status_columns_test:
                error_message += "Test dataframe does not contain all columns.\n"
            if not status_numerical_train:
                error_message += "Train dataframe does not contain all numerical columns.\n"
            if not status_numerical_test:
                error_message += "Test dataframe does not contain all numerical columns.\n"

            # Stop if validation fails
            if error_message:
                raise ValueError(error_message)

            # Validate column consistency
            columns_valid = status_columns_train and status_columns_test

            # Check for data drift
            drift_status = self.detect_dataset_drift(train_dataframe, test_dataframe)

            # Final validation status
            status = columns_valid and drift_status

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # Create DataValidationArtifact
            return DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
         raise NetworkSecurityException(e, sys)




