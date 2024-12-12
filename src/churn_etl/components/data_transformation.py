from src.churn_etl.entity import DataTransformation, DataExtractingArtifact, DataTransformationArtifact
from src.churn_etl.logging.coustom_log import logger
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class DataTrasformer:
    def __init__(self, config: DataTransformation, artifact: DataExtractingArtifact):
        self.config = config
        self.artifact = artifact
        self.transformation_status = True

    def validate_schema(self, df: pd.DataFrame):
        try:
            # 1. Check for missing or extra columns
            actual_columns = df.columns
            expected_columns = self.config.validation_schema

            if set(actual_columns) != set(expected_columns):
                logger.info(f"Column mismatch. Expected columns: {expected_columns}, Actual columns: {actual_columns}")
                self.transformation_status = False
                return 

            # 2. Check for data type consistency
            for col, expected_dtype in expected_columns.items():
                if df[col].dtype != expected_dtype:
                    logger.info(f"Column '{col}' has incorrect data type. Expected {expected_dtype}, got {df[col].dtype}")
                    self.transformation_status = False
                    return 

            logger.info("Schema Validation Passed: Column names and data types match expectations.")
            return 
        except Exception as e:
            self.transformation_status = False
            logger.info(f"error validating extracted data {e}")
    
    def handle_missing_values(self, df: pd.DataFrame):
        try:
            # Step 1: Identify numeric and categorical columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Step 2: Impute numeric columns using mean strategy
            numeric_imputer = SimpleImputer(strategy='mean')
            df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
            
            # Step 3: Impute categorical columns using the most frequent value (mode)
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
            
            # Step 4: Random Forest Iterative Imputer for numeric columns
            rf_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), max_iter=5)
            df[numeric_columns] = rf_imputer.fit_transform(df[numeric_columns])
            
            logger.info("misssing values handled succesfully.")
            return df

        except Exception as e:
            self.transformation_status = False
            logger.info(f"Error handling missing values: {e}")
            return df
    
    def handle_duplicates(self, df: pd.DataFrame):
        try:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.info(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
                df = df.drop_duplicates().reset_index(drop=True)
            else:
                logger.info("No duplicate rows found.")
            return df
        except Exception as e:
            self.transformation_status = False
            logger.error(f"Error handling duplicates: {e}")
            return df

    def transform_data(self):
        
        try:
            raw_data_frame = self.artifact.loaded_data_frame
            self.validate_schema(raw_data_frame)

            if self.transformation_status:
                transformed_data = self.handle_duplicates(raw_data_frame)

            if self.transformation_status:
                transformed_data = self.handle_missing_values(transformed_data)

            transformation_artifact = DataTransformationArtifact(
                transformation_status= self.transformation_status,
                validation_status= True,
                transformed_data_frame=transformed_data
            )

            return transformation_artifact
        
        except Exception as e:
            self.transformation_status = False

            transformation_artifact = DataTransformationArtifact(
                transformation_status= self.transformation_status,
                validation_status= True,
                transformed_data_frame=transformed_data
            )
        
            logger.error(f"Error transforming data: {e}")
            return transformation_artifact


    