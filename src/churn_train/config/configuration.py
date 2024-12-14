from src.churn_train.entity import (DataIngestion, 
                                    DataValidation,
                                    ModelTraining)
from src.churn_etl.utils.common import create_directories
from src.churn_train.constants.__init import *
from src.churn_etl.utils.common import read_yaml, create_directories

class TrainingConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH) -> None:
        
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestion(
            root_dir= config.root_dir,
            data_input= config.data_input,
            data_output= config.data_output
        )

        return data_ingestion_config
    
    def get_data_validation_config(self):
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidation(
            root_dir= config.root_dir,
            validation_input= config.validation_input, 
            validation_schema=config.validation_schema,
            validation_output= config.validation_output
        )

        return data_validation_config
    
    def get_model_trainer_config(self):
        config = self.config.Model_training
        
        model_tarining_config = ModelTraining(
            training_input= config.training_input,
            experiments= config.experiments,
        )

        return model_tarining_config
        
