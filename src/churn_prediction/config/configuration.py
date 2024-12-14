from src.churn_prediction.entity import (DataValidation, ModelInference)
from pathlib import Path
from src.churn_etl.utils.common import create_directories
from src.churn_train.constants.__init import *
from src.churn_etl.utils.common import read_yaml, create_directories

class PredictionConfigurationManager:
    def __init__(self) -> None:
        self.config = read_yaml(Path(r"D:\repositories\coustomer_churn_prediction_system\config\prediction_config.yaml"))
        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self):
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidation(
            root_dir= config.root_dir,
            validation_schema=config.validation_schema,
            validation_output= config.validation_output
        )

        return data_validation_config
    
    def get_model_inference_config(self):
        config = self.config.model_inference

        model_inference_config = ModelInference(
            columns= config.columns,
        )

        return model_inference_config