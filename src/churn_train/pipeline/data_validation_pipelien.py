from src.churn_train.config.configuration import TrainingConfigurationManager
from src.churn_train.components.data_validation import DataValidator
from src.churn_train.logging.coustom_log import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        try:
            # Load configuration
            config_manager = TrainingConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            data_validator = DataValidator(config= data_validation_config)
            data_validator.validate_schema()
            logger.info("Data Validation completed successfully")
        except Exception as e:
            logger.error(f"Error in initiating data Validaton pipeline: {str(e)}")
            raise