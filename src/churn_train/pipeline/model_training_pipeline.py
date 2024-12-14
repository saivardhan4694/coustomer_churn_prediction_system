from src.churn_train.config.configuration import TrainingConfigurationManager
from src.churn_train.components.model_training import ModelTrainer
from src.churn_train.logging.coustom_log import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            # Load configuration
            config_manager = TrainingConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config= model_trainer_config)
            model_trainer.train_models()
            logger.info("model training completed successfully")
        except Exception as e:
            logger.error(f"Error in initiating model training pipeline: {str(e)}")
            raise