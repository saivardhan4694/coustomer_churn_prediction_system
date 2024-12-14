from src.churn_train.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.churn_train.pipeline.data_validation_pipelien import DataValidationTrainingPipeline
from src.churn_train.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.churn_train.logging.coustom_log import logger

stage_name = "Data ingestion Stage"
try:
    logger.info(f">>>>>>> stage {stage_name} started <<<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {stage_name} started <<<<<<<<")
    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.initiate_data_validation()
    logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Model training Stage"
try:
    logger.info(f">>>>>>> stage {stage_name} started <<<<<<<<")
    model_trianing_pipeline = ModelTrainingPipeline()
    model_trianing_pipeline.initiate_model_training()
    logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e


