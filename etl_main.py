from src.churn_etl.logging.coustom_log import logger
from src.churn_etl.pipeline.data_extraction_pipeline import DataExtractingPipeline
from src.churn_etl.pipeline.data_transformation_pipeline import DataTransformationPipeline

STAGE_NAME = "Data Extraction Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_extraction = DataExtractingPipeline()
    dataextractionartifact = data_extraction.initiate_data_extraction()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_transformation = DataTransformationPipeline(artifact= dataextractionartifact)
    data_trasformation_artifact = data_transformation.initiate_data_Transformation()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\n=================x")
except Exception as e:
    logger.exception(e)
    raise e

