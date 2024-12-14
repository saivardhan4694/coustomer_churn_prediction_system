from src.churn_train.config.configuration import TrainingConfigurationManager
from src.churn_train.components.data_ingestion import DataIngestor
from src.churn_train.logging.coustom_log import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            # Load configuration
            config_manager = TrainingConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestor = DataIngestor(config= data_ingestion_config)
            data_ingestor.ingest()
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error in initiating data ingestion pipeline: {str(e)}")
            raise