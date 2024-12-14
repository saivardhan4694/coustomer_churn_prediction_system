from src.churn_train.entity import DataIngestion
from src.churn_train.logging.coustom_log import logger
import pandas as pd

class DataIngestor:
    def __init__(self, config: DataIngestion):
        self.config = config

    def ingest(self):
        try:
            # Ingest data from the source
            ingested_data = pd.read_csv(self.config.data_input)

            # save the ingested data as ingestion output
            ingested_data.to_csv(self.config.data_output, index=False)
            logger.info("Ingestion completed. Data loaded to csv")
        except Exception as e:
            logger.error(f"Error occurred during ingestion: {str(e)}")
            raise Exception("Ingestion failed") from e