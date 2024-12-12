from src.churn_etl.entity import DataLoading, DataTransformationArtifact
from src.churn_etl.logging.coustom_log import logger
from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, config: DataLoading, artifact: DataTransformationArtifact):
        self.config = config
        self.artifact = artifact

    def load_data(self):
        try:
            if self.artifact.transformation_status == True:
                data_frame = self.artifact.transformed_data_frame
                data_frame.to_csv(self.config.etl_output)
                logger.info(f"data succesfully loaded to {self.config.etl_output}")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            
            