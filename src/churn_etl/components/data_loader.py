from src.churn_etl.entity import DataLoading
from src.churn_etl.logging.coustom_log import logger
from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, config: DataLoading):
        self.config = config

    def load_data(self):
        try:
            with open(self.config.transformation_status_outut, "r") as file:
                transformation_status = file.read().strip().lower() == "true"
            if transformation_status:
                data_frame = pd.read_csv(self.config.input_csv)
                data_frame.to_csv(self.config.etl_output, index=False)
                logger.info(f"data succesfully loaded to {self.config.etl_output}")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            
            