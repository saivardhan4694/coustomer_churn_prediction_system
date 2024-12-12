from src.churn_etl.config.configuration import ETLConfigurationManager
from src.churn_etl.components.data_loader import DataLoader
from src.churn_etl.logging.coustom_log import logger

class DataLoadingPipeline:
    def __init__(self):
        pass

    def initiate_data_Loading(self):
        try:
            config = ETLConfigurationManager()
            data_loading_config = config.get_data_laoding_config()
            data_loader= DataLoader(config=data_loading_config)
            data_loader.load_data()
        except Exception as e:
            raise e