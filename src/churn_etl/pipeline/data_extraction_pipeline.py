from src.churn_etl.config.configuration import ETLConfigurationManager
from src.churn_etl.components.data_extracting import DataExtractor
from src.churn_etl.logging.coustom_log import logger

class DataExtractingPipeline:
    def __init__(self):
        pass

    def initiate_data_extraction(self):
        try:
            config = ETLConfigurationManager()
            data_extraction_config = config.get_data_extraction_config()
            data_extractor= DataExtractor(config=data_extraction_config)
            data_extractor.Extract_from_postgresSQL()
        except Exception as e:
            raise e