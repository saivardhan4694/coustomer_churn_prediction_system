from src.churn_etl.config.configuration import ETLConfigurationManager
from src.churn_etl.components.data_transformation import DataTrasformer
from src.churn_etl.logging.coustom_log import logger

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_Transformation(self):
        try:
            config = ETLConfigurationManager()
            data_Transformation_config = config.get_data_transformation_config()
            data_Transformation= DataTrasformer(config=data_Transformation_config)
            data_Transformation.transform_data()
        except Exception as e:
            raise e