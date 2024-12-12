from src.churn_etl.config.configuration import ETLConfigurationManager
from src.churn_etl.components.data_transformation import DataTrasformer
from src.churn_etl.logging.coustom_log import logger
from src.churn_etl.entity import DataExtractingArtifact

class DataTransformationPipeline:
    def __init__(self, artifact: DataExtractingArtifact):
        self.artifact = artifact

    def initiate_data_Transformation(self):
        try:
            config = ETLConfigurationManager()
            data_Transformation_config = config.get_data_transformation_config()
            data_Transformation= DataTrasformer(config=data_Transformation_config, artifact=self.artifact)
            data_trasformation_artifact = data_Transformation.transform_data()
            return data_trasformation_artifact
        except Exception as e:
            raise e