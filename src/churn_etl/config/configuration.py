from src.churn_etl.entity import DataExtracting, DataTransformation, DataLoading
from src.churn_etl.utils.common import create_directories
from src.churn_etl.constants.__init import *
from src.churn_etl.utils.common import read_yaml, create_directories

class ETLConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH) -> None:
        
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_extraction_config(self):

        config = self.config.data_Extracting
        create_directories([config.root_dir])

        data_extracting_config = DataExtracting(

            root_dir= config.root_dir,
            csv_data= config.csv_data,
            raw_data= config.raw_data,
            host=config.postgresSQL.host,
            port=config.postgresSQL.port,
            database=config.postgresSQL.database,
            table_name=config.postgresSQL.table_name,
            user=config.postgresSQL.user,
            password=config.postgresSQL.password

        )

        return data_extracting_config
    
    def get_data_transformation_config(self):

        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation = DataTransformation(
            root_dir=config.root_dir,
            input_csv= config.input_csv,
            transformation_output_csv= config.transformation_output_csv,
            transformation_status_file=config.transformation_status_file,
        validation_schema = {
            "CustomerID": str(config.validation_schema.CustomerID),
            "Churn": str(config.validation_schema.Churn),
            "Tenure": str(config.validation_schema.Tenure),
            "PreferredLoginDevice": str(config.validation_schema.PreferredLoginDevice),
            "CityTier": str(config.validation_schema.CityTier),
            "WarehouseToHome": str(config.validation_schema.WarehouseToHome),
            "PreferredPaymentMode": str(config.validation_schema.PreferredPaymentMode),
            "Gender": str(config.validation_schema.Gender),
            "HourSpendOnApp": str(config.validation_schema.HourSpendOnApp),
            "NumberOfDeviceRegistered": str(config.validation_schema.NumberOfDeviceRegistered),
            "PreferedOrderCat": str(config.validation_schema.PreferedOrderCat),
            "SatisfactionScore": str(config.validation_schema.SatisfactionScore),
            "MaritalStatus": str(config.validation_schema.MaritalStatus),
            "NumberOfAddress": str(config.validation_schema.NumberOfAddress),
            "Complain": str(config.validation_schema.Complain),
            "OrderAmountHikeFromlastYear": str(config.validation_schema.OrderAmountHikeFromlastYear),
            "CouponUsed": str(config.validation_schema.CouponUsed),
            "OrderCount": str(config.validation_schema.OrderCount),
            "DaySinceLastOrder": str(config.validation_schema.DaySinceLastOrder),
            "CashbackAmount": str(config.validation_schema.CashbackAmount)
        }
        )
    
        return data_transformation
    
    def get_data_laoding_config(self):
        
        config = self.config.data_loading
        create_directories([config.root_dir])
        
        data_loading_config = DataLoading(
            root_dir=config.root_dir,
            input_csv= config.input_csv,
            transformation_status_outut= config.transformation_status_outut,
            etl_output= config.etl_output
        )

        return data_loading_config