from src.churn_etl.entity import DataExtracting
from src.churn_etl.logging.coustom_log import logger
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2 import OperationalError
from sqlalchemy import create_engine

class DataExtractor:
    def __init__(self, config: DataExtracting):
        self.config = config
    
    def Extract_from_postgresSQL(self):
        try:
            # Create a connection to the PostgreSQL database
            engine = create_engine(f'postgresql+psycopg2://{str(self.config.user)}:{str(self.config.password)}@{str(self.config.host)}:{str(self.config.port)}/{str(self.config.database)}')
            # Write the SQL query to fetch data
            query = f"SELECT * FROM {str(self.config.table_name)};"
            # Execute the query and load data into a DataFrame
            raw_data = pd.read_sql(query, engine)
            raw_data.to_csv(self.config.raw_data, index=False)
            
            logger.info(f"data succesfully fetched from {self.config.database} and table {self.config.table_name}")

            return 
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    
    
