from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataExtracting:
    root_dir: Path
    csv_data: Path
    host: str
    port: int
    database: str
    table_name: str
    user: str
    password: str

@dataclass
class DataExtractingArtifact:
    data_loading_status: bool
    loaded_data_frame: pd.DataFrame

@dataclass
class DataTransformation:
    root_dir: Path
    validation_schema: dict

@dataclass
class DataTransformationArtifact:
    validation_status: bool
    transformation_status: bool
    transformed_data_frame: pd.DataFrame

@dataclass
class DataLoading:
    root_dir: Path
    etl_output: Path
    