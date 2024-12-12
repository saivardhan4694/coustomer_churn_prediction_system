from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataExtracting:
    root_dir: Path
    csv_data: Path
    raw_data: Path
    host: str
    port: int
    database: str
    table_name: str
    user: str
    password: str

@dataclass
class DataTransformation:
    root_dir: Path
    input_csv: Path
    transformation_output_csv: Path
    transformation_status_file: Path
    validation_schema: dict

@dataclass
class DataLoading:
    root_dir: Path
    input_csv: Path
    transformation_status_outut: Path
    etl_output: Path
    