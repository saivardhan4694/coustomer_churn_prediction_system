from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestion:
    root_dir: Path
    data_input: Path
    data_output: Path

@dataclass
class DataValidation:
    root_dir: Path
    validation_input: Path
    validation_schema: dict
    validation_output: Path

@dataclass
class ModelTraining:
    training_input: Path
    experiments: dict
    training_output: Path