from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataValidation:
    root_dir: Path
    validation_schema: dict
    validation_output: Path

@dataclass
class ModelInference:
    columns: list