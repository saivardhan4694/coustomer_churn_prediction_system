from src.churn_etl.logging.coustom_log import logger
import os
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from box.exceptions import BoxValueError
import pandas as pd
import yaml


@ensure_annotations
def create_directories(path_to_derectory: list, verbose = True):
    for path in path_to_derectory:
        os.makedirs(path, exist_ok= True)
        if verbose:
            logger.info(f"Directory {path} created successfully")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yamlfile:
            data = yaml.safe_load(yamlfile)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(data)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def save_as_csv(dataframe: pd.DataFrame, path_to_csv: Path, index: bool = False):
    try:
        dataframe.to_csv(path_to_csv, index=index)
        logger.info(f"csv file: {path_to_csv} saved successfully")
    except Exception as e:
        raise e 