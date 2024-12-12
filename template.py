import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s: ")

etl_src = "churn_etl"
training_src = "churn_train"
prediction_src = "churn_prediction"

list_of_files = {
    f"src/{etl_src}/__init__.py",
    f"src/{etl_src}/config/__init__.py",
    f"src/{etl_src}/config/configuration.py",
    f"src/{etl_src}/components/ __init__.py",
    f"src/{etl_src}/utils/__init__.py",
    f"src/{etl_src}/utils/common.py",
    f"src/{etl_src}/logging/__init__.py",
    f"src/{etl_src}/pipeline/__init__.py",
    f"src/{etl_src}/entity/__init__.py",
    f"src/{etl_src}/constants/__init.py",
    f"src/{training_src}/__init__.py",
    f"src/{training_src}/config/__init__.py",
    f"src/{training_src}/config/configuration.py",
    f"src/{training_src}/components/ __init__.py",
    f"src/{training_src}/utils/__init__.py",
    f"src/{training_src}/utils/common.py",
    f"src/{training_src}/logging/__init__.py",
    f"src/{training_src}/pipeline/__init__.py",
    f"src/{training_src}/entity/__init__.py",
    f"src/{training_src}/constants/__init.py",
    f"src/{prediction_src}/__init__.py",
    f"src/{prediction_src}/config/__init__.py",
    f"src/{prediction_src}/config/configuration.py",
    f"src/{prediction_src}/components/ __init__.py",
    f"src/{prediction_src}/utils/__init__.py",
    f"src/{prediction_src}/utils/common.py",
    f"src/{prediction_src}/logging/__init__.py",
    f"src/{prediction_src}/pipeline/__init__.py",
    f"src/{prediction_src}/entity/__init__.py",
    f"src/{prediction_src}/constants/__init.py",
    "config/etl_config.yaml",
    "config/training_config.yaml",
    "config/prediction_config.yaml",
    "params.yaml",
    "app.py",
    "etl_main.py",
    "training_main.py",
    "prediction_main.py",
    "Dockerfile",
    "requirements.txt",
    "research/trails.ipynb"
}

for file_path in list_of_files:

    
    file_path = Path(file_path)
    filedir, filename = os.path.split(file_path)

    # Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"creating directory {filedir} for the file {filename}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"creating empty file: {file_path}")

    else:
        logging.info(f"file {file_path} already exists")

