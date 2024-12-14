import sys
from src.churn_etl.logging.coustom_log import logger
from src.churn_etl.pipeline.data_extraction_pipeline import DataExtractingPipeline
from src.churn_etl.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.churn_etl.pipeline.data_loading_pipeline import DataLoadingPipeline

def run_stage(stage_name):
    try:
        logger.info(f">>>>>>> stage {stage_name} started <<<<<<<<")
        
        if stage_name == "extract":
            data_extraction = DataExtractingPipeline()
            data_extraction.initiate_data_extraction()
        elif stage_name == "transform":
            data_transformation = DataTransformationPipeline()
            data_transformation.initiate_data_Transformation()
        elif stage_name == "load":
            data_loading = DataLoadingPipeline()
            data_loading.initiate_data_Loading()
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<<\n\n=================x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--stage":
        print("Usage: python etl_main.py --stage [extract|transform|load]")
        sys.exit(1)
    
    stage_name = sys.argv[2]
    run_stage(stage_name)
