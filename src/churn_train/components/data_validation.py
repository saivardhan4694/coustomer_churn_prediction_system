import pandas as pd
from src.churn_train.logging.coustom_log import logger
from src.churn_train.entity import DataValidation

class DataValidator:
    def __init__(self, config: DataValidation):
        self.config = config
        self.validation_status = True
        self.validation_errors = []

    def validate_columns(self, df: pd.DataFrame):
        """Check for missing or extra columns."""
        expected_columns = set(self.config.validation_schema.keys())
        actual_columns = set(df.columns)

        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            self.validation_errors.append(f"Missing columns: {missing_columns}")
            self.validation_status = False

        if extra_columns:
            logger.warning(f"Extra columns: {extra_columns}")
            self.validation_errors.append(f"Extra columns: {extra_columns}")

    def validate_data_types(self, df: pd.DataFrame):
        """Check if the data types of the columns match the expected types."""
        for col, specs in self.config.validation_schema.items():
            expected_dtype = specs['dtype']
            if col in df.columns and df[col].dtype != expected_dtype:
                logger.error(f"Column '{col}' has incorrect data type. Expected {expected_dtype}, got {df[col].dtype}")
                self.validation_errors.append(f"Column '{col}' has incorrect data type. Expected {expected_dtype}, got {df[col].dtype}")
                self.validation_status = False

    def validate_constraints(self, df: pd.DataFrame):
        """Check for constraints like range checks and allowed values."""
        for col, specs in self.config.validation_schema.items():
            if col not in df.columns:
                continue

            constraints = specs.get('constraints', {})

            # Check for minimum value constraint
            if 'min' in constraints:
                if (df[col] < constraints['min']).any():
                    logger.error(f"Column '{col}' has values below the minimum of {constraints['min']}")
                    self.validation_errors.append(f"Column '{col}' has values below the minimum of {constraints['min']}")
                    self.validation_status = False

            # Check for maximum value constraint
            if 'max' in constraints:
                if (df[col] > constraints['max']).any():
                    logger.error(f"Column '{col}' has values above the maximum of {constraints['max']}")
                    self.validation_errors.append(f"Column '{col}' has values above the maximum of {constraints['max']}")
                    self.validation_status = False

            # Check for allowed values constraint
            if 'allowed_values' in constraints:
                invalid_values = df[~df[col].isin(constraints['allowed_values'])]
                if not invalid_values.empty:
                    logger.error(f"Column '{col}' contains invalid values: {invalid_values[col].unique()}")
                    self.validation_errors.append(f"Column '{col}' contains invalid values: {invalid_values[col].unique()}")
                    self.validation_status = False

    def validate_schema(self):
        """Orchestrate all validation checks."""

        df = pd.read_csv(self.config.validation_input)
        try:
            self.validate_columns(df)
            self.validate_data_types(df)
            self.validate_constraints(df)

            # Log final validation status
            if self.validation_status:
                with open(self.config.validation_output, "w") as file:
                    file.write(str(self.validation_status))
                logger.info("Schema Validation Passed: Column names, data types, and constraints match expectations.")
            else:
                with open(self.config.validation_output, "w") as file:
                    file.write(str(self.validation_status))
                logger.error("Schema Validation Failed. See errors for details.")


        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            self.validation_errors.append(f"Error during schema validation: {e}")
            self.validation_status = False
            with open(self.config.validation_output, "w") as file:
                file.write(str(self.validation_status))
            return self.validation_status
