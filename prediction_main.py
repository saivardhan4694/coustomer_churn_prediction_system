
from mlflow.tracking import MlflowClient
from src.churn_prediction.config.configuration import PredictionConfigurationManager
from src.churn_prediction.components.validator import DataValidator
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class Predictor:
    def __init__(self):
        self.configuration_manager = PredictionConfigurationManager()
        

    def load_model(self):
        try:
            model_path = Path(__file__).resolve().parent / "models" / "esemble_churn_model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        
        except Exception as e:
            raise
    
    def validate_data(self, dataframe: pd.DataFrame):
        prediction_config = self.configuration_manager.get_data_validation_config()
        validator = DataValidator(config=prediction_config)
        status = validator.validate_schema(df=dataframe)
        return status
    
    def preprocess_data(self, data: pd.DataFrame):
        preprocess_config = self.configuration_manager.get_model_inference_config()
        print("preprocessing for predictions")
        # 1. One-hot encode categorical variables
        categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # 2. Add missing columns with default value 0 to match training columns
        for col in preprocess_config.columns:
            if col not in data.columns:
                data[col] = 0
        
        features = data[preprocess_config.columns]

        # 3. Scale numerical features
        scaler = StandardScaler()
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        print("preprocessing done.")
        return features
            
    def predict(self, data: pd.DataFrame):
        model = self.load_model()
        validation_status = self.validate_data(dataframe=data)
        print(validation_status)
        if validation_status:
            original_data = data.copy()
            
            # Preprocess the data
            data = self.preprocess_data(data)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Attach predictions to the original DataFrame
            original_data['Predictions'] = predictions
            
            return original_data
        else:
            return None

