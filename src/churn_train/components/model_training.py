import pandas as pd
from src.churn_train.logging.coustom_log import logger
from src.churn_train.entity import ModelTraining
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import pickle
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, config: ModelTraining):
        self.config = config
        dagshub.init(repo_owner='saivardhan4694', repo_name='coustomer_churn_prediction_system', mlflow=True)
        self.auto_register = False
        self.retrain_all_models = False
        
    
    def load_latest_version_of_model(self, model_name):
        try:
            client = MlflowClient()

            # Fetch the latest version of the model
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])

            # get the latest version of the mdoel
            latest_version = latest_versions[0].version

            # Load the model URI for the latest version
            model_uri = f"models:/{model_name}/{latest_version}"
            
            # Load the model
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"model {model_name} loaded succesfully.")
            return model
        
        except Exception as e:
            logger.error(f"Error in getting latest version of model: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame):

        # 1. One-hot encode categorical variables
        data = pd.get_dummies(data, columns=['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus'], drop_first=True)

        # 2. Split features and target
        features = data.drop(columns=['CustomerID', 'Churn']) 
        target = data['Churn']

        # 3. Scale numerical features
        scaler = StandardScaler()
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

        # 4. Apply SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        features_resampled, target_resampled = smote.fit_resample(features, target)

        # 5. Split the resampled dataset into train and test sets
        features_train, features_test, target_train, target_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)

        return features_train, features_test, target_train, target_test
    
    def model_trainer(self, model_name, experiment_name,
                      features_train, features_test, target_train, target_test,
                      ):
            
            model = self.load_latest_version_of_model(model_name= model_name)

            logger.info(f"initiating {model_name} training")
            mlflow.set_experiment(experiment_name=experiment_name)

            with mlflow.start_run():
                model.fit(features_train,target_train)

                predictions = model.predict(features_test)
                prediction_probabiliteis = model.predict_proba(features_test)[:, 1]

                # Metrics Calculation for the best model
                accuracy = accuracy_score(target_test, predictions)
                precision = precision_score(target_test, predictions)
                recall = recall_score(target_test, predictions)
                f1 = f1_score(target_test, predictions)
                report = classification_report(target_test, predictions, output_dict=True)
                conf_matrix = confusion_matrix(target_test, predictions)

                # Log all metrics for the best model
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log the classification report as an artifact
                mlflow.log_dict(report, "classification_report_best_model.json")

                # Log the classification report as an image
                plt.figure(figsize=(8, 6))
                plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.xticks(np.arange(conf_matrix.shape[1]))
                plt.yticks(np.arange(conf_matrix.shape[0]))

                # Annotate the confusion matrix with the counts
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        plt.text(j, i, f'{conf_matrix[i, j]}', ha='center', va='center', color='black')

                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
                plt.close()

                # Log ROC curve
                fpr, tpr, _ = roc_curve(target_test, prediction_probabiliteis)
                roc_auc = auc(fpr, tpr)

                # Log ROC curve as an artifact
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                mlflow.log_figure(plt.gcf(), "roc_curve.png")
                plt.close()
                

                # Define the input example for the model
                input_example = features_train.iloc[0].to_dict()

                # Log the best model along with input example to avoid the warning
                mlflow.sklearn.log_model(model, "latest_knn_model", input_example=input_example)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{model_name}s",
                    input_example=input_example,
                )
            
                # End the MLflow run
                mlflow.end_run()

                logger.info(f"{model_name} training completed. metrics and artifacts are loaded to MLflow experiment {experiment_name}")

    def train_ensemble_model(self, features_train, features_test, target_train, target_test):

        #Ensemble_Model_Experiment: churn_ensemble_model
        ensemble_model = self.load_latest_version_of_model(model_name="churn_ensemble_model")

        mlflow.set_experiment(experiment_name="Ensemble_Model_Experiment")

        with mlflow.start_run():
            # Extract the individual models from the ensemble
            individual_models = {name: estimator for name, estimator in ensemble_model.named_estimators_.items()}

            # Retrain each individual model with the new data
            for name, model in individual_models.items():
                print(f"Retraining model: {name}")
                model.fit(features_train, target_train)

            # Create a new VotingClassifier with the retrained models
            retrained_ensemble_model = VotingClassifier(estimators=[
                (name, model) for name, model in individual_models.items()
            ], voting='hard')

            # Train the ensemble model itself
            retrained_ensemble_model.fit(features_train, target_train)

            # Evaluate the ensemble model
            y_pred_ensemble = retrained_ensemble_model.predict(features_test)

            # Calculate metrics for the ensemble model
            ensemble_accuracy = accuracy_score(target_test, y_pred_ensemble)
            ensemble_precision = precision_score(target_test, y_pred_ensemble)
            ensemble_recall = recall_score(target_test, y_pred_ensemble)
            ensemble_f1 = f1_score(target_test, y_pred_ensemble)
            report = classification_report(target_test, y_pred_ensemble, output_dict=True)
            conf_matrix = confusion_matrix(target_test, y_pred_ensemble)

            # Log metrics for the ensemble model
            mlflow.log_metric("accuracy", ensemble_accuracy)
            mlflow.log_metric("precision", ensemble_precision)
            mlflow.log_metric("recall", ensemble_recall)
            mlflow.log_metric("f1_score", ensemble_f1)

            # Log the classification report as an artifact
            mlflow.log_dict(report, "classification_report_best_model.json")

            # Log the classification report as an image
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(np.arange(conf_matrix.shape[1]))
            plt.yticks(np.arange(conf_matrix.shape[0]))

            # Annotate the confusion matrix with the counts
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, f'{conf_matrix[i, j]}', ha='center', va='center', color='black')

            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()

            # Define the input example for the model
            input_example = features_train.iloc[0].to_dict()

            if self.auto_register:
                mlflow.sklearn.log_model(
                    sk_model=retrained_ensemble_model,
                    artifact_path="ensemble_models",
                    input_example= input_example,
                    registered_model_name="churn_ensemble_model"
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=retrained_ensemble_model,
                    artifact_path="ensemble_models",
                    registered_model_name="churn_ensemble_model"
                )

            with open(self.config.training_output, 'wb') as f:
                pickle.dump(model, f)

            mlflow.end_run()
            logger.info("Retrained ensemble model logged and registered successfully.")


    def prompt_user_for_settings(self):
        user_choice = input("would you like to retrain all the individual models along with ensemble model? (y/n): ").strip().lower()
        user_choice2 = input("would like to auto register the esemble model? (y/n): ").strip().lower()
        if user_choice == 'y' and user_choice2 == "y":
            self.retrain_all_models = True
            self.auto_register = True
            return
        elif user_choice == 'y' and user_choice2 == "n":
            self.retrain_all_models = True
            return
        elif user_choice == 'n' and user_choice2 == "y":
            self.auto_register = True
        else:
            self.prompt_user_for_settings()


    def train_models(self):
        # load the data form csv
        data = pd.read_csv(self.config.training_input)

        # preprocess the data
        features_train, features_test, target_train, target_test = self.preprocess_data(data)
        
        self.prompt_user_for_settings()

        if self.retrain_all_models:
            # train all the idividual models 
            for experiment, model in self.config.experiments.items():
                self.model_trainer(
                                model_name=str(model),
                                experiment_name=str(experiment),
                                features_train=features_train,
                                features_test=features_test,
                                target_train=target_train,
                                target_test=target_test)
                
            # train the ensemble model
            self.train_ensemble_model(
                                features_train=features_train,
                                features_test=features_test,
                                target_train=target_train,
                                target_test=target_test
            )
        else:
            # train the ensemble model
            self.train_ensemble_model(
                                features_train=features_train,
                                features_test=features_test,
                                target_train=target_train,
                                target_test=target_test
            )