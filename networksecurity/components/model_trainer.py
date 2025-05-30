import os
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv

import mlflow
import dagshub

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


# Load environment variables from .env file
load_dotenv()


# Initialize dagshub (credentials must be in env vars)
dagshub.init(repo_owner='shrey.jiwane09', repo_name='NetworkSecurity', mlflow=True)

# Set MLflow URI and experiment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("NetworkSecurity")


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, model, metric, run_name, run_type, log_model=False):
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("Run Type", run_type)
            mlflow.log_metrics({
                "f1_score": metric.f1_score,
                "precision": metric.precision_score,
                "recall": metric.recall_score
            })
            if log_model:
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="BestNetworkModel")
                else:
                    mlflow.sklearn.log_model(model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

        best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])
        best_model = models[best_model_name]
        logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

        # Training metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric, run_name="Training Run", run_type="Train", log_model=False)

        # Testing metrics
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric, run_name="Test Run", run_type="Test", log_model=True)

        # Save the final model
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        final_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        
        model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir, exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, final_model)

        # Also save raw model for quick loading
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        # Return artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_path = self.data_transformation_artifact.transformed_train_file_path
            test_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_path)
            test_arr = load_numpy_array_data(test_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
