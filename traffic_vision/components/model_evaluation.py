import os
import sys
import yaml
from ultralytics import YOLO
from traffic_vision.logger import logging
from traffic_vision.exception import AppException
from traffic_vision.entity.config_entity import ModelEvaluationConfig
from traffic_vision.entity.artifacts_entity import ModelTrainerArtifact, ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)
    
    def evaluate_model(self) -> dict:
        """
        Evaluate the trained model on validation data
        """
        try:
            logging.info("Starting model evaluation")
            
            # Load the trained model
            model = YOLO(self.model_trainer_artifact.trained_model_file_path)
            
            # Get the path to data.yaml from model trainer
            data_yaml_path = os.path.join("artifacts", "model_trainer", "data.yaml")
            
            if not os.path.exists(data_yaml_path):
                logging.warning(f"data.yaml not found at {data_yaml_path}, trying to find it in feature store")
                # Try to find it in the feature store
                feature_store_yaml_path = os.path.join("artifacts", "data_ingestion", "feature_store", "data.yaml")
                if os.path.exists(feature_store_yaml_path):
                    logging.info(f"Using data.yaml from feature store: {feature_store_yaml_path}")
                    data_yaml_path = feature_store_yaml_path
                else:
                    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path} or {feature_store_yaml_path}")
            
            logging.info(f"Using data.yaml from: {data_yaml_path}")
            
            # Run validation on the validation dataset
            results = model.val(
                data=data_yaml_path,
                conf=self.model_evaluation_config.threshold,
                iou=self.model_evaluation_config.iou_threshold,
                verbose=True
            )
            
            # Extract metrics using the correct attributes
            # The DetMetrics object has a results_dict method that returns all metrics
            metrics_dict = results.results_dict
            
            # Log available keys for debugging
            logging.info(f"Available metrics keys: {list(metrics_dict.keys())}")
            
            # Extract the metrics we need
            metrics = {
                "mAP50": results.maps[0],  # mAP at IoU 0.5
                "mAP50-95": results.maps[1],  # mAP at IoU 0.5-0.95
                "fitness": results.fitness  # Overall fitness score
            }
            
            # Add additional metrics if available
            if 'metrics/precision(B)' in metrics_dict:
                metrics["precision"] = metrics_dict['metrics/precision(B)']
            if 'metrics/recall(B)' in metrics_dict:
                metrics["recall"] = metrics_dict['metrics/recall(B)']
            
            # Add mean results for box metrics
            box_mean_results = results.box.mean_results()
            if len(box_mean_results) >= 3:
                metrics["precision"] = box_mean_results[0]
                metrics["recall"] = box_mean_results[1]
            
            logging.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise AppException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation")
            
            # Create evaluation directory
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Save metrics to file
            metrics_file_path = os.path.join(
                self.model_evaluation_config.model_evaluation_dir,
                "metrics.yaml"
            )
            
            with open(metrics_file_path, 'w') as f:
                yaml.dump(metrics, f)
            
            # Determine if model meets acceptance criteria
            is_model_accepted = metrics.get("mAP50", 0) >= self.model_evaluation_config.min_map_threshold
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                model_metrics=metrics,
                metrics_file_path=metrics_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise AppException(e, sys) 