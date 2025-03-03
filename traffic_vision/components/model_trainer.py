import os, sys
import yaml
import shutil
from ultralytics import YOLO
from traffic_vision.utils.main_utils import read_yaml_file
from traffic_vision.logger import logging
from traffic_vision.exception import AppException
from traffic_vision.entity.config_entity import ModelTrainerConfig
from traffic_vision.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Create model trainer directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            
            # Get the path to data.yaml in the feature store
            feature_store_path = os.path.join("artifacts", "data_ingestion", "feature_store")
            data_yaml_path = os.path.join(feature_store_path, "data.yaml")
            
            logging.info(f"Reading data configuration from {data_yaml_path}")
            
            # Load and update data.yaml with absolute paths
            with open(data_yaml_path, 'r') as stream:
                data_config = yaml.safe_load(stream)
                num_classes = data_config['nc']
                class_names = data_config['names']
            
            # Create a copy of data.yaml with absolute paths for training
            training_yaml_path = os.path.join(self.model_trainer_config.model_trainer_dir, "data.yaml")
            
            # Update paths to be absolute
            data_config['train'] = os.path.abspath(os.path.join(feature_store_path, "train", "images"))
            data_config['val'] = os.path.abspath(os.path.join(feature_store_path, "valid", "images"))
            data_config['test'] = os.path.abspath(os.path.join(feature_store_path, "test", "images"))
            
            # Write the updated config to the training directory
            with open(training_yaml_path, 'w') as f:
                yaml.dump(data_config, f)
            
            logging.info(f"Updated data.yaml created at {training_yaml_path}")
            logging.info(f"Training model for {num_classes} classes: {class_names}")
            
            # Initialize YOLOv8 model with pretrained weights
            model = YOLO(self.model_trainer_config.weight_name)
            
            # Train the model with the updated data.yaml
            logging.info(f"Starting model training with {self.model_trainer_config.no_epochs} epochs")
            results = model.train(
                data=training_yaml_path,
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                imgsz=self.model_trainer_config.image_size,
                name="vehicle_detection_results",
                verbose=True
            )
            
            # Copy best model to model trainer directory
            best_model_path = os.path.join("runs", "detect", "vehicle_detection_results", "weights", "best.pt")
            trained_model_path = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            
            # Use shutil instead of os.system for file operations
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, trained_model_path)
                logging.info(f"Best model saved to {trained_model_path}")
            else:
                logging.warning(f"Best model not found at {best_model_path}")
            
            # Clean up
            if os.path.exists("runs"):
                shutil.rmtree("runs")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_path,
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)




            