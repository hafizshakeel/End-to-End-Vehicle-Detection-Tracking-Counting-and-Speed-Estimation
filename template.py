import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = 'traffic_vision' # Vehile detection, tracking, counting, and speed estimation

list_of_files = [
    '.github/workflows/.gitkeep',
    'data/.gitkeep',
    f'{project_name}/__init__.py',
    f'{project_name}/components/__init__.py',
    f'{project_name}/components/data_ingestion.py',
    f'{project_name}/components/data_validation.py',
    f'{project_name}/components/model_trainer.py',
    f'{project_name}/constants/__init__.py',
    f'{project_name}/constants/training_pipeline/__init__.py',
    f'{project_name}/constants/application.py',
    f'{project_name}/entity/config_entity.py',
    f'{project_name}/entity/artifacts_entity.py',
    f'{project_name}/exception/__init__.py',
    f'{project_name}/logger/__init__.py',
    f'{project_name}/pipeline/__init__.py',
    f'{project_name}/pipeline/training_pipeline.py',
    f'{project_name}/utils/__init__.py',
    f'{project_name}/utils/main_utils.py',
    'weights/model_epoch.pt',
    'notes/train_model_gc.ipynb',
    'app.py',
    'Dockerfile',
    'requirements.txt',
    'setup.py',
]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Created directory: {filedir} for file: {filename}')

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass  
        logging.info(f'Created new file: {filepath}')
    elif os.path.getsize(filepath) == 0:
        logging.info(f'File: {filepath} already exists but is empty.')
    else:
        logging.info(f'File: {filepath} already exists with content. Skipping.')


logging.info('Project structure created successfully')