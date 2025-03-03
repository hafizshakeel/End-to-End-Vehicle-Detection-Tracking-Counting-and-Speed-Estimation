#  End to End Vehicle Detection, Tracking, Counting, and Speed Estimation

This project implements a comprehensive vehicle analysis system using computer vision and deep learning. It provides capabilities for vehicle detection, tracking, counting, and speed estimation, along with a complete MLOps pipeline for model training and deployment.

## Features

- **Vehicle Detection**: Accurately identify vehicles in videos using YOLOv8
- **Vehicle Tracking**: Track vehicles across frames using ByteTrack
- **Vehicle Counting**: Count vehicles crossing a defined line or region
- **Speed Estimation**: Estimate vehicle speeds using perspective transformation
- **Custom Region of Interest (ROI)**: Select specific areas for analysis
- **Real-time Processing**: Process live video feeds from webcams or IP cameras
- **Docker containerization**: Easy deployment with Docker
- **CI/CD pipeline**: Automated deployment to AWS
- **MLOps Pipeline**: Complete pipeline for data ingestion, validation, model training, and evaluation

## Streamlit Web Application

The project includes a user-friendly Streamlit web application that provides:

1. **Intuitive User Interface**:
   - Upload and process videos
   - Interactive configuration options for detection and tracking
   - Real-time progress tracking during video processing

2. **Enhanced Visualization**:
   - Custom ROI selection directly on the video frame
   - Comprehensive statistics and visualizations
   - Vehicle counts and speed distribution
   - Downloadable results (processed video and CSV data)

3. **Real-time Processing**:
   - Connect to webcams or IP cameras
   - Process live video feeds in real-time
   - View statistics and analytics as they happen
   - Record processed video for later analysis

## Architecture

The project is structured with a modular architecture:

1. **Core Vehicle Tracking Module** (`vehicle_tracker/`):
   - `detector.py`: YOLOv8-based vehicle detection
   - `tracker.py`: ByteTrack implementation for vehicle tracking
   - `counter.py`: Vehicle counting logic
   - `speed_estimator.py`: Speed estimation using perspective transformation

2. **MLOps Pipeline** (`traffic_vision/`):
   - Data ingestion and validation
   - Model training and evaluation
   - Pipeline orchestration
   - Logging and exception handling

3. **Web Applications**:
   - `app.py`: Streamlit application for video processing
   - `live_camera.py`: Real-time processing with camera input

4. **CI/CD Pipeline** (`.github/workflows/`):
   - Automated testing and linting
   - Docker image building and pushing to ECR
   - Deployment to EC2 instances

## Usage

### Video Processing Application

To run the video processing application:

```bash
streamlit run app.py
```

### Real-time Processing Application

To run the real-time processing application with camera input:

```bash
streamlit run live_camera.py
```

### Docker Deployment

Build the Docker image:

```bash
docker build -t vehicle-tracking .
```

Run the container:

```bash
docker run -p 8501:8501 vehicle-tracking
```

Access the application at: http://localhost:8501

## CI/CD Pipeline

This project includes a GitHub Actions workflow for continuous integration and deployment to AWS:

1. **Continuous Integration**:
   - Code checkout
   - Linting and testing
   - Build validation

2. **Continuous Delivery**:
   - Building Docker image
   - Pushing to Amazon ECR

3. **Continuous Deployment**:
   - Deploying to AWS infrastructure
   - Health checks and validation

## AWS CI/CD Pipeline ☁️

### Infrastructure Setup
1. **IAM Configuration**:
   - Create user with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`
   - Store credentials in GitHub Secrets:
     ```
     AWS_ACCESS_KEY_ID
     AWS_SECRET_ACCESS_KEY
     AWS_REGION
     AWS_ECR_LOGIN_URI
     ECR_REPOSITORY_NAME
     ```

2. **ECR Setup**:
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name vehicle-tracking --region us-east-1
   
   # Get repository URI (save this)
   aws ecr describe-repositories --repository-names vehicle-tracking --query 'repositories[0].repositoryUri'
   ```

3. **EC2 Instance**:
   - Ubuntu t2.large or t2.xlarge (32GB+ storage)
   - Install Docker:
     ```bash
     sudo apt-get update && sudo apt-get upgrade -y
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     sudo usermod -aG docker ubuntu
     newgrp docker
     ```

4. **Self-Hosted Runner**:
   - Configure from GitHub Settings > Actions > Runners
   - Follow the instructions to set up a self-hosted runner on your EC2 instance

## Project Structure

```
├── app.py                  # Streamlit application for video processing
├── live_camera.py          # Streamlit application for real-time processing
├── main.py                 # Core vehicle tracking implementation
├── train.py                # Script to run the training pipeline
├── zone.py                 # Zone definition for vehicle counting
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── .github/workflows       # CI/CD configuration
├── traffic_vision/         # MLOps pipeline components
│   ├── components/         # Pipeline components
│   ├── config/             # Configuration
│   ├── constants/          # Constants
│   ├── entity/             # Entity definitions
│   ├── exception/          # Custom exceptions
│   ├── logger/             # Logging
│   ├── pipeline/           # Pipeline definitions
│   └── utils/              # Utility functions
└── vehicle_tracker/        # Core tracking functionality
    ├── core/               # Core components
    │   ├── detector.py     # Vehicle detection
    │   ├── tracker.py      # Vehicle tracking
    │   ├── counter.py      # Vehicle counting
    │   └── speed_estimator.py # Speed estimation
    ├── config/             # Configuration
    ├── models/             # Model definitions
    ├── utils/              # Utility functions
    └── visualization/      # Visualization components
```

## MLOps Pipeline

The project includes a complete MLOps pipeline with the following components:

1. **Data Ingestion**: Import and prepare data for training
2. **Data Validation**: Validate data quality and structure
3. **Model Training**: Train YOLOv8 models on vehicle data
4. **Model Evaluation**: Evaluate model performance and save metrics

To run the complete pipeline:

```bash
python train.py
```

## Requirements

- Python 3.9+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Supervision
- Streamlit
- Pandas
- NumPy
- Plotly

<!-- ## Future Improvements

Here are some areas for future development:

1. **Model Improvements**:
   - Fine-tune YOLOv8 on a larger vehicle dataset
   - Implement vehicle classification (car, truck, bus, motorcycle)
   - Add vehicle color and make/model recognition

2. **Tracking Enhancements**:
   - Improve tracking stability in crowded scenes
   - Implement occlusion handling
   - Add trajectory prediction

3. **Speed Estimation**:
   - Improve accuracy with better camera calibration
   - Add automatic camera calibration
   - Implement speed estimation without perspective transformation

4. **Application Features**:
   - Add user authentication
   - Implement database storage for analytics
   - Create dashboard for historical data analysis
   - Add alert system for speeding vehicles

5. **Infrastructure**:
   - Implement model serving with TorchServe or ONNX Runtime
   - Add GPU support for faster inference
   - Implement streaming architecture for distributed processing
   - Add monitoring and alerting for the deployed application

6. **Testing**:
   - Add comprehensive unit tests
   - Implement integration tests
   - Add performance benchmarks -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Supervision library for computer vision
- ByteTrack for object tracking
- Streamlit for the web application