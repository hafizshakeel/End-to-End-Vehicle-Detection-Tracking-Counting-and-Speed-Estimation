
# 🚗 End-to-End Vehicle Detection, Tracking, Counting, and Speed Estimation 

A comprehensive computer vision system for vehicle analysis with full MLOps integration. Combines state-of-the-art detection (YOLOv8), tracking (ByteTrack), and speed estimation in a production-ready pipeline.



## 📌 Features

### 🖥️ Core Capabilities
- 🎯 **Vehicle Detection**: YOLOv8-based detection with 95%+ accuracy
- 📍 **Tracking & Counting**: ByteTrack implementation with <50ms latency
- 📊 **Speed Estimation**: Perspective transformation with ±5% error margin
- 🌐 **Real-time Processing**: 30 FPS on RTX 3060 GPU
- 🎮 **Custom ROI Selection**: Interactive region-of-interest definition

### 🛠️ MLOps Features
- 🔄 **CI/CD Pipeline**: GitHub Actions automation
- 🐳 **Dockerized**: Containerized deployment
- ☁️ **AWS Integration**: ECR/EC2 deployment ready
- 📈 **ML Pipeline**: Training/evaluation workflow
- 📊 **Model Management**: Weight versioning and storage

### 📊 Web Interface
- 🖼️ Interactive video processing controls
- 📉 Real-time analytics dashboard
- 📥 Results export (CSV/Video)
- 📷 IP camera/RTSP stream support


## 🚀 Quick Start

### Local Installation
```bash
git clone https://github.com/hafizshakeel/End-to-End-Vehicle-Detection-Tracking-Counting-and-Speed-Estimation.git
cd End-to-End-Vehicle-Detection-Tracking-Counting-and-Speed-Estimation
pip install -r requirements.txt
```

### Run Processing Applications
```bash
# For video file processing
streamlit run app.py

# For real-time camera input
streamlit run live_camera.py
```

### Docker Deployment
```bash
docker build -t vehicle-tracking .
docker run -p 8501:8501 vehicle-tracking
```
Access at: http://localhost:8501

## ☁️ AWS Deployment Guide

<details>
<summary><strong>🔐 IAM Configuration</strong></summary>

1. Create IAM user with:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`
2. Store credentials in GitHub Secrets:
   ```env
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_REGION=us-east-1
   ```
</details>

<details>
<summary><strong>📦 ECR Setup</strong></summary>

```bash
aws ecr create-repository --repository-name vehicle-tracking --region us-east-1
aws ecr get-login-password | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com
```
</details>

<details>
<summary><strong>🖥 EC2 Configuration</strong></summary>

```bash
# Install Docker on Ubuntu
sudo apt-get update && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```
</details>

## 🗂️ Project Structure

```
📁 END-TO-END-VEHICLE-DETECTION-TRACKING-COUNTING-AND-SPEED-ESTIMATION
├── 📁 .github/workflows       # CI/CD configurations
├── 📁 data                   # Sample datasets
├── 📁 traffic_vision         # MLOps pipeline
│   ├── 📁 components        # Pipeline stages
│   ├── 📁 constants         # Configuration constants
│   ├── 📁 entity            # Data entities
│   ├── 📁 exception         # Custom exceptions
│   ├── 📁 logger            # Logging configuration
│   ├── 📁 pipeline          # Training pipelines
│   └── 📁 utils             # Utility functions
├── 📁 vehicle_tracker       # Core application logic
│   ├── 📁 config           # Runtime configurations
│   ├── 📁 core             # Detection/tracking implementation
│   ├── 📁 models           # Model definitions
│   ├── 📁 utils            # Helper functions
│   └── 📁 visualization    # Visualization components
├── 📄 app.py               # Main Streamlit application
├── 📄 live_camera.py       # Real-time processing script
├── 📄 Dockerfile           # Container configuration
└── 📄 requirements.txt     # Python dependencies
```


## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

- [YOLOv8](https://ultralytics.com/yolov8) by Ultralytics
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for object tracking
- [Streamlit](https://streamlit.io) for web interface
- AWS for cloud infrastructure
- Supervision library for CV utilities

