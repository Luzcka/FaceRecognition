# Facial Recognition System - API

A high-performance facial recognition API built with DeepFace and Milvus for similarity search operations.

## Overview

This system provides a REST API for facial recognition with user registration and search capabilities. It uses state-of-the-art deep learning models for face embedding extraction and vector similarity search for fast retrieval.

## Key Features

- Face embedding extraction using DeepFace with Facenet512 model
- Vector similarity search with Milvus database
- REST API with FastAPI framework
- Bearer token authentication
- Comprehensive logging and error handling
- Docker containerization support
- Configurable similarity thresholds

## Technical Specifications

- **Face Model**: Facenet512 (512-dimensional embeddings)
- **Distance Metric**: Cosine similarity
- **Vector Database**: Milvus v2.5.x
- **Web Framework**: FastAPI
- **Authentication**: Bearer token
- **Image Processing**: DeepFace and OpenCV detector backend

## Project Structure

```
facial-recognition-api/
├── api/
│   ├── dependencies.py      # Dependency injection
│   └── endpoints/
│       ├── register.py      # User registration endpoint
│       └── search.py        # Face search endpoint
├── core/
│   ├── config.py           # Application configuration
│   └── security.py         # Authentication and security
├── models/
│   └── user.py             # Pydantic data models
├── services/
│   ├── face_service.py     # Facial recognition service
│   └── milvus_service.py   # Vector database service
├── .env                    # Environment variables
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── docker-compose.yml     # Container orchestration
```

## Installation

### Method 1: Conda Environment

```bash
# Create conda environment
conda create -n facial-recognition python=3.9
conda activate facial-recognition

# Install dependencies
pip install -r requirements.txt

# Start Milvus

# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables (.env)

```env
# API Configuration
API_KEY=your-api-key-here
SECRET_KEY=your-secret-key-here

# Milvus Configuration
MILVUS_MODE=remote
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_LOCAL_PATH=data/milvus_faces.db

# Face Recognition Configuration
FACE_MODEL=Facenet512
FACE_DETECTOR=opencv
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=5

# Logging
LOG_LEVEL=INFO
```

### Milvus Setup

For local development, you can run Milvus with Docker:

#### Method 1
```bash
# Use the docker compose located in: milvus_docker_compose 

# Create folder for docker compose
mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

# Copy the docker-compose.yml file to the created directory

sudo docker compose up -d

# Set proper ownership for volumes
sudo chown -R $USER:$USER volumes
sudo chown -R 1000:1000 volumes/milvus volumes/minio volumes/etcd
```

#### Method 2
```bash
# Download and run Milvus standalone
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

## API Usage

### Authentication

All endpoints require Bearer token authentication:

```
Authorization: Bearer your-api-key-here
```

### Register User

```bash
curl -X POST "http://localhost:8000/api/v1/register" \
  -H "Authorization: Bearer your-api-key-here" \
  -F "name=John Doe" \
  -F "registration_number=EMP001" \
  -F "image=@person.jpg"
```

### Search for User

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Authorization: Bearer your-api-key-here" \
  -F "image=@search.jpg"
```

### Health Check

```bash
curl http://localhost:8000/health
```

## API Response Examples

### Successful Search

```json
[
  {
    "name": "John Doe",
    "registration_number": "EMP001",
    "similarity_score": 0.987,
    "distance": 0.013
  },
  {
    "name": "Jane Smith",
    "registration_number": "EMP002",
    "similarity_score": 0.956,
    "distance": 0.044
  }
]
```

### Successful Registration

```json
{
  "message": "User registered successfully",
  "name": "John Doe",
  "registration_number": "EMP001"
}
```

## Development

### Logging System

Logs are written to the `backend/app/logs/` directory. Configure log level in `.env`:

- DEBUG: Detailed information for debugging
- INFO: General information about system operation
- WARN: Warning messages
- ERROR: Error conditions
- CRITICAL: Critical error conditions

## Troubleshooting

### Common Issues

1. **Milvus Connection Error**: Ensure Milvus server is running and accessible
2. **Face Not Detected**: Check image quality and lighting conditions
3. **Low Similarity Scores**: Adjust SIMILARITY_THRESHOLD in configuration
4. **Memory Issues**: Monitor Docker container resources

### Debug Information

Access debug endpoint for system information:

```bash
curl -H "Authorization: Bearer your-api-key-here" \
  http://localhost:8000/debug/info
```

## Requirements

- Python 3.11+
- Docker (optional)
- Milvus v2.5.16
- Minimum 4GB RAM
- GPU support (optional, for faster processing)

## License

Use only under authorization