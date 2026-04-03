# Student Performance Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" />
  <img src="https://img.shields.io/badge/Database-MongoDB-green.svg" />
  <img src="https://img.shields.io/badge/Cloud-Railway-orange.svg" />
  <img src="https://img.shields.io/badge/Container-Docker-blue.svg" />
  <img src="https://img.shields.io/badge/CI/CD-GitHub%20Actions-black.svg" />
</p>

> An **end-to-end Machine Learning project** for predicting student academic performance. Built with **MLOps best practices**.

---

## Features

- Interactive Web Application with prediction probability charts
- Prediction History tracking (last 50 predictions)
- Input Validation with error messages
- Natural Data Generator (15,000+ realistic student records)
- Complete ML Pipeline (Ingestion → Validation → Transformation → Training → Evaluation)
- Docker Containerization
- CI/CD with GitHub Actions
- AWS S3 Integration for model storage

---

## Tech Stack

| Layer | Tools & Services |
| ------------------- | ---------------------- |
| **Language** | Python (3.10) |
| **Web Framework** | FastAPI |
| **ML Framework** | Scikit-learn |
| **Imbalanced Data** | SMOTEENN (imblearn) |
| **Database** | MongoDB Atlas (optional) |
| **Cloud** | AWS S3 |
| **Container** | Docker, docker-compose |
| **CI/CD** | GitHub Actions |

---

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd student_performance_predictor

# Create virtual environment
conda create -n student_perf python=3.10 -y
conda activate student_perf
pip install -r requirements.txt
```

### 2. Generate Data (Already Done)

Data has been pre-generated in the `data/` folder:
- `data/train.csv` - 12,006 training records
- `data/test.csv` - 3,001 test records

To regenerate:
```bash
python generate_data.py
```

### 3. Train the Model

```bash
python app.py
# Then open http://localhost:5000/train
```

Or run directly:
```bash
python -c "from src.pipline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"
```

### 4. Run the Application

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t student-performance-predictor .

# Run the container
docker run -p 5000:5000 student-performance-predictor
```

### Using docker-compose

```bash
# Create .env file from example
cp .env.example .env

# Edit .env with your credentials

# Run all services
docker-compose up --build
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with prediction form |
| `/predict` | POST | Make prediction with probabilities |
| `/history` | GET | Get prediction history |
| `/train` | GET | Train the model |
| `/health` | GET | Health check |
| `/model/status` | GET | Check model status |

---

## Input Features

| Feature | Range | Description |
| -------- | ------- | ----------- |
| Study_Hours | 0-24 | Daily study hours |
| Sleep_Hours | 0-24 | Daily sleep hours |
| Attendance_Percentage | 0-100 | Class attendance % |
| Previous_Score | 0-100 | Previous exam score % |
| Internet_Usage | 0-24 | Daily internet hours |
| Social_Activity_Level | 1-5 | Social activity scale |

---

## Output

**Prediction**: PASS or FAIL (threshold: 40%)

**Probability**: Percentage confidence for each outcome

---

## Project Structure

```
student_performance_predictor/
├── app.py                      # FastAPI application
├── generate_data.py            # Natural data generator
├── src/
│   ├── components/              # ML pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/           # Database & Cloud configs
│   ├── cloud_storage/           # AWS S3 operations
│   ├── data_access/             # MongoDB data export
│   ├── entity/                  # Config & Artifact classes
│   ├── pipline/                 # Training & Prediction pipelines
│   ├── utils/                   # Utility functions
│   ├── constants/               # Project constants
│   ├── exception/               # Custom exceptions
│   └── logger/                  # Logging configuration
├── data/                       # Generated data (15K+ records)
├── config/                     # YAML configuration files
├── templates/                  # HTML templates
├── static/                     # CSS/JS assets
├── docker-compose.yml          # Docker compose config
├── .github/workflows/          # CI/CD pipeline
└── .env.example               # Environment variables template
```

---

## Environment Variables

```bash
# MongoDB (optional - will use local CSV if not set)
export MONGODB_URL="your_mongodb_connection_string"

# AWS (optional - will use local model if not set)
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

---

## CI/CD Pipeline

The project includes GitHub Actions workflow that:
1. Lints code (flake8, black, isort)
2. Runs basic functionality tests
3. Builds Docker image
4. (Optional) Deploys to AWS ECS

---

## License

MIT License
