"""
Student Performance Predictor - Enhanced Web Application
=========================================================
Features:
- Interactive prediction with probability percentages
- Prediction history stored locally
- Beautiful charts and visualizations
- Input validation
- Error handling
"""

from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional
import os
import json
from datetime import datetime
from bson import ObjectId
import ast
import pandas as pd
import pickle
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import StudentData, StudentPerformanceClassifier
from src.pipline.training_pipeline import TrainPipeline
from src.logger import logging
from src.data_access.mongodb_handler import get_mongodb_handler

# Initialize FastAPI application
app = FastAPI(
    title="Student Performance Predictor",
    description="ML-Based Student Performance Prediction with Interactive UI",
    version="2.0.0"
)

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# History file path
HISTORY_FILE = "prediction_history.json"


def convert_to_serializable(obj):
    """Convert objects that are not JSON serializable to serializable types."""
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def load_history():
    """Load prediction history - tries MongoDB first, falls back to file"""
    try:
        # Try MongoDB first
        mongodb = get_mongodb_handler()
        if mongodb.is_connected():
            history = mongodb.get_prediction_history(limit=50)
            if history:
                # Format for display - convert all non-serializable objects
                return convert_to_serializable(history)

        # Fall back to local file
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                return convert_to_serializable(history)
    except Exception as e:
        logging.error(f"Error loading history: {e}")
    return []


def save_history(history):
    """Save prediction history to JSON file"""
    # Ensure all objects are JSON serializable
    serializable_history = convert_to_serializable(history)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(serializable_history, f, indent=2)


def add_to_history(input_data, prediction, probability):
    """Add prediction to history - tries MongoDB first, falls back to file"""
    # Try MongoDB first
    mongodb = get_mongodb_handler()
    if mongodb.is_connected():
        entry = {
            "input": input_data,
            "prediction": prediction,
            "probability": probability
        }
        mongodb.save_prediction(entry)
        # Also save locally for backup
        save_history_local(entry)
    else:
        # Fall back to local file
        save_history_local({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": input_data,
            "prediction": prediction,
            "probability": probability
        })


def save_history_local(entry):
    """Save prediction to local JSON file (fallback)"""
    # Ensure entry is a dict
    if isinstance(entry, dict):
        entry_with_ts = entry.copy()
    else:
        entry_with_ts = {"input": entry}

    if "timestamp" not in entry_with_ts:
        entry_with_ts["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load history and ensure it's a list
    history = load_history()
    if not isinstance(history, list):
        history = []

    history.insert(0, entry_with_ts)
    history = history[:50]
    save_history(history)


class DataForm:
    """DataForm class to handle and process incoming form data."""
    def __init__(self, request: Request):
        self.request: Request = request
        self.Study_Hours: Optional[str] = None
        self.Sleep_Hours: Optional[str] = None
        self.Attendance_Percentage: Optional[str] = None
        self.Previous_Score: Optional[str] = None
        self.Internet_Usage: Optional[str] = None
        self.Social_Activity_Level: Optional[str] = None

    async def get_student_data(self):
        """Method to retrieve and assign form data to class attributes."""
        form = await self.request.form()
        self.Study_Hours = str(form.get("Study_Hours") or "")
        self.Sleep_Hours = str(form.get("Sleep_Hours") or "")
        self.Attendance_Percentage = str(form.get("Attendance_Percentage") or "")
        self.Previous_Score = str(form.get("Previous_Score") or "")
        self.Internet_Usage = str(form.get("Internet_Usage") or "")
        self.Social_Activity_Level = str(form.get("Social_Activity_Level") or "")

    def validate(self) -> tuple:
        """Validate input data and return (is_valid, error_message)"""
        errors = []

        # Study Hours: 0-24
        if self.Study_Hours is None or self.Study_Hours == '':
            errors.append("Study Hours is required")
        else:
            try:
                sh = float(self.Study_Hours)
                if sh < 0 or sh > 24:
                    errors.append("Study Hours must be between 0 and 24")
            except:
                errors.append("Study Hours must be a number")

        # Sleep Hours: 0-24
        if self.Sleep_Hours is None or self.Sleep_Hours == '':
            errors.append("Sleep Hours is required")
        else:
            try:
                sl = float(self.Sleep_Hours)
                if sl < 0 or sl > 24:
                    errors.append("Sleep Hours must be between 0 and 24")
            except:
                errors.append("Sleep Hours must be a number")

        # Attendance: 0-100
        if self.Attendance_Percentage is None or self.Attendance_Percentage == '':
            errors.append("Attendance Percentage is required")
        else:
            try:
                att = float(self.Attendance_Percentage)
                if att < 0 or att > 100:
                    errors.append("Attendance Percentage must be between 0 and 100")
            except:
                errors.append("Attendance Percentage must be a number")

        # Previous Score: 0-100
        if self.Previous_Score is None or self.Previous_Score == '':
            errors.append("Previous Score is required")
        else:
            try:
                ps = float(self.Previous_Score)
                if ps < 0 or ps > 100:
                    errors.append("Previous Score must be between 0 and 100")
            except:
                errors.append("Previous Score must be a number")

        # Internet Usage: 0-24
        if self.Internet_Usage is None or self.Internet_Usage == '':
            errors.append("Internet Usage is required")
        else:
            try:
                iu = float(self.Internet_Usage)
                if iu < 0 or iu > 24:
                    errors.append("Internet Usage must be between 0 and 24")
            except:
                errors.append("Internet Usage must be a number")

        # Social Activity: 1-5
        if self.Social_Activity_Level is None or self.Social_Activity_Level == '':
            errors.append("Social Activity Level is required")
        else:
            try:
                sal = int(self.Social_Activity_Level)
                if sal < 1 or sal > 5:
                    errors.append("Social Activity Level must be between 1 and 5")
            except:
                errors.append("Social Activity Level must be an integer (1-5)")

        if errors:
            return False, "; ".join(errors)
        return True, ""


# Route to render the main page with the form
@app.get("/", tags=["prediction"])
async def index(request: Request):
    """Renders the main HTML form page for student data input."""

    history = load_history()
    return templates.TemplateResponse(
        "studentdata.html",
        {
            "request": request,
            "context": "Enter student details to predict performance",
            "history": history,
            "chart_data": json.dumps(history[:10]) if history else "[]"
        }
    )


# Route to trigger the model training process
@app.get("/train", tags=["training"])
async def trainRouteClient():
    """Endpoint to initiate the model training pipeline."""
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return PlainTextResponse("Training successful!!!")
    except Exception as e:
        return PlainTextResponse(f"Error Occurred! {e}")


# Route to handle form submission and make predictions
@app.post("/predict", tags=["prediction"])
async def predictRouteClient(request: Request):
    """Endpoint to receive form data, process it, and make a prediction."""
    try:
        form = DataForm(request)
        await form.get_student_data()

        # Validate input
        is_valid, error_msg = form.validate()
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"status": False, "error": error_msg}
            )

        # Convert to float/int with proper handling
        study_hours = float(form.Study_Hours) if form.Study_Hours else 0.0
        sleep_hours = float(form.Sleep_Hours) if form.Sleep_Hours else 0.0
        attendance = float(form.Attendance_Percentage) if form.Attendance_Percentage else 0.0
        previous_score = float(form.Previous_Score) if form.Previous_Score else 0.0
        internet_usage = float(form.Internet_Usage) if form.Internet_Usage else 0.0
        social_activity = int(form.Social_Activity_Level) if form.Social_Activity_Level else 1

        student_data = StudentData(
            Study_Hours=study_hours,
            Sleep_Hours=sleep_hours,
            Attendance_Percentage=attendance,
            Previous_Score=previous_score,
            Internet_Usage=internet_usage,
            Social_Activity_Level=social_activity
        )

        # Convert form data into a DataFrame for the model
        student_df = student_data.get_student_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = StudentPerformanceClassifier()

        # Make a prediction
        result = model_predictor.predict(dataframe=student_df)

        # Get probability prediction
        probability = model_predictor.predict_proba(dataframe=student_df)

        # Prepare response data
        prediction_result = result[0] if isinstance(result, (list, tuple)) else result
        prob_pass = float(probability.get('Pass', 0)) if isinstance(probability, dict) else float(probability[0][1]) if len(probability) > 0 else 0
        prob_fail = float(probability.get('Fail', 0)) if isinstance(probability, dict) else float(probability[0][0]) if len(probability) > 0 else 0

        # Format the prediction for display
        status = "PASS" if prediction_result == 1 else "FAIL"
        confidence = max(prob_pass, prob_fail) * 100

        # Add to history
        input_data = {
            "Study_Hours": study_hours,
            "Sleep_Hours": sleep_hours,
            "Attendance_Percentage": attendance,
            "Previous_Score": previous_score,
            "Internet_Usage": internet_usage,
            "Social_Activity_Level": social_activity
        }

        add_to_history(
            input_data=input_data,
            prediction=status,
            probability=f"{confidence:.1f}%"
        )

        # Return JSON response
        return JSONResponse(content={
            "status": True,
            "prediction": status,
            "probability": {
                "pass": f"{prob_pass * 100:.1f}%",
                "fail": f"{prob_fail * 100:.1f}%"
            },
            "confidence": f"{confidence:.1f}%"
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": False, "error": str(e)}
        )


# API endpoint for history
@app.get("/history", tags=["history"])
async def get_history():
    """Get prediction history"""
    return JSONResponse(content=load_history())


# API endpoint for model status
@app.get("/model/status", tags=["model"])
async def model_status():
    """Check if model is loaded"""
    model_path = "artifact/model_trainer/trained_model/model.pkl"
    exists = os.path.exists(model_path)
    return JSONResponse(content={
        "model_loaded": exists,
        "model_path": model_path
    })


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)